"""Document processing pipeline."""

import asyncio
from dataclasses import dataclass
import hashlib
import os
from typing import AsyncIterator, List, Optional, Protocol, Tuple

import asyncpg
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
from pyspark.storagelevel import StorageLevel

from ..resolvers import Resolver
from .aws import AWSCredentials, create_aws_client
from .entity_reltype_processing import (
    create_extracted_entities_df,
    create_extracted_rel_types_df,
    link_relationships_with_references,
    save_relationships_to_db,
)
from .file_processor import create_relationship_extractor_udf
from .models import Entity, Relationship, RelationshipType
from .s3 import DocumentBatch, S3DocumentReader
from .schemas import RELATIONSHIP_SCHEMA


def generate_hash(name: str, description: str) -> str:
    """Generate a unique hash for the entity or relationship type."""
    return hashlib.md5(f"{name}:{description}".encode()).hexdigest()


class DocumentProcessor(Protocol):
    """Protocol for document processors that extract information."""

    async def process(self, df: DataFrame) -> DataFrame:
        """Process a DataFrame of documents."""
        ...


@dataclass
class RelationshipExtractorProcessor:
    """Processes documents to extract relationships using an LLM."""

    model: str = "gemini-2.0-flash"

    def create_udf(self):
        """Create Spark UDF for relationship extraction."""
        return create_relationship_extractor_udf(self.model)

    async def process(self, df: DataFrame) -> DataFrame:
        """Extract relationships from document content."""
        # Create UDF for extraction
        extract_rels_udf = self.create_udf()

        # https://community.databricks.com/t5/data-engineering/accelerating-row-wise-python-udf-functions-without-using-pandas/td-p/15328
        df = df.repartition(os.cpu_count())
        # Extract relationships and cache BEFORE any transformations
        # (to avoid UDFs being run multiple times)
        df_with_json = df.withColumn(
            "relationships_json",
            extract_rels_udf(F.col("content")),
        ).persist(StorageLevel.MEMORY_AND_DISK)

        await asyncio.to_thread(df_with_json.count)

        # Now do the JSON parsing
        df_with_relationships = df_with_json.withColumn(
            "extracted_relationships",
            F.from_json(
                F.col("relationships_json"),
                ArrayType(RELATIONSHIP_SCHEMA),
            ),
        )

        # Explode the extracted relationships into separate rows
        exploded_df = df_with_relationships.select(
            "uri",  # Keep the document URI
            F.explode("extracted_relationships").alias(
                "relationship"
            ),  # Explode the array
        )

        extracted_relationships_df = exploded_df.select(
            "relationship." + Relationship.FROM_ENTITY_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_TYPE_COLUMN,
            "relationship." + Relationship.TO_ENTITY_COLUMN,
            "relationship." + Relationship.CONSTRAINT_CONDITION_COLUMN,
            "relationship." + Relationship.REASON_COLUMN,
            "relationship." + Relationship.IS_CAUSAL_COLUMN,
            "relationship." + Relationship.FROM_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.TO_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_DESCRIPTION_COLUMN,
            F.col("uri").alias(Relationship.SOURCE_DOCUMENT_URI_COLUMN),
        )

        # Add hash columns for deduplication
        extracted_relationships_df = (
            extracted_relationships_df.withColumn(
                Relationship.FROM_ENTITY_HASH_COLUMN,
                F.md5(
                    F.concat_ws(
                        ": ",
                        F.col(Relationship.FROM_ENTITY_COLUMN),
                        F.col(Relationship.FROM_ENTITY_DESCRIPTION_COLUMN),
                    )
                ),
            )
            .withColumn(
                Relationship.TO_ENTITY_HASH_COLUMN,
                F.md5(
                    F.concat_ws(
                        ": ",
                        F.col(Relationship.TO_ENTITY_COLUMN),
                        F.col(Relationship.TO_ENTITY_DESCRIPTION_COLUMN),
                    )
                ),
            )
            .withColumn(
                Relationship.RELATIONSHIP_TYPE_HASH_COLUMN,
                F.md5(
                    F.concat_ws(
                        ": ",
                        F.col(Relationship.RELATIONSHIP_TYPE_COLUMN),
                        F.col(Relationship.RELATIONSHIP_DESCRIPTION_COLUMN),
                    )
                ),
            )
        )

        return extracted_relationships_df


class Pipeline:
    """Document processing pipeline."""

    def __init__(
        self,
        spark: SparkSession,
        credentials: Optional[AWSCredentials] = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize pipeline."""
        self.spark = spark

        # Initialize S3 reader
        s3_client = create_aws_client("s3", credentials)
        self.reader = S3DocumentReader(s3_client, credentials)

        # Initialize processors
        self.processors = [RelationshipExtractorProcessor(model=model)]

    async def process_documents(
        self,
        base_uri: str,
    ) -> AsyncIterator[DataFrame]:
        """
        Process documents through the pipeline.

        Args:
            base_uri: Base S3 URI to process

        Yields:
            Processed DataFrames
        """
        async for batch in self.reader.read_documents(base_uri):
            # Convert batch to DataFrame
            df = self._batch_to_dataframe(batch)

            # Deduplicate
            df = df.dropDuplicates(["content"])

            # Run through processors
            for processor in self.processors:
                df = await processor.process(df)

            yield df

    def _batch_to_dataframe(self, batch: DocumentBatch) -> DataFrame:
        """Convert document batch to Spark DataFrame."""
        rows = [(doc.uri, doc.content, doc.file_type) for doc in batch.documents]
        return self.spark.createDataFrame(
            rows,
            ["uri", "content", "file_type"],
        )

    async def process_and_store_relationships(
        self,
        extracted_relationships_df: pd.DataFrame,
        embedding_provider,
        entity_resolver: Resolver[Entity],
        rel_type_resolver: Resolver[RelationshipType],
        conn: asyncpg.Connection,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Process extracted relationships, resolve entities and relationship types,
        and store everything in the database.

        Args:
            extracted_relationships_df: DataFrame containing extracted relationships
            embedding_provider: Provider for computing embeddings
            entity_resolver: Resolver for entities
            rel_type_resolver: Resolver for relationship types
            conn: asyncpg connection

        Returns:
            Tuple containing:
            - DataFrame with relationships linked to database references
            - List of database IDs for the saved relationships
        """
        # Process entities and get hash-to-reference mapping
        _, entity_hash_to_reference = await create_extracted_entities_df(
            extracted_relationships_df,
            embedding_provider,
            entity_resolver,
        )

        # Process relationship types and get hash-to-reference mapping
        _, rel_type_hash_to_reference = await create_extracted_rel_types_df(
            extracted_relationships_df,
            embedding_provider,
            rel_type_resolver,
        )

        # Associate relationships with entity and relationship type references
        relationships_with_refs = await link_relationships_with_references(
            extracted_relationships_df,
            entity_hash_to_reference,
            rel_type_hash_to_reference,
        )

        # Save relationships to database
        relationship_ids = await save_relationships_to_db(
            relationships_with_refs,
            conn,
        )

        return relationships_with_refs, relationship_ids
