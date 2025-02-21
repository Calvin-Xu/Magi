"""Document processing pipeline."""

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
import asyncio
from pyspark.storagelevel import StorageLevel
import pandas as pd

from .aws import AWSCredentials, create_aws_client
from .s3 import DocumentBatch, S3DocumentReader
from .schemas import RELATIONSHIP_SCHEMA
from .file_processor import create_relationship_extractor_udf
from .models import Relationship

import hashlib


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
                "from_entity_hash",
                F.md5(
                    F.concat_ws(
                        ":",
                        F.col(Relationship.FROM_ENTITY_COLUMN),
                        F.col(Relationship.FROM_ENTITY_DESCRIPTION_COLUMN),
                    )
                ),
            )
            .withColumn(
                "to_entity_hash",
                F.md5(
                    F.concat_ws(
                        ":",
                        F.col(Relationship.TO_ENTITY_COLUMN),
                        F.col(Relationship.TO_ENTITY_DESCRIPTION_COLUMN),
                    )
                ),
            )
            .withColumn(
                "relationship_type_hash",
                F.md5(
                    F.concat_ws(
                        ":",
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
        credentials: Optional[AWSCredentials] = None,
        spark: Optional[SparkSession] = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize pipeline."""
        # Initialize Spark
        self.spark = spark or SparkSession.builder.appName("magi").getOrCreate()

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

    async def associate_relationships(
        extracted_relationships_df: pd.DataFrame,
        entities_df: pd.DataFrame,
        rel_types_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Associate relationships with their corresponding entities and relationship types."""
        # Create a mapping from hash to entity and relationship type
        entity_hash_map = {
            generate_hash(entity.name, entity.description): entity
            for entity in entities_df.to_dict(orient="records")
        }
        rel_type_hash_map = {
            generate_hash(rel_type.name, rel_type.description): rel_type
            for rel_type in rel_types_df.to_dict(orient="records")
        }

        # Associate relationships
        for index, row in extracted_relationships_df.iterrows():
            from_entity_hash = generate_hash(
                row[Relationship.FROM_ENTITY_COLUMN],
                row[Relationship.FROM_ENTITY_DESCRIPTION_COLUMN],
            )
            to_entity_hash = generate_hash(
                row[Relationship.TO_ENTITY_COLUMN],
                row[Relationship.TO_ENTITY_DESCRIPTION_COLUMN],
            )
            relationship_type_hash = generate_hash(
                row[Relationship.RELATIONSHIP_TYPE_COLUMN],
                row[Relationship.RELATIONSHIP_DESCRIPTION_COLUMN],
            )

            # Find corresponding entities and relationship types
            if from_entity_hash in entity_hash_map:
                row[Relationship.FROM_ENTITY_COLUMN] = entity_hash_map[from_entity_hash]
            if to_entity_hash in entity_hash_map:
                row[Relationship.TO_ENTITY_COLUMN] = entity_hash_map[to_entity_hash]
            if relationship_type_hash in rel_type_hash_map:
                row[Relationship.RELATIONSHIP_TYPE_COLUMN] = rel_type_hash_map[
                    relationship_type_hash
                ]

        return extracted_relationships_df
