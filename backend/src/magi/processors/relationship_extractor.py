"""Processor for extracting relationships from documents."""

import asyncio
import os
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
from pyspark.storagelevel import StorageLevel

from ..services.file_processor import create_relationship_extractor_udf
from ..services.models import Relationship
from ..services.schemas import RELATIONSHIP_SCHEMA
from .base import DocumentProcessor


@dataclass
class RelationshipExtractorProcessor(DocumentProcessor):
    """Processes documents to extract relationships using an LLM."""

    model: str = "gemini-2.0-flash"

    def create_udf(self):
        """Create Spark UDF for relationship extraction."""
        return create_relationship_extractor_udf(self.model)

    async def process(self, df: DataFrame) -> DataFrame:
        """
        Extract relationships from document content.

        Args:
            df: DataFrame containing documents with content

        Returns:
            DataFrame with extracted relationships
        """
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
