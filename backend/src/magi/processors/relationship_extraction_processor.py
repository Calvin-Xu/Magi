"""Processor for extracting relationships from documents."""

import asyncio
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
from pyspark.storagelevel import StorageLevel

from magi.extractors.gemini_extractor import GeminiExtractor
from magi.extractors.openai_extractor import OpenAIExtractor
from magi.services.models import Relationship, RELATIONSHIP_SCHEMA
from magi.utils.logging import get_logger

from .base import SparkDataFrameProcessor

# Initialize logger
logger = get_logger(__name__)

# Define available models and their corresponding extractor classes
AVAILABLE_MODELS = {
    # OpenAI models
    "o3-mini-2025-01-31": OpenAIExtractor,
    "gpt-4o-2024-11-20": OpenAIExtractor,
    # Gemini models
    "gemini-2.0-flash": GeminiExtractor,
    "gemini-2.5-pro-exp-03-25": GeminiExtractor,
    # 2.0 thinking has no json mode
    # "gemini-2.0-flash-thinking-exp": GeminiExtractor,
}

# Default model to use if none specified
DEFAULT_MODEL = "o3-mini-2025-01-31"


@dataclass
class RelationshipExtractionProcessor(SparkDataFrameProcessor):
    """Processes documents with pre-extracted relationships."""

    model: str = DEFAULT_MODEL
    _extractor_cache = None  # Instance-level cache for the extractor

    async def process(self, df: DataFrame) -> DataFrame:
        """
        Process pre-extracted relationships from documents.

        Args:
            df: DataFrame containing documents with relationships_json

        Returns:
            DataFrame with extracted relationships
        """
        logger.info(f"Processing pre-extracted relationships from model: {self.model}")

        # Parse the pre-extracted relationships JSON
        df_with_relationships = df.withColumn(
            "extracted_relationships",
            F.from_json(
                F.col("relationships_json"),
                ArrayType(RELATIONSHIP_SCHEMA),
            ),
        ).persist(StorageLevel.MEMORY_AND_DISK)

        # Ensure the parsing is triggered
        await asyncio.to_thread(df_with_relationships.count)

        # Check if the source document is from the document pipeline
        source_document_exists = "source_document_uri" in df_with_relationships.columns
        logger.info(f"Source document URI column exists: {source_document_exists}")

        # Explode the array of relationships into separate rows
        if source_document_exists:
            # Include source_document_uri when exploding relationships
            exploded_df = df_with_relationships.select(
                F.explode_outer("extracted_relationships").alias("relationship"),
                F.col("source_document_uri"),
            )
        else:
            # Just explode relationships without source document
            exploded_df = df_with_relationships.select(
                F.explode_outer("extracted_relationships").alias("relationship"),
            )

        # Build selection columns based on available fields
        select_cols = [
            "relationship." + Relationship.FROM_ENTITY_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_TYPE_COLUMN,
            "relationship." + Relationship.TO_ENTITY_COLUMN,
            "relationship." + Relationship.CONSTRAINT_CONDITION_COLUMN,
            "relationship." + Relationship.REASON_COLUMN,
            "relationship." + Relationship.IS_CAUSAL_COLUMN,
            "relationship." + Relationship.FROM_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.TO_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN,
            # Include confidence field if available
            "relationship." + Relationship.CONFIDENCE_COLUMN,
            # Default value for from_imported_schema
            F.lit(False).alias(Relationship.FROM_IMPORTED_SCHEMA_COLUMN),
        ]

        # Add the source URI, preferring document source when available
        if source_document_exists:
            select_cols.append(
                F.col("source_document_uri").alias(Relationship.SOURCE_URI_COLUMN)
            )
            logger.info("Using document source_document_uri as the relationship source")
        else:
            select_cols.append("relationship." + Relationship.SOURCE_URI_COLUMN)
            logger.info("Using relationship's own source_uri")

        # Select fields from the relationship
        extracted_relationships_df = exploded_df.select(*select_cols)

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
                        F.col(Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN),
                    )
                ),
            )
        )

        return extracted_relationships_df
