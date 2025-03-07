"""Processor for extracting relationships from documents."""

import asyncio
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
from pyspark.storagelevel import StorageLevel

from magi.extractors.gemini_extractor import GeminiExtractor
from magi.extractors.openai_extractor import OpenAIExtractor
from magi.services.models import Relationship
from magi.services.schemas import RELATIONSHIP_SCHEMA
from magi.utils.logging import get_logger

from .base import DocumentProcessor

# Initialize logger
logger = get_logger(__name__)

# Define available models and their corresponding extractor classes
AVAILABLE_MODELS = {
    # OpenAI models
    "o3-mini-2025-01-31": OpenAIExtractor,
    "gpt-4o-2024-11-20": OpenAIExtractor,
    # Gemini models
    "gemini-2.0-flash": GeminiExtractor,
    # 2.0 thinking has no json mode
    # "gemini-2.0-flash-thinking-exp": GeminiExtractor,
}

# Default model to use if none specified
DEFAULT_MODEL = "o3-mini-2025-01-31"


@dataclass
class RelationshipExtractorProcessor(DocumentProcessor):
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

        # Explode the array of relationships into separate rows
        exploded_df = df_with_relationships.select(
            F.col("uri").alias("source_document_uri"),
            F.explode_outer("extracted_relationships").alias("relationship"),
        )

        # Select fields from the relationship and add the source_document_uri from the document
        extracted_relationships_df = exploded_df.select(
            "relationship." + Relationship.FROM_ENTITY_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_TYPE_COLUMN,
            "relationship." + Relationship.TO_ENTITY_COLUMN,
            "relationship." + Relationship.CONSTRAINT_CONDITION_COLUMN,
            "relationship." + Relationship.REASON_COLUMN,
            "relationship." + Relationship.IS_CAUSAL_COLUMN,
            "relationship." + Relationship.FROM_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.TO_ENTITY_DESCRIPTION_COLUMN,
            "relationship." + Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN,
            # Use the source_document_uri from the document, not from the relationship
            F.col("source_document_uri").alias(Relationship.SOURCE_DOCUMENT_URI_COLUMN),
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
                        F.col(Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN),
                    )
                ),
            )
        )

        return extracted_relationships_df
