"""Processor for extracting relationships from documents."""

import asyncio
from dataclasses import dataclass
import json
import os
from typing import Optional, Type

from magi.extractors.base import RelationshipExtractor
from magi.extractors.gemini import GeminiExtractor
from magi.extractors.openai import OpenAIExtractor
from magi.services.models import Relationship
from magi.services.schemas import RELATIONSHIP_SCHEMA
from magi.utils.logging import get_logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    StringType,
)
from pyspark.storagelevel import StorageLevel

from .base import DocumentProcessor

# Initialize logger
logger = get_logger(__name__)

# Define available models and their corresponding extractor classes
AVAILABLE_MODELS = {
    # OpenAI models
    "o3-mini-2025-01-31": OpenAIExtractor,
    # Gemini models
    "gemini-2.0-flash": GeminiExtractor,
    # 2.0 thinking has no json mode
    # "gemini-2.0-flash-thinking-exp": GeminiExtractor,
}

# Default model to use if none specified
DEFAULT_MODEL = "o3-mini-2025-01-31"


@dataclass
class RelationshipExtractorProcessor(DocumentProcessor):
    """Processes documents to extract relationships using an LLM."""

    model: str = DEFAULT_MODEL

    async def extract_relationships_from_text(
        self,
        text: str,
        extractor: RelationshipExtractor,
    ) -> list[dict]:
        """Extract relationships from text using the provided extractor."""
        relationships = []
        async for rel in extractor.extract_relationships(text):
            relationships.append(rel.model_dump())
        return relationships

    def get_extractor_class(self) -> Type[RelationshipExtractor]:
        """Get the appropriate extractor class for the specified model."""
        if self.model not in AVAILABLE_MODELS:
            logger.warning(
                f"Model {self.model} not found in available models. Using default model {DEFAULT_MODEL}."
            )
            return AVAILABLE_MODELS[DEFAULT_MODEL]
        return AVAILABLE_MODELS[self.model]

    # Cache extractor per process
    def get_extractor(self) -> RelationshipExtractor:
        """Get or create an extractor instance."""
        extractor_class = self.get_extractor_class()
        logger.info(f"Creating {extractor_class.__name__} for model {self.model}")
        return extractor_class(model=self.model)

    def create_udf(self) -> F.UserDefinedFunction:
        """Create a Spark UDF for relationship extraction."""

        def extract_relationships(text: str) -> Optional[str]:
            """Wrapper for async extraction that returns JSON string."""
            # Create new event loop and extractor for this executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get cached extractor instance
                extractor = self.get_extractor()

                # Run extraction
                relationships = loop.run_until_complete(
                    self.extract_relationships_from_text(text, extractor)
                )

                # Always return a valid JSON array, even if empty
                return json.dumps(relationships or [])

            except Exception as e:
                logger.exception(f"Error in relationship extraction: {str(e)}")
                return json.dumps([])  # Return empty array instead of None
            finally:
                loop.close()

        return F.udf(extract_relationships, StringType())

    async def process(self, df: DataFrame) -> DataFrame:
        """
        Extract relationships from document content.

        Args:
            df: DataFrame containing documents with content

        Returns:
            DataFrame with extracted relationships
        """
        logger.info(f"Processing documents with model: {self.model}")

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

        # Explode the array of relationships into separate rows
        exploded_df = df_with_relationships.select(
            F.col("uri").alias("source_document_uri"),
            F.explode("extracted_relationships").alias("relationship"),
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
