"""Document processing pipeline."""

import logging
from typing import AsyncIterator, Optional

import asyncpg
from pyspark.sql import DataFrame, SparkSession

from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.processors.object_resolution_processor import ObjectResolutionProcessor
from magi.processors.relationship_extractor import RelationshipExtractorProcessor
from magi.resolvers import OpenAIResolver
from magi.resolvers.base import Resolver
from magi.utils import get_logger, set_global_log_level

from .aws import AWSCredentials, create_aws_client
from .models import Entity, RelationshipType
from .relationship_extraction import RelationshipExtractionService
from .s3 import S3DocumentReader
from .documents import DocumentBatch

# Create a logger for this module
logger = get_logger(__name__)


class Pipeline:
    """Document processing pipeline."""

    def __init__(
        self,
        spark: SparkSession,
        conn: asyncpg.Connection,
        model: str,
        embedding_provider: Optional[VoyageEmbeddingProvider] = None,
        entity_resolver: Optional[Resolver[Entity]] = None,
        rel_type_resolver: Optional[Resolver[RelationshipType]] = None,
        credentials: Optional[AWSCredentials] = None,
        log_level: int = logging.DEBUG,
    ):
        """
        Initialize pipeline.

        Args:
            spark: SparkSession
            conn: Database connection
            embedding_provider: Embedding provider
            entity_resolver: Resolver for entities
            rel_type_resolver: Resolver for relationship types
            credentials: AWS credentials
            model: Model name
            log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        # Set the global log level
        set_global_log_level(log_level)

        self.spark = spark
        self.conn = conn
        self.model = model

        if embedding_provider is None:
            embedding_provider = VoyageEmbeddingProvider()

        if entity_resolver is None:
            entity_resolver = OpenAIResolver[Entity](
                conn=conn, embedding_provider=embedding_provider, table_name="entities"
            )

        if rel_type_resolver is None:
            rel_type_resolver = OpenAIResolver[RelationshipType](
                conn=conn,
                embedding_provider=embedding_provider,
                table_name="relationship_types",
            )

        self.embedding_provider = embedding_provider
        self.entity_resolver = entity_resolver
        self.rel_type_resolver = rel_type_resolver

        # Create an extractor based on the model
        from magi.processors.relationship_extractor import (
            AVAILABLE_MODELS,
            DEFAULT_MODEL,
        )

        extractor_class = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
        self.extractor = extractor_class(model=model)

        # Create the relationship extraction service
        self.relationship_extraction_service = RelationshipExtractionService(
            self.extractor
        )

        # Initialize S3 reader
        s3_client = create_aws_client("s3", credentials)
        self.reader = S3DocumentReader(s3_client, credentials=credentials)

        # Initialize processors
        self.processors = [
            RelationshipExtractorProcessor(model=model),
            ObjectResolutionProcessor(
                embedding_provider=embedding_provider,
                entity_resolver=entity_resolver,
                rel_type_resolver=rel_type_resolver,
                conn=conn,
            ),
        ]

        logger.info("Pipeline initialized")

    async def process_documents(
        self,
        base_uri: str,
    ) -> AsyncIterator[DataFrame]:
        """
        Process documents through the pipeline.

        Args:
            base_uri: Base URI for documents (e.g., s3://bucket/prefix)

        Yields:
            Processed DataFrames
        """
        logger.info(f"Processing documents from {base_uri}")

        # Read documents from S3
        document_batches = self.reader.read_documents(base_uri)

        # Extract relationships from documents
        document_batches_with_relationships = (
            self.relationship_extraction_service.process_batches(document_batches)
        )

        # Process each batch
        async for batch in document_batches_with_relationships:
            # Convert batch to DataFrame
            df = self._batch_to_dataframe(batch)

            # Apply each processor in sequence
            for processor in self.processors:
                logger.info(f"Applying processor {processor.__class__.__name__}")
                df = await processor.process(df)

            yield df

    def _batch_to_dataframe(self, batch: DocumentBatch) -> DataFrame:
        """Convert document batch to Spark DataFrame."""
        rows = [
            (doc.uri, doc.content, doc.file_type, doc.relationships_json)
            for doc in batch.documents
        ]
        return self.spark.createDataFrame(
            rows,
            ["uri", "content", "file_type", "relationships_json"],
        )
