"""Document and graph augmentation pipelines."""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Type

import asyncpg
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from magi.augmenters.base import GraphAugmenter
from magi.augmenters.perplexity import PerplexityAugmenter
from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.processors.object_resolution_processor import ObjectResolutionProcessor
from magi.processors.relationship_extraction_processor import (
    RelationshipExtractionProcessor,
)
from magi.resolvers import OpenAIResolver
from magi.resolvers.base import SemanticObjectResolver
from magi.services.models import (
    RELATIONSHIP_SCHEMA,
    Entity,
    ExtractedRelationship,
    RelationshipType,
)
from magi.utils import get_logger, set_global_log_level

from .aws import AWSCredentials, create_aws_client
from .documents import DocumentBatch
from .relationship_extraction import DocumentsRelationshipExtractionService
from .s3 import S3DocumentReader

# Create a logger for this module
logger = get_logger(__name__)


class DocumentPipeline:
    """Document processing pipeline."""

    def __init__(
        self,
        spark: SparkSession,
        conn: asyncpg.Connection,
        model: str,
        embedding_provider: Optional[VoyageEmbeddingProvider] = None,
        entity_resolver: Optional[SemanticObjectResolver[Entity]] = None,
        rel_type_resolver: Optional[SemanticObjectResolver[RelationshipType]] = None,
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
        from magi.processors.relationship_extraction_processor import (
            AVAILABLE_MODELS,
            DEFAULT_MODEL,
        )

        extractor_class = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS[DEFAULT_MODEL])
        self.extractor = extractor_class(model=model)

        # Create the relationship extraction service
        self.relationship_extraction_service = DocumentsRelationshipExtractionService(
            self.extractor
        )

        # Initialize S3 reader
        s3_client = create_aws_client("s3", credentials)
        self.reader = S3DocumentReader(s3_client, credentials=credentials)

        # Initialize processors
        self.processors = [
            RelationshipExtractionProcessor(model=model),
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
            ["source_document_uri", "content", "file_type", "relationships_json"],
        )


class GraphAugmentationPipeline:
    """Pipeline for augmenting knowledge graphs with domain-specific information."""

    def __init__(
        self,
        spark: SparkSession,
        conn: asyncpg.Connection,
        augmenter: Optional[GraphAugmenter] = None,
        augmenter_type: Type[GraphAugmenter] = PerplexityAugmenter,
        embedding_provider: Optional[VoyageEmbeddingProvider] = None,
        entity_resolver: Optional[SemanticObjectResolver[Entity]] = None,
        rel_type_resolver: Optional[SemanticObjectResolver[RelationshipType]] = None,
        log_level: int = logging.DEBUG,
    ):
        """
        Initialize the graph augmentation pipeline.

        Args:
            spark: SparkSession for data processing
            conn: Database connection
            augmenter: GraphAugmenter instance for research. If None, one will be created
            augmenter_type: Type of augmenter to create if augmenter is None
            embedding_provider: Provider for embeddings
            entity_resolver: Resolver for entities
            rel_type_resolver: Resolver for relationship types
            log_level: Logging level
        """
        # Set the global log level
        set_global_log_level(log_level)

        self.spark = spark
        self.conn = conn

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

        # Create augmenter if not provided
        self.augmenter = augmenter or augmenter_type()

        # Initialize processors
        self.processors = [
            RelationshipExtractionProcessor(model="perplexity"),
            ObjectResolutionProcessor(
                embedding_provider=embedding_provider,
                entity_resolver=entity_resolver,
                rel_type_resolver=rel_type_resolver,
                conn=conn,
            ),
        ]

        logger.info("Graph augmentation pipeline initialized")

    async def augment_graph(
        self,
        context: Optional[str] = None,
        user_instruction: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Augment the graph with domain-specific knowledge.

        This method:
        1. Gets the schema context if not provided
        2. Uses the augmenter to research and extract relationships
        3. Converts the relationships to a Spark DataFrame
        4. Processes them through the pipeline to resolve and save to the database

        Args:
            context: Schema context for research. If None, it will be created
            user_instruction: Optional guidance for the research
            **kwargs: Additional arguments to pass to the augmenter

        Returns:
            Dictionary containing statistics about the augmentation process and the final DataFrame
        """
        logger.info("Starting graph augmentation")

        # Step 1: Create schema context if not provided
        if context is None:
            logger.info("Creating schema context")
            context = await self.augmenter.create_context(self.conn)

        # Step 2: Use augmenter to extract relationships
        logger.info("Performing domain research with augmenter")
        summary, relationships = await self.augmenter.get_augmented_relationships(
            context=context, user_instruction=user_instruction, **kwargs
        )

        if not relationships:
            logger.warning("No relationships found during augmentation")
            return {
                "relationships_processed": 0,
                "relationships_df": self.spark.createDataFrame([], []),
            }

        # Step 3: Convert to DataFrame
        logger.info(f"Converting {len(relationships)} relationships to DataFrame")
        relationships_df = self._relationships_to_dataframe(relationships)

        # Process the relationships using a pipeline
        logger.info("Processing relationships through pipeline")

        # Verify the processors and dependency relationship
        if not hasattr(self, "processors") or not self.processors:
            # Recreate the processors if missing
            logger.warning("No processors configured, initializing default processors")
            self.processors = [
                RelationshipExtractionProcessor(model="perplexity"),
                ObjectResolutionProcessor(
                    embedding_provider=self.embedding_provider,
                    entity_resolver=self.entity_resolver,
                    rel_type_resolver=self.rel_type_resolver,
                    conn=self.conn,
                ),
            ]

        # Log processor chain
        processor_names = [p.__class__.__name__ for p in self.processors]
        logger.info(f"Processing chain: {' -> '.join(processor_names)}")

        # Start with the initial DataFrame
        processed_df = relationships_df

        # Apply each processor in sequence
        for processor in self.processors:
            processor_name = processor.__class__.__name__
            logger.info(f"Starting processor: {processor_name}")

            # Apply the processor
            try:
                processed_df = await processor.process(processed_df)
                logger.info(f"Successfully completed processor: {processor_name}")
            except Exception as e:
                logger.exception(f"Error in processor {processor_name}: {str(e)}")
                raise

        # Create a user-friendly display DataFrame
        display_df = processed_df.select(
            F.col("from_entity").alias("Subject"),
            F.col("relationship_type").alias("Relationship"),
            F.col("to_entity").alias("Object"),
            F.col("reason").alias("Reason"),
            F.col("constraint_condition").alias("Constraint"),
            F.col("is_causal").alias("Is Causal"),
            F.col("source_uri").alias("Source"),
        )

        # Display info about the processed data
        row_count = processed_df.count()

        logger.info(f"Graph augmentation complete: {row_count} relationships processed")

        return {
            "summary": summary,
            "context_length": len(context) if context else 0,
            "relationships_found": len(relationships),
            "relationships_processed": row_count,
            "relationships_df": display_df,
        }

    def _relationships_to_dataframe(
        self, relationships: List[ExtractedRelationship]
    ) -> DataFrame:
        """
        Convert a list of ExtractedRelationship objects to a Spark DataFrame.

        The DataFrame will have the structure expected by RelationshipExtractionProcessor.

        Args:
            relationships: List of extracted relationships

        Returns:
            Spark DataFrame with relationships in the expected format
        """
        # Convert relationships to JSON
        relationships_json = json.dumps([rel.model_dump() for rel in relationships])

        # Create a single-row DataFrame with the relationships_json column
        df = self.spark.createDataFrame([(relationships_json,)], ["relationships_json"])

        # Parse the JSON into the expected schema
        df_with_relationships = df.withColumn(
            "extracted_relationships",
            F.from_json(
                F.col("relationships_json"),
                RELATIONSHIP_SCHEMA,
            ),
        )

        return df_with_relationships
