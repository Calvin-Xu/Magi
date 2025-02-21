"""Document processing pipeline."""

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
import asyncio
from pyspark.storagelevel import StorageLevel

from .aws import AWSCredentials, create_aws_client
from .s3 import DocumentBatch, S3DocumentReader
from .schemas import RELATIONSHIP_SCHEMA
from .file_processor import create_relationship_extractor_udf


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
            extract_rels_udf(F.col("content"), F.col("uri")),
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

        final_df = exploded_df.select(
            "relationship.from_entity",
            "relationship.relationship_type",
            "relationship.to_entity",
            "relationship.constraint_condition",
            "relationship.reason",
            "relationship.is_causal",
            "relationship.from_entity_description",
            "relationship.to_entity_description",
            "relationship.relationship_description",
            F.col("uri").alias("source_document_uri"),
        )

        return final_df


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
