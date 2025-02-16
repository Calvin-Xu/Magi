"""File processing utilities."""

from dataclasses import dataclass
from typing import Optional, AsyncIterator
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import boto3
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType

from .s3 import S3Uri, list_s3_objects
from .aws import AWSCredentials, create_aws_client


class UnsupportedFileTypeError(Exception):
    """Raised when a file cannot be converted to plaintext."""

    pass


# Constants for tuning
MAX_CONCURRENT_DOWNLOADS = 50
BATCH_SIZE = 1000  # Number of documents to process before creating a DataFrame
MAX_RETRIES = 3


@dataclass
class TextDocument:
    """Represents a text document with its metadata."""

    uri: str  # Full S3 URI
    content: str  # Plain text content
    file_type: str  # Original file type


def convert_to_text(content: bytes, file_type: str) -> str:
    """
    Convert file content to plaintext.

    Args:
        content: Raw file content
        file_type: File extension (without dot)

    Returns:
        Plaintext content

    Raises:
        UnsupportedFileTypeError: If file type cannot be converted to text
    """
    if file_type.lower() in {"txt", "md"}:
        return content.decode("utf-8")

    raise UnsupportedFileTypeError(
        f"Cannot convert file type '{file_type}' to plaintext"
    )


async def fetch_and_convert_to_text(
    uri: str,
    s3_client: boto3.client,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> Optional[TextDocument]:
    """Fetch and convert file content to text with rate limiting."""
    async with semaphore:  # Limit concurrent downloads
        try:
            s3_uri = S3Uri.parse(uri)

            # Use ThreadPoolExecutor for blocking S3 operations
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: s3_client.get_object(Bucket=s3_uri.bucket, Key=s3_uri.prefix),
            )

            content = await loop.run_in_executor(
                executor, lambda: response["Body"].read()
            )

            file_type = Path(uri).suffix.lstrip(".")

            try:
                text_content = convert_to_text(content, file_type)
                return TextDocument(
                    uri=uri,
                    content=text_content,
                    file_type=file_type,
                )
            except UnsupportedFileTypeError as e:
                print(f"Skipping {uri}: {str(e)}")
                return None

        except Exception as e:
            print(f"Error processing {uri}: {str(e)}")
            return None


async def process_batch(
    uris: list[str],
    s3_client: boto3.client,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> list[TextDocument]:
    """Process a batch of documents concurrently."""
    tasks = [
        fetch_and_convert_to_text(uri, s3_client, semaphore, executor) for uri in uris
    ]
    results = await asyncio.gather(*tasks)
    return [doc for doc in results if doc is not None]


async def process_documents(
    base_uri: str,
    credentials: Optional[AWSCredentials] = None,
    spark: SparkSession = None,
) -> AsyncIterator[DataFrame]:
    """
    Process documents and yield DataFrames in batches.

    Args:
        base_uri: Base S3 URI to process
        credentials: Optional AWS credentials
        spark: SparkSession for creating DataFrames

    Yields:
        Spark DataFrames containing processed documents
    """
    s3_client = create_aws_client("s3", credentials)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    # Create schema once
    schema = StructType(
        [
            StructField("uri", StringType(), False),
            StructField("content", StringType(), False),
            StructField("file_type", StringType(), False),
        ]
    )

    current_batch: list[str] = []

    # Use ThreadPoolExecutor for S3 operations
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        async for uri in list_s3_objects(base_uri, credentials):
            current_batch.append(uri)

            if len(current_batch) >= BATCH_SIZE:
                # Process batch
                documents = await process_batch(
                    current_batch, s3_client, semaphore, executor
                )

                # Create and yield DataFrame if we have documents
                if documents:
                    rows = [(doc.uri, doc.content, doc.file_type) for doc in documents]
                    yield spark.createDataFrame(rows, schema)

                # Reset batch
                current_batch = []

        # Process remaining documents
        if current_batch:
            documents = await process_batch(
                current_batch, s3_client, semaphore, executor
            )
            if documents:
                rows = [(doc.uri, doc.content, doc.file_type) for doc in documents]
                yield spark.createDataFrame(rows, schema)
