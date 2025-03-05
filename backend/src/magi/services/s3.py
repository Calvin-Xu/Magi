"""S3 utilities and document reading."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import boto3
from botocore.exceptions import ClientError

from magi.config import FILE_PROCESSOR_CONFIG
from magi.services.aws import AWSCredentials, create_aws_client
from magi.services.schemas import (
    DocumentBatch,
    TextDocument,
    UnsupportedFileTypeError,
    convert_to_text,
)


@dataclass
class S3Uri:
    """Parsed S3 URI components."""

    bucket: str
    prefix: str

    @classmethod
    def parse(cls, uri: str) -> "S3Uri":
        """Parse an S3 URI into bucket and prefix."""
        if not uri.startswith("s3://"):
            raise ValueError("URI must start with s3://")

        # Remove s3:// prefix and split into bucket and key
        path = uri[5:]
        parts = path.split("/", 1)

        if not parts[0]:
            raise ValueError("No bucket specified")

        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        return cls(bucket=bucket, prefix=prefix)


async def list_s3_objects(
    uri: str,
    credentials: Optional[AWSCredentials] = None,
) -> AsyncIterator[str]:
    """
    List all objects under the given S3 URI.

    Args:
        uri: S3 URI (e.g., s3://bucket/prefix/path)
        credentials: Optional AWS credentials (overrides environment variables)

    Yields:
        Full S3 URIs for each object found.
    """
    try:
        s3_uri = S3Uri.parse(uri)
        s3_client = create_aws_client("s3", credentials)

        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_uri.bucket, Prefix=s3_uri.prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    yield f"s3://{s3_uri.bucket}/{obj['Key']}"

    except ClientError as e:
        raise RuntimeError(f"AWS S3 error: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Error listing S3 objects: {str(e)}") from e


class S3DocumentReader:
    """Reads documents from S3 with batching and rate limiting."""

    def __init__(
        self,
        s3_client: boto3.client,
        credentials: Optional[AWSCredentials] = None,
    ):
        """Initialize reader."""
        self.s3_client = s3_client
        self.credentials = credentials
        self.semaphore = asyncio.Semaphore(
            FILE_PROCESSOR_CONFIG["max_concurrent_downloads"]
        )
        self.executor = ThreadPoolExecutor(
            max_workers=FILE_PROCESSOR_CONFIG["max_concurrent_downloads"]
        )

    async def read_documents(
        self,
        base_uri: str,
    ) -> AsyncIterator[DocumentBatch]:
        """Read documents in batches."""
        try:
            current_batch: list[TextDocument] = []

            async for uri in list_s3_objects(base_uri, self.credentials):
                if doc := await fetch_and_convert_to_text(
                    uri,
                    self.s3_client,
                    self.semaphore,
                    self.executor,
                ):
                    current_batch.append(doc)

                if len(current_batch) >= FILE_PROCESSOR_CONFIG["batch_size"]:
                    yield DocumentBatch(documents=current_batch)
                    current_batch = []

            if current_batch:
                yield DocumentBatch(documents=current_batch)

        finally:
            self.executor.shutdown()


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
