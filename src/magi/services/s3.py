"""S3 service utilities."""

from dataclasses import dataclass
from typing import Optional, AsyncIterator
from botocore.exceptions import ClientError

from .aws import AWSCredentials, create_aws_client


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
