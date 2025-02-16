from dataclasses import dataclass
from typing import Optional, AsyncIterator
import boto3
from botocore.exceptions import ClientError
from ..config import AWS_CONFIG


@dataclass
class S3Credentials:
    """AWS credentials configuration."""

    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if both credentials are provided."""
        return bool(self.access_key_id and self.secret_access_key)

    def get_effective_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get effective credentials, falling back to environment variables."""
        return (
            self.access_key_id or AWS_CONFIG["access_key_id"],
            self.secret_access_key or AWS_CONFIG["secret_access_key"],
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
    credentials: Optional[S3Credentials] = None,
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

        # Get credentials (or None if not provided/configured)
        if credentials:
            access_key, secret_key = credentials.get_effective_credentials()
        else:
            # Use None so that default AWS credential chain is used
            access_key, secret_key = None, None

        # Create the session
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        sts_client = session.client("sts")
        assumed = sts_client.assume_role(
            RoleArn=AWS_CONFIG["role_arn"],
            RoleSessionName="magi-session",
        )
        creds = assumed["Credentials"]

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )

        # List objects with pagination
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_uri.bucket, Prefix=s3_uri.prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    yield f"s3://{s3_uri.bucket}/{obj['Key']}"

    except ClientError as e:
        raise RuntimeError(f"AWS S3 error: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Error listing S3 objects: {str(e)}") from e
