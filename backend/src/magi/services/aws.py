"""AWS service utilities."""

from dataclasses import dataclass
from typing import Optional

import boto3

from magi.config import AWS_CONFIG


@dataclass
class AWSCredentials:
    """AWS credentials configuration."""

    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None

    def get_effective_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get effective credentials, falling back to environment variables."""
        return (
            self.access_key_id or AWS_CONFIG["access_key_id"],
            self.secret_access_key or AWS_CONFIG["secret_access_key"],
        )


def create_aws_client(
    service: str,
    credentials: Optional[AWSCredentials] = None,
    assume_role: bool = False,
) -> boto3.client:
    """
    Create an AWS service client with optional role assumption.

    Args:
        service: AWS service name (e.g., 's3', 'sts')
        credentials: Optional AWS credentials
        assume_role: Whether to assume the configured role

    Returns:
        Configured AWS service client
    """
    # Get base credentials
    access_key, secret_key = (
        credentials.get_effective_credentials() if credentials else (None, None)
    )

    # Create initial session
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    if assume_role and AWS_CONFIG["role_arn"]:
        # Assume role using STS
        sts = session.client("sts")
        assumed = sts.assume_role(
            RoleArn=AWS_CONFIG["role_arn"],
            RoleSessionName="magi-session",
        )
        creds = assumed["Credentials"]

        # Create client with temporary credentials
        return boto3.client(
            service,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )

    # Create client with base credentials
    return session.client(service)
