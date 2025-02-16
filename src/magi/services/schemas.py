"""Schemas for data structures used in Magi."""

from dataclasses import dataclass
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
)


class UnsupportedFileTypeError(Exception):
    """Raised when a file cannot be converted to plaintext."""

    pass


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


@dataclass
class TextDocument:
    """Represents a text document with its metadata."""

    uri: str  # Full S3 URI
    content: str  # Plain text content
    file_type: str  # Original file type


@dataclass
class DocumentBatch:
    """A batch of documents."""

    documents: list[TextDocument]


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


# Schema for relationship triples in Spark
RELATIONSHIP_SCHEMA = StructType(
    [
        StructField("from_entity", StringType(), False),
        StructField("from_entity_description", StringType(), False),
        StructField("to_entity", StringType(), False),
        StructField("to_entity_description", StringType(), False),
        StructField("relationship_type", StringType(), False),
        StructField("relationship_description", StringType(), False),
        StructField("constraint_condition", StringType(), True),
        StructField("reason", StringType(), False),
        StructField("is_causal", BooleanType(), False),
        StructField("source_document", StringType(), True),
    ]
)
