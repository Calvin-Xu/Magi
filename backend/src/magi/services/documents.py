"""Schemas for data structures used in Magi."""

from dataclasses import dataclass


class UnsupportedFileTypeError(Exception):
    """Raised when a file cannot be converted to plaintext."""

    pass


@dataclass
class TextDocument:
    """Represents a text document with its metadata."""

    uri: str  # Full S3 URI
    content: str  # Plain text content
    file_type: str  # Original file type
    relationships_json: str = ""  # JSON string of extracted relationships


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
