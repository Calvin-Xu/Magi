"""
Data models for the resolver module.

This module contains Pydantic models used by resolvers for entity and relationship type resolution.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")  # Generic type for the object (entity or relationship type)


class ObjectWithEmbedding(BaseModel):
    """Base model for objects with embeddings."""

    name: str
    description: str
    embedding: List[float] = Field(default_factory=list)
    reference_id: Optional[int] = None

    # Additional fields will be stored here
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def id_field(self) -> Optional[int]:
        """Return the ID field value."""
        return self.reference_id


class SimilarityResult(BaseModel):
    """Model for representing similarity between two objects."""

    similarity: float
    is_from_input_list: bool = False


class SimilarObject(ObjectWithEmbedding):
    """Model for an object with similarity information."""

    similarity: float = 0.0


class ObjectPair(BaseModel, Generic[T]):
    """Model for a pair of objects to be compared."""

    input_object: T
    similar_object: Optional[T] = None
    is_from_input_list: bool = False


class VerificationResult(BaseModel, Generic[T]):
    """Model for the result of object verification."""

    input_object: T
    db_object: Optional[T] = None
    is_same: bool = False
    updated_name: Optional[str] = None
    updated_description: Optional[str] = None
    is_from_input_list: bool = False


class ProcessedObject(BaseModel, Generic[T]):
    """Model for a processed object ready to be returned."""

    original: T
    resolved: T
    reference_id: int
    is_new: bool = False
    has_updates: bool = False
