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
    hash_key: str  # Non-optional hash_key field for tracking

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

    pair: ObjectPair[T] = Field(..., description="Pair of objects being verified")
    are_same: bool = False
    updated_name: Optional[str] = None
    updated_description: Optional[str] = None


class LLMObjectPair(BaseModel):
    """Model for a pair of objects to be presented to the LLM."""

    pair_id: int = Field(..., description="Unique identifier for this pair")
    input_name: str = Field(..., description="Name of the input object")
    input_description: str = Field(..., description="Description of the input object")
    candidate_name: str = Field(..., description="Name of the candidate object")
    candidate_description: str = Field(
        ..., description="Description of the candidate object"
    )


class LLMVerificationResult(BaseModel):
    """Model for the LLM to return verification results."""

    pair_id: int = Field(..., description="ID of the pair being verified")
    are_same: bool = Field(
        ..., description="Whether the objects refer to the same entity or concept"
    )
    updated_name: Optional[str] = Field(
        None,
        description="Updated name combining information from both objects (if they are the same)",
    )
    updated_description: Optional[str] = Field(
        None,
        description="Updated description combining information from both objects (if they are the same)",
    )


class LLMVerificationResponse(BaseModel):
    """Model for the complete verification response from the LLM."""

    results: List[LLMVerificationResult] = Field(
        ..., description="List of verification results for each pair of objects"
    )


class ProcessedObject(BaseModel, Generic[T]):
    """Model for a processed object ready to be returned."""

    resolved: T
    reference_id: int
    hash_key: str  # Added hash_key field to replace the need for original object
    is_new: bool = False
    has_updates: bool = False
