"""
Revised data models for the resolver module.

This module contains Pydantic models used by resolvers for entity resolution.
We have added a new model, IntraBatchMergeResult, to capture merges of input objects.
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

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def id_field(self) -> Optional[int]:
        """Return the ID field value (alias for reference_id)."""
        return self.reference_id


class IntraBatchMergeResult(BaseModel):
    """
    Model to store the LLM's merging of multiple input objects that it thinks are duplicates.

    For example:
    {
      "merged_id": "A short unique identifier for this merged entity",
      "merged_name": "A new or chosen name that best represents the group",
      "merged_description": "A combined description from all duplicates",
      "member_hash_keys": ["abc123", "def456"]  # hash_keys for the objects that merged
    }
    """

    merged_id: str
    merged_name: str
    merged_description: str
    member_hash_keys: List[str]


class MergedEntity(BaseModel):
    """
    Internal container for a single new merged entity. This is the “representative” object
    after merging. Once created, we can do DB matching on it. We keep track of which
    input hash_keys got merged into it.
    """

    temp_id: str  # e.g. the same as merged_id from the LLM or "group_1"
    name: str
    description: str
    embedding: List[float] = Field(default_factory=list)
    member_hash_keys: List[str] = Field(default_factory=list)

    # Once we match with DB or insert:
    reference_id: Optional[int] = None


class LLMIntraBatchMergeResponse(BaseModel):
    """
    Example structure for a response from the LLM after it merges duplicates within the batch.

    {
      "merged_entities": [
        {
          "merged_id": "entity_0",
          "merged_name": "Mark Antony",
          "merged_description": "Combined description",
          "member_hash_keys": ["7b472bd2...", "7d71a06f..."]
        },
        {
          "merged_id": "entity_1",
          "merged_name": "Octavian",
          "merged_description": "....",
          "member_hash_keys": ["7d71a06f..."]
        },
        ...
      ]
    }
    """

    merged_entities: List[IntraBatchMergeResult]


class ProcessedObject(BaseModel, Generic[T]):
    """Model for a processed object ready to be returned."""

    resolved: T
    reference_id: int
    hash_key: str  # Matches the original input object's hash_key
    is_new: bool = False
    has_updates: bool = False
