"""Resolvers for entities and relationship types."""

from .base import SemanticObjectResolver
from .openai_resolver import OpenAIResolver

__all__ = [
    "SemanticObjectResolver",
    "OpenAIResolver",
]
