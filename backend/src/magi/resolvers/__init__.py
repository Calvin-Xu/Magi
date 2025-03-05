"""Resolvers for entities and relationship types."""

from .base import Resolver
from .openai_resolver import OpenAIResolver

__all__ = [
    "Resolver",
    "OpenAIResolver",
]
