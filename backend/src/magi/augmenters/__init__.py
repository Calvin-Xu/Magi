"""Knowledge Graph Augmentation module for Magi.

This module provides functionality to augment a schema graph with domain-specific
knowledge derived from LLM research, creating a hybrid schema-knowledge graph.
"""

from magi.augmenters.base import GraphAugmenter
from magi.augmenters.perplexity import PerplexityAugmenter

__all__ = ["GraphAugmenter", "PerplexityAugmenter"]
