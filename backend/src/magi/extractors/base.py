"""Base classes for relationship extractors."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, List

from pydantic import BaseModel, Field


class RelationshipTriple(BaseModel):
    """A relationship triple extracted from text."""

    from_entity: str = Field(..., min_length=1)
    from_entity_description: str = Field(
        ...,
        description="A globally unique, disambiguating description of the source entity",
    )
    to_entity: str = Field(..., min_length=1)
    to_entity_description: str = Field(
        ...,
        description="A globally unique, disambiguating description of the target entity",
    )
    relationship_type: str = Field(..., min_length=1)
    relationship_description: str = Field(
        ...,
        description="A globally unique identifying description for this relationship type",
    )
    constraint_condition: str = Field(
        "", description="Conditions under which the relationship holds"
    )
    reason: str = Field(..., min_length=1)
    is_causal: bool = Field(
        False, description="Whether the relationship represents a causal connection"
    )
    source_document: str = Field(
        "", description="The document from which this relationship was extracted"
    )

    def __hash__(self) -> int:
        """Make relationship triples hashable for deduplication."""
        return hash(
            (
                self.from_entity,
                self.from_entity_description,
                self.to_entity,
                self.to_entity_description,
                self.relationship_type,
                self.relationship_description,
                self.constraint_condition,
                self.reason,
                self.is_causal,
            )
        )


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction run."""

    input_tokens: int
    output_tokens: int
    duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.now())


class TextChunk(BaseModel):
    """A chunk of text with metadata."""

    text: str
    start_char: int  # Position in original text
    end_char: int
    is_paragraph_boundary: bool = False
    is_sentence_boundary: bool = False


class RelationshipExtractor(ABC):
    """Base class for relationship extractors."""

    def __init__(
        self,
        model: str,
        max_input_tokens: int,
        max_concurrent_requests: int = 100000,  # should have a separate rate-limiter
    ):
        """Initialize the extractor.

        Args:
            model: Model identifier (e.g., "gemini-2.0-flash")
            max_input_tokens: Maximum tokens per request
            max_concurrent_requests: Rate limiting
        """
        self.model = model
        self.max_input_tokens = max_input_tokens
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._metrics: List[ExtractionMetrics] = []

    @abstractmethod
    async def _count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer."""
        pass

    @abstractmethod
    async def _extract_relationships_raw(
        self,
        text: str,
        **kwargs,
    ) -> List[RelationshipTriple]:
        """Extract relationships from a single chunk of text."""
        pass

    @abstractmethod
    def _chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks that fit within token limit."""
        pass

    async def extract_relationships(
        self,
        text: str,
        **kwargs,
    ) -> AsyncIterator[RelationshipTriple]:
        """
        Extract relationships from text, handling chunking and rate limiting.

        Args:
            text: Input text
            **kwargs: Additional arguments passed to concrete implementation

        Yields:
            Relationship triples
        """
        chunks = self._chunk_text(text)

        for chunk in chunks:
            # Rate limit API calls
            async with self._semaphore:
                try:
                    relationships = await self._extract_relationships_raw(
                        chunk.text,
                        **kwargs,
                    )

                except Exception as e:
                    print(f"Error extracting relationships: {str(e)}")
                    continue

            # Yield relationships from this chunk
            for relationship in relationships:
                yield relationship

    @property
    def metrics(self) -> List[ExtractionMetrics]:
        """Get metrics from all extractions."""
        return self._metrics.copy()
