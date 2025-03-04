"""Gemini-based relationship extractor."""

import re
from typing import List, Optional
import asyncio
from datetime import datetime
from dataclasses import dataclass
import random

from google import genai
from pydantic import BaseModel
from vertexai.preview import tokenization

from .base import (
    RelationshipExtractor,
    RelationshipTriple,
    TextChunk,
    ExtractionMetrics,
)
from ..config import GEMINI_CONFIG
from ..services.rate_limiter import DistributedRateLimiter, RateLimit
from .prompts import RELATIONSHIP_EXTRACTION_PROMPT


@dataclass
class ModelLimits:
    """Model-specific rate and token limits."""

    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    input_token_limit: int  # Maximum input tokens
    vertex_model_id: str  # Model ID for Vertex AI tokenizer


# Model-specific limits
MODEL_LIMITS = {
    "gemini-2.0-flash": ModelLimits(
        rpm=2000,
        tpm=4_000_000,
        input_token_limit=1_048_576,
        vertex_model_id="gemini-1.5-flash-002",  # TODO: 2.0 tokenizer not supported yet
    ),
    "gemini-2.0-flash-thinking-exp": ModelLimits(
        rpm=10,
        tpm=4_000_000,
        input_token_limit=1_048_576,
        vertex_model_id="gemini-1.5-flash-002",  # TODO: 2.0 tokenizer not supported yet
    ),
}


class RelationshipList(BaseModel):
    """Schema for relationship list from Gemini."""

    class Relationship(BaseModel):
        """Schema for a single relationship."""

        subject: str
        subject_description: str
        object: str
        object_description: str
        predicate: str
        predicate_description: str
        constraint_condition: str
        reason: str
        is_causal: str

    relationships: list[Relationship]


class GeminiError(Exception):
    """Base error for Gemini API calls."""

    pass


class GeminiExtractor(RelationshipExtractor):
    """Relationship extractor using Google's Gemini models."""

    def __init__(
        self,
        model: str,
        max_concurrent_requests: int = 10000,  # we have rate limiter
        max_retries: int = 5,
    ):
        """Initialize the Gemini extractor."""
        if model not in MODEL_LIMITS:
            raise ValueError(f"Unsupported Gemini model: {model}")

        limits = MODEL_LIMITS[model]
        super().__init__(
            model=model,
            max_input_tokens=limits.input_token_limit,
            max_concurrent_requests=max_concurrent_requests,
        )

        # Initialize client and rate limiter
        self.client = genai.Client(api_key=GEMINI_CONFIG["api_key"])
        self._rate_limiter = DistributedRateLimiter()
        self._model_limits = limits
        self.max_retries = max_retries

        # Initialize tokenizer using Vertex AI model ID
        self._tokenizer = tokenization.get_tokenizer_for_model(
            self._model_limits.vertex_model_id
        )

        # Gemini-specific limits
        self._rate_limit = RateLimit(
            name="gemini-2.0-flash",
            rpm=self._model_limits.rpm,
            tpm=self._model_limits.tpm,
            window_size=60,
            num_shards=10,
            max_concurrent=4,
        )

    async def close(self):
        """Clean up resources."""
        await self._rate_limiter.close()

    async def _count_tokens(self, text: str) -> int:
        """Count tokens using local Gemini tokenizer."""
        return self._tokenizer.count_tokens(text).total_tokens

    def _count_tokens_sync(self, text: str) -> int:
        """Synchronous version of token counting."""
        return self._tokenizer.count_tokens(text).total_tokens

    async def _call_gemini_api(self, prompt: str) -> Optional[dict]:
        """Make API call to Gemini with rate limiting and retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Get rate limit approval
                retry_after = await self._rate_limiter.acquire(
                    rate_limit=self._rate_limit,
                    tokens=self._count_tokens_sync(prompt),
                    reserve=True,
                )

                if retry_after:
                    # Add more initial delay for first attempt
                    if attempt == 0:
                        retry_after += (
                            random.random() * 30
                        )  # Up to 30 seconds initial delay
                    await asyncio.sleep(retry_after - datetime.now().timestamp())

                # Make the API call
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": RelationshipList,
                    },
                )
                print(f"Response: {response.text}")
                return response

            except Exception as e:
                last_error = e
                print(
                    f"Error in Gemini API call (prompt hash: {hash(prompt)}) (attempt {attempt + 1}): {str(e)}"
                )

                if "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(30 * 2**attempt)
                        continue

                # For non-rate-limit errors, fail fast
                if "RESOURCE_EXHAUSTED" not in str(e):
                    break

        raise GeminiError(
            f"Failed after {attempt + 1} attempts. Last error: {str(last_error)}"
        ) from last_error

    async def _extract_relationships_raw(
        self,
        text: str,
        **kwargs,
    ) -> List[RelationshipTriple]:
        """Extract relationships from a single chunk of text."""
        start_time = datetime.now()

        try:
            # Prepare prompt with text
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(text=text)

            # Call Gemini API with retries and rate limiting
            response = await self._call_gemini_api(prompt)

            # Handle None response or missing content
            if not response or not response.candidates:
                print(f"No valid response for prompt hash: {hash(prompt)}")
                return []

            # Handle missing parsed data
            if not hasattr(response, "parsed") or not response.parsed:
                print(f"No parsed relationships for prompt hash: {hash(prompt)}")
                return []

            # Record metrics and process response
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._metrics.append(
                ExtractionMetrics(
                    input_tokens=self._count_tokens_sync(text),
                    output_tokens=(
                        response.usage_metadata.candidates_token_count
                        if response.usage_metadata
                        else 0
                    ),
                    duration_ms=duration,
                    timestamp=start_time,
                )
            )

            # Parse response into RelationshipTriples
            relationships = []
            for rel in response.parsed.relationships:
                # Handle empty constraint condition
                constraint = (
                    rel.constraint_condition if rel.constraint_condition else ""
                )

                relationships.append(
                    RelationshipTriple(
                        from_entity=rel.subject,
                        from_entity_description=rel.subject_description,
                        to_entity=rel.object,
                        to_entity_description=rel.object_description,
                        relationship_type=rel.predicate,
                        relationship_description=rel.predicate_description,
                        constraint_condition="" if constraint == "None" else constraint,
                        reason=rel.reason,
                        is_causal=rel.is_causal.lower() == "yes",
                    )
                )
            return relationships

        except Exception as e:
            print(f"Error in Gemini extraction (prompt hash: {hash(prompt)}): {str(e)}")
            return []

    def _chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks that fit within token limit."""
        chunks = []
        current_pos = 0

        # First split on multiple newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens_sync(para)

            if para_tokens > self.max_input_tokens:
                # Paragraph too long, split on sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent_tokens = self._count_tokens_sync(sent)
                    if sent_tokens > self.max_input_tokens:
                        # Sentence too long, split on token limit
                        words = sent.split()
                        current_sent = []
                        current_sent_tokens = 0

                        for word in words:
                            word_tokens = self._count_tokens_sync(word)
                            if (
                                current_sent_tokens + word_tokens
                                > self.max_input_tokens
                            ):
                                # Create new chunk
                                chunk_text = " ".join(current_sent)
                                chunks.append(
                                    TextChunk(
                                        text=chunk_text,
                                        start_char=current_pos,
                                        end_char=current_pos + len(chunk_text),
                                        is_sentence_boundary=True,
                                    )
                                )
                                current_pos += len(chunk_text) + 1
                                current_sent = [word]
                                current_sent_tokens = word_tokens
                            else:
                                current_sent.append(word)
                                current_sent_tokens += word_tokens

                        if current_sent:
                            chunk_text = " ".join(current_sent)
                            chunks.append(
                                TextChunk(
                                    text=chunk_text,
                                    start_char=current_pos,
                                    end_char=current_pos + len(chunk_text),
                                    is_sentence_boundary=True,
                                )
                            )
                            current_pos += len(chunk_text) + 1
                    else:
                        if current_tokens + sent_tokens > self.max_input_tokens:
                            # Create new chunk from accumulated sentences
                            chunk_text = " ".join(current_chunk)
                            chunks.append(
                                TextChunk(
                                    text=chunk_text,
                                    start_char=current_pos,
                                    end_char=current_pos + len(chunk_text),
                                    is_sentence_boundary=True,
                                )
                            )
                            current_pos += len(chunk_text) + 1
                            current_chunk = [sent]
                            current_tokens = sent_tokens
                        else:
                            current_chunk.append(sent)
                            current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > self.max_input_tokens:
                    # Create new chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            start_char=current_pos,
                            end_char=current_pos + len(chunk_text),
                            is_paragraph_boundary=True,
                        )
                    )
                    current_pos += len(chunk_text) + 2  # +2 for paragraph break
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text),
                    is_paragraph_boundary=True,
                )
            )

        return chunks
