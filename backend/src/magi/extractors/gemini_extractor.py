"""Gemini-based relationship extractor."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from google import genai
from pydantic import BaseModel
from vertexai.preview import tokenization

from magi.config import GEMINI_CONFIG
from magi.services.rate_limiter import RateLimit, rate_limiter
from magi.utils.logging import get_logger

from .base import (
    ExtractionMetrics,
    RelationshipExtractor,
    RelationshipTriple,
)
from .prompts import RELATIONSHIP_EXTRACTION_PROMPT

# Initialize logger
logger = get_logger(__name__)


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

    # Class-level tokenizer cache
    _tokenizer_cache = {}

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
        self.client = genai.Client(api_key=GEMINI_CONFIG.api_key)
        self._rate_limiter = rate_limiter
        self._model_limits = limits
        self.max_retries = max_retries

        logger.info(f"Initialized GeminiExtractor with model: {model}")
        logger.debug(
            f"Model limits: RPM={limits.rpm}, TPM={limits.tpm}, Max input tokens={limits.input_token_limit}"
        )

        # Initialize tokenizer using Vertex AI model ID - use class-level cache
        vertex_model_id = self._model_limits.vertex_model_id
        if vertex_model_id not in GeminiExtractor._tokenizer_cache:
            GeminiExtractor._tokenizer_cache[vertex_model_id] = (
                tokenization.get_tokenizer_for_model(vertex_model_id)
            )
            logger.debug(f"Created cached tokenizer for model: {vertex_model_id}")

        self._tokenizer = GeminiExtractor._tokenizer_cache[vertex_model_id]
        logger.debug(f"Using tokenizer for model: {vertex_model_id}")

        # Gemini-specific limits
        self._rate_limit = RateLimit(
            name="gemini-2.0-flash",
            rpm=self._model_limits.rpm,
            tpm=self._model_limits.tpm,
            window_size=60,
            num_shards=10,
            max_concurrent=16,
        )
        logger.debug(f"Rate limit configured: {self._rate_limit}")

    async def close(self):
        """Clean up resources."""
        logger.debug("Closing GeminiExtractor resources")
        await self._rate_limiter.close()

    async def _count_tokens(self, text: str) -> int:
        """Count tokens using local Gemini tokenizer."""
        token_count = self._tokenizer.count_tokens(text).total_tokens
        logger.debug(f"Token count for text: {token_count} tokens")
        return token_count

    def _count_tokens_sync(self, text: str) -> int:
        """Synchronous version of token counting."""
        return self._tokenizer.count_tokens(text).total_tokens

    async def _call_gemini_api(self, prompt: str) -> Optional[dict]:
        """Make API call to Gemini with rate limiting and retries."""
        prompt_hash = hash(prompt)
        logger.debug(f"Calling Gemini API (prompt hash: {prompt_hash})")

        token_count = self._count_tokens_sync(prompt)
        logger.debug(f"Prompt token count: {token_count}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Use the context manager for rate limiting
                async with self._rate_limiter.acquire_context(
                    rate_limit=self._rate_limit,
                    tokens=token_count,
                    reserve=True,
                ) as retry_after:
                    if retry_after:
                        # In case we are told to wait until a specific time:
                        wait_seconds = max(
                            0.0, retry_after - datetime.now().timestamp()
                        )
                        # if attempt == 0:
                        #     # Add a small random initial jitter
                        #     wait_seconds += random.random() * 5
                        logger.debug(
                            f"Rate limited, waiting for {wait_seconds:.2f} seconds"
                        )
                        await asyncio.sleep(wait_seconds)
                        continue  # Try again after waiting

                    logger.info(
                        f"Making Gemini API call (attempt {attempt + 1}/{self.max_retries})"
                    )
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
                    logger.debug(f"Received response: {response.text}")
                    return response

            except Exception as e:
                last_error = e
                logger.error(
                    f"Error in Gemini API call (prompt hash: {prompt_hash}) (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )

                if "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < self.max_retries - 1:
                        backoff_time = 30 * 2**attempt
                        logger.info(
                            f"Resource exhausted, backing off for {backoff_time} seconds before retry"
                        )
                        await asyncio.sleep(backoff_time)
                        continue

                # For non-rate-limit errors, fail fast
                if "RESOURCE_EXHAUSTED" not in str(e):
                    logger.error(f"Non-rate-limit error, failing fast: {str(e)}")
                    break

        error_msg = (
            f"Failed after {attempt + 1} attempts. Last error: {str(last_error)}"
        )
        logger.error(error_msg)
        raise GeminiError(error_msg) from last_error

    async def _extract_relationships_raw(
        self,
        text: str,
        **kwargs,
    ) -> List[RelationshipTriple]:
        """Extract relationships from a single chunk of text."""
        start_time = datetime.now()
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"Extracting relationships from text: {text_preview}")

        try:
            # Prepare prompt with text
            prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(text=text)
            prompt_hash = hash(prompt)
            logger.debug(f"Prepared prompt with hash: {prompt_hash}")

            # Call Gemini API with retries and rate limiting
            logger.info("Calling Gemini API with prompt")
            response = await self._call_gemini_api(prompt)

            # Handle None response or missing content
            if not response or not response.candidates:
                logger.warning(f"No valid response for prompt hash: {prompt_hash}")
                return []

            # Handle missing parsed data
            if not hasattr(response, "parsed") or not response.parsed:
                logger.warning(
                    f"No parsed relationships for prompt hash: {prompt_hash}"
                )
                return []

            # Record metrics and process response
            duration = (datetime.now() - start_time).total_seconds() * 1000
            output_tokens = (
                response.usage_metadata.candidates_token_count
                if response.usage_metadata
                else 0
            )

            logger.info(
                f"Extraction completed in {duration:.2f}ms. "
                f"Input tokens: {self._count_tokens_sync(text)}, "
                f"Output tokens: {output_tokens}"
            )

            self._metrics.append(
                ExtractionMetrics(
                    input_tokens=self._count_tokens_sync(text),
                    output_tokens=output_tokens,
                    duration_ms=duration,
                    timestamp=start_time,
                )
            )

            # Parse response into RelationshipTriples
            relationships = []
            relationship_count = (
                len(response.parsed.relationships) if response.parsed else 0
            )
            logger.info(f"Found {relationship_count} relationships in response")

            for i, rel in enumerate(response.parsed.relationships):
                # Handle empty constraint condition
                constraint = (
                    rel.constraint_condition if rel.constraint_condition else ""
                )

                logger.debug(
                    f"Relationship {i + 1}/{relationship_count}: "
                    f"{rel.subject} -> {rel.predicate} -> {rel.object}"
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
            logger.exception(f"Error in Gemini extraction: {str(e)}")
            return []
