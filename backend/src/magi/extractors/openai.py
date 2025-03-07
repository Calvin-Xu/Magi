"""OpenAI-based relationship extractor."""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import openai
import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from magi.config import OPENAI_CONFIG
from magi.services.rate_limiter import RateLimit, rate_limiter
from magi.utils.logging import get_logger, log_async_function_call

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
    input_token_limit: int  # Maximum input tokens per request
    max_concurrent: int  # Max concurrent calls in the rate limiter


# Model-specific limits
MODEL_LIMITS: Dict[str, ModelLimits] = {
    "o3-mini-2025-01-31": ModelLimits(
        rpm=10_000,
        tpm=10_000_000,
        input_token_limit=8192,
        max_concurrent=16,
    ),
    "gpt-4o-2024-11-20": ModelLimits(
        rpm=10_000,
        tpm=2_000_000,
        input_token_limit=8192,
        max_concurrent=16,
    ),
}


#
# Pydantic schema for the JSON-based response from OpenAI.
# We'll prompt the model to return data in this structure.
#
class RelationshipItem(BaseModel):
    subject: str
    subject_description: str
    object: str
    object_description: str
    predicate: str
    predicate_description: str
    constraint_condition: str
    reason: str
    is_causal: bool


class RelationshipList(BaseModel):
    """Schema for the list of extracted relationships."""

    relationships: List[RelationshipItem]


class OpenAIError(Exception):
    """Base error for OpenAI API calls."""

    pass


class OpenAIExtractor(RelationshipExtractor):
    """Relationship extractor using OpenAI's ChatCompletion API and JSON Schema."""

    # Class-level tokenizer cache
    _tokenizer_cache = {}

    def __init__(
        self,
        model: str,
        openai_api_key: str = OPENAI_CONFIG.api_key,
        max_concurrent_requests: int = 10000,
        max_retries: int = 5,
    ):
        """
        Initialize the OpenAI-based extractor.

        Args:
            model: Model name (e.g. "gpt-4o-2024-08-06") that supports JSON schema.
            openai_api_key: Your OpenAI API key.
            max_concurrent_requests: Upper bound on concurrency for this class (extra layer).
            max_retries: Number of times to retry on transient errors.
        """

        if model not in MODEL_LIMITS:
            logger.error(f"Unsupported model or missing config: {model}")
            raise ValueError(f"Unsupported model or missing config: {model}")

        # Get known constraints
        limits = MODEL_LIMITS[model]

        super().__init__(
            model=model,
            max_input_tokens=limits.input_token_limit,
            max_concurrent_requests=max_concurrent_requests,
        )
        openai.api_key = openai_api_key

        # For structured outputs
        self._client = OpenAI(api_key=openai_api_key)

        # A simple distributed rate limiter from your codebase
        self._rate_limiter = rate_limiter
        self._rate_limit = RateLimit(
            name=model,
            rpm=limits.rpm,
            tpm=limits.tpm,
            window_size=60,
            num_shards=10,
            max_concurrent=limits.max_concurrent,
        )

        self.max_retries = max_retries

        # Prepare tiktoken for counting - use class-level cache
        if model not in OpenAIExtractor._tokenizer_cache:
            try:
                OpenAIExtractor._tokenizer_cache[model] = tiktoken.encoding_for_model(
                    model
                )
                logger.info(f"Using cached tokenizer for model: {model}")
            except KeyError:
                logger.warning(
                    f"No tokenizer found for {model}, falling back to cl100k_base"
                )
                OpenAIExtractor._tokenizer_cache[model] = tiktoken.get_encoding(
                    "cl100k_base"
                )

        self._tokenizer = OpenAIExtractor._tokenizer_cache[model]

        logger.info(
            f"Initialized OpenAIExtractor with model: {model}, max_concurrent: {max_concurrent_requests}"
        )

    async def close(self):
        """Clean up any async resources (e.g. distributed rate limiter)."""
        logger.info("Closing OpenAIExtractor resources")
        await self._rate_limiter.close()

    #
    # Token counting
    #
    async def _count_tokens(self, text: str) -> int:
        # We do a sync approach inside an async function because Tiktoken is synchronous.
        return self._count_tokens_sync(text)

    def _count_tokens_sync(self, text: str) -> int:
        """Synchronously count tokens in `text` using tiktoken."""
        token_count = len(self._tokenizer.encode(text))
        logger.debug(f"Text contains {token_count} tokens")
        return token_count

    #
    # Actual relationship extraction from a single chunk of text
    #
    @log_async_function_call(logger)
    async def _extract_relationships_raw(
        self,
        text: str,
        **kwargs,
    ) -> List[RelationshipTriple]:
        """Extract relationships from a single chunk of text."""
        start_time = datetime.now()
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(text=text)
        text_hash = hash(text)
        logger.info(
            f"Extracting relationships from text (hash: {text_hash}, length: {len(text)})"
        )

        try:
            # Make the call with retries + rate limiting
            logger.debug(f"Calling OpenAI API for text hash: {text_hash}")
            response = await self._call_openai_api(prompt)
            if not response:
                logger.warning(f"No valid response for text hash: {text_hash}")
                return []

            # The parse() method returns an object with .choices[0].message.parsed
            # which should be a RelationshipList pydantic object, or a refusal.
            result = response.choices[0].message

            if result.refusal:
                # The model refused for safety reasons or other policy
                logger.warning(
                    f"Model refused to answer for text hash: {text_hash}. Reason: {result.refusal}"
                )
                return []

            if not result.parsed:
                logger.warning(
                    f"No parsed object from the model for text hash: {text_hash}"
                )
                return []

            print(f"Response: {result.parsed}")

            # Everything looks good. `parsed` is our RelationshipList
            extracted_list: RelationshipList = result.parsed
            logger.info(
                f"Successfully extracted {len(extracted_list.relationships)} relationships from text hash: {text_hash}"
            )

            # Record usage metrics
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # The usage object has token counts for prompt & completion
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            logger.debug(
                f"Extraction metrics - Duration: {duration_ms:.2f}ms, "
                f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}"
            )

            self._metrics.append(
                ExtractionMetrics(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    duration_ms=duration_ms,
                    timestamp=start_time,
                )
            )

            # Convert RelationshipItem -> RelationshipTriple
            relationships_out: List[RelationshipTriple] = []
            for item in extracted_list.relationships:
                relationships_out.append(
                    RelationshipTriple(
                        from_entity=item.subject,
                        from_entity_description=item.subject_description,
                        to_entity=item.object,
                        to_entity_description=item.object_description,
                        relationship_type=item.predicate,
                        relationship_description=item.predicate_description,
                        constraint_condition=(
                            ""
                            if item.constraint_condition.lower() == "none"
                            else item.constraint_condition
                        ),
                        reason=item.reason,
                        is_causal=item.is_causal,
                    )
                )

            return relationships_out

        except Exception as e:
            logger.exception(
                f"Error extracting relationships for text hash {text_hash}: {str(e)}"
            )
            return []

    #
    # Internal method to handle backoff & rate-limiting
    #
    @log_async_function_call(logger)
    async def _call_openai_api(self, prompt: str):
        """
        Call the OpenAI ChatCompletion with structured output constraints.
        Raises OpenAIError on repeated failures.
        """
        last_error = None
        prompt_hash = hash(prompt)
        tokens_needed = self._count_tokens_sync(prompt)
        logger.debug(
            f"Calling OpenAI API with prompt hash: {prompt_hash}, tokens: {tokens_needed}"
        )

        for attempt in range(self.max_retries):
            try:
                # Use the context manager for rate limiting
                logger.debug(
                    f"Acquiring rate limit for {tokens_needed} tokens (attempt {attempt + 1})"
                )
                async with self._rate_limiter.acquire_context(
                    rate_limit=self._rate_limit,
                    tokens=tokens_needed,
                    reserve=True,
                ) as retry_after:
                    if retry_after:
                        # In case we are told to wait until a specific time:
                        wait_seconds = max(
                            0.0, retry_after - asyncio.get_event_loop().time()
                        )
                        # if attempt == 0:
                        #     # Add a small random initial jitter
                        #     wait_seconds += random.random() * 5
                        logger.debug(
                            f"Rate limited, waiting for {wait_seconds:.2f} seconds"
                        )
                        await asyncio.sleep(wait_seconds)
                        continue  # Try again after waiting

                    # Use the new structured output parse API
                    logger.debug(
                        f"Sending request to OpenAI API (attempt {attempt + 1})"
                    )
                    completion = self._client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a relationship extraction assistant.",
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        response_format=RelationshipList,
                    )
                    logger.debug(
                        f"Successfully received response from OpenAI API (attempt {attempt + 1})"
                    )
                    return completion

            except Exception as e:
                last_error = e
                logger.error(
                    f"Error in OpenAI API call (prompt hash: {prompt_hash}, attempt {attempt + 1}): {str(e)}"
                )

                # Attempt naive backoff for rate-limit errors
                if "RateLimitError" in str(
                    e
                ) or "Please reduce your request rate" in str(e):
                    # exponential backoff
                    backoff = 30 * (2**attempt)
                    logger.warning(
                        f"Rate limit error, backing off for {backoff} seconds"
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Non-rate-limiting error => break fast
                break

        # If we got here, we failed all attempts
        error_msg = (
            f"Failed after {attempt + 1} attempts. Last error: {str(last_error)}"
        )
        logger.error(error_msg)
        raise OpenAIError(error_msg)
