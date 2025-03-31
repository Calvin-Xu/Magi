"""OpenAI service for API interactions and model management."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import tiktoken
from openai import OpenAI

from magi.config import OPENAI_CONFIG
from magi.services.rate_limiter import RateLimit, rate_limiter
from magi.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ModelLimits:
    """Model-specific rate and token limits."""

    rpm: int
    tpm: int
    input_token_limit: int
    max_concurrent: int


# Model-specific limits
MODEL_LIMITS: Dict[str, ModelLimits] = {
    "o3-mini-2025-01-31": ModelLimits(
        rpm=3500, tpm=350000, input_token_limit=30000, max_concurrent=100
    ),
    "gpt-4o-2024-11-20": ModelLimits(
        rpm=10000, tpm=300000, input_token_limit=128000, max_concurrent=100
    ),
}


# Class-level tokenizer cache shared across OpenAI implementations
_TOKENIZER_CACHE = {}


def get_tokenizer(model: str) -> tiktoken.Encoding:
    """Get a tokenizer for the specified model.

    Args:
        model: OpenAI model name

    Returns:
        tiktoken Encoding for the model
    """
    if model not in _TOKENIZER_CACHE:
        try:
            _TOKENIZER_CACHE[model] = tiktoken.encoding_for_model(model)
            logger.info(f"Created tokenizer for model: {model}")
        except KeyError:
            logger.warning(
                f"No tokenizer found for {model}, falling back to cl100k_base"
            )
            _TOKENIZER_CACHE[model] = tiktoken.get_encoding("cl100k_base")

    return _TOKENIZER_CACHE[model]


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text using the model's tokenizer.

    Args:
        text: Text to count tokens for
        model: OpenAI model name

    Returns:
        Token count
    """
    tokenizer = get_tokenizer(model)
    token_count = len(tokenizer.encode(text))
    logger.debug(f"Text contains {token_count} tokens")
    return token_count


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client with the given API key.

    Args:
        api_key: OpenAI API key, defaults to config

    Returns:
        OpenAI client

    Raises:
        ValueError: If no API key is provided or configured
    """
    api_key = api_key or OPENAI_CONFIG.api_key
    if not api_key:
        raise ValueError("OpenAI API key is required")

    return OpenAI(api_key=api_key)


async def call_openai_with_backoff(
    client: OpenAI,
    model: str,
    messages: list,
    response_format: Any,
    rate_limit: RateLimit,
    max_retries: int = 3,
) -> Any:
    """Call OpenAI API with exponential backoff retry and rate limiting.

    Args:
        client: OpenAI client
        model: Model to use
        messages: Messages for the chat completion
        response_format: Expected response format
        rate_limit: Rate limit object for throttling
        max_retries: Maximum number of retries

    Returns:
        API response

    Raises:
        Exception: If all retries fail
    """
    prompt_text = " ".join([m.get("content", "") for m in messages])
    prompt_hash = hash(prompt_text)
    tokens_needed = count_tokens(prompt_text, model)

    logger.debug(
        f"Calling OpenAI API with prompt hash: {prompt_hash}, tokens: {tokens_needed}"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            # Use the context manager for rate limiting
            logger.debug(
                f"Acquiring rate limit for {tokens_needed} tokens (attempt {attempt + 1})"
            )
            async with rate_limiter.acquire_context(
                rate_limit=rate_limit,
                tokens=tokens_needed,
                reserve=True,
            ) as retry_after:
                if retry_after:
                    # In case we are told to wait until a specific time:
                    wait_seconds = max(0.0, retry_after - datetime.now().timestamp())
                    logger.debug(
                        f"Rate limited, waiting for {wait_seconds:.2f} seconds"
                    )
                    await asyncio.sleep(wait_seconds)
                    continue  # Try again after waiting

                logger.debug(f"Sending request to OpenAI API (attempt {attempt + 1})")

                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )

                logger.debug(
                    f"Successfully received response from OpenAI API (attempt {attempt + 1})"
                )
                return completion

        except Exception as e:
            last_error = e
            logger.error(
                f"Error in OpenAI API call (prompt hash: {prompt_hash}, "
                f"attempt {attempt + 1}): {str(e)}"
            )

            # Attempt naive backoff for rate-limit errors
            if "RateLimitError" in str(e) or "Please reduce your request rate" in str(
                e
            ):
                wait_time = (2**attempt) + (attempt * 0.1)
                logger.info(f"Rate limited. Backing off for {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            elif "InvalidRequestError" in str(e) or "BadRequestError" in str(e):
                # Don't retry malformed requests
                break
            else:
                # Basic exponential backoff for other errors
                wait_time = (2**attempt) * 0.5
                logger.info(f"API error. Backing off for {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    # If we've exhausted retries, raise the last error
    if last_error:
        logger.error(f"Failed after {max_retries} attempts: {last_error}")
        raise last_error

    return None


def get_model_limits(model: str) -> ModelLimits:
    """Get the limits for a specific model, with fallback to a default.

    Args:
        model: OpenAI model name

    Returns:
        ModelLimits for the specified model
    """
    return MODEL_LIMITS.get(
        model,
        MODEL_LIMITS["gpt-4o-2024-11-20"],  # Default to GPT-4o
    )
