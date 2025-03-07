"""
OpenAI implementation of the Resolver class for entity and relationship type resolution.
This resolver uses OpenAI's LLM capabilities to verify if objects are the same.
"""

import asyncio
import random
from typing import Dict, List, Set, TypeVar

import tiktoken
from openai import OpenAI

from magi.config import OPENAI_CONFIG
from magi.services.models import Entity, RelationshipType
from magi.services.rate_limiter import RateLimit, rate_limiter
from magi.utils import get_logger

from .base import Resolver
from .models import (
    LLMVerificationResponse,
    ObjectPair,
)
from .models import VerificationResult as ModelVerificationResult

logger = get_logger(__name__)
T = TypeVar("T", Entity, RelationshipType)


class OpenAIResolver(Resolver[T]):
    """
    OpenAI implementation of the Resolver class.

    This resolver uses OpenAI's LLM capabilities to verify if objects are the same
    and to generate updated names and descriptions when needed.
    """

    def __init__(
        self,
        conn,
        embedding_provider,
        table_name: str,
        reference_column: str = "id",
        similarity_threshold: float = 0.4,
        max_tokens_per_batch: int = 4000,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: str = OPENAI_CONFIG.api_key,
        max_retries: int = 5,
    ):
        """
        Initialize the OpenAI resolver.

        Args:
            conn: asyncpg connection
            embedding_provider: Provider for computing embeddings
            table_name: Name of the table to search in (e.g., 'entities', 'relationship_types')
            reference_column: Name of the column that serves as a reference (e.g., 'id')
            similarity_threshold: Threshold for considering objects as similar (0-1)
            max_tokens_per_batch: Maximum number of tokens per LLM batch
            model: OpenAI model to use
            temperature: Temperature for LLM generation (0-1)
            api_key: OpenAI API key
            max_retries: Maximum number of retries for API calls (default: 5)
        """
        super().__init__(
            conn,
            embedding_provider,
            table_name,
            reference_column,
            similarity_threshold,
            max_tokens_per_batch,
        )
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
        # Initialize the tokenizer for the specified model
        self.tokenizer = (
            tiktoken.encoding_for_model(model)
            if model.startswith("gpt")
            else tiktoken.get_encoding("cl100k_base")
        )
        # Reserve tokens for system message, response format, and some overhead
        self.reserved_tokens = 500

        # Initialize rate limiter
        self._rate_limiter = rate_limiter

        # TODO: make rate limits configurable
        self._rate_limit = RateLimit(  # Tier 4
            name="gpt-4o",
            rpm=10000,  # 10,000 requests per minute
            tpm=2_000_000,  # 2,000,000 tokens per minute
            window_size=60,
            num_shards=10,
            max_concurrent=self.max_concurrent_requests,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    async def _create_verification_batches(
        self, object_pairs: List[ObjectPair]
    ) -> List[List[ObjectPair]]:
        """
        Group object pairs into batches to avoid token limit.

        Args:
            object_pairs: List of ObjectPair instances

        Returns:
            List of batches, where each batch is a list of object pairs
        """
        batches = []
        current_batch = []
        current_token_count = 0

        for pair in object_pairs:
            # Skip pairs where no similar object was found
            if pair.similar_object is None:
                batches.append([pair])
                continue

            # Create a sample prompt for this pair to estimate token count
            sample_prompt = self._create_verification_prompt([pair])
            token_count = self._count_tokens(sample_prompt)

            # If adding this pair would exceed the token limit, start a new batch
            if (
                current_token_count + token_count + self.reserved_tokens
                > self.max_tokens_per_batch
                and current_batch
            ):
                batches.append(current_batch)
                current_batch = [pair]
                current_token_count = token_count
            else:
                current_batch.append(pair)
                current_token_count += token_count

        # Add the last batch if it exists
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _verify_objects_batch(
        self, batch: List[ObjectPair]
    ) -> List[ModelVerificationResult]:
        """
        Verify if the retrieved objects are the same as the input objects using OpenAI.

        Args:
            batch: List of ObjectPair instances

        Returns:
            List of verification results
        """
        # Filter out pairs where no similar object was found
        pairs_to_verify = [pair for pair in batch if pair.similar_object is not None]

        # If there are no pairs to verify, return results immediately
        if not pairs_to_verify:
            return [
                ModelVerificationResult(
                    pair=pair,
                    are_same=False,
                    updated_name=None,
                    updated_description=None,
                )
                for pair in batch
            ]

        # Create a mapping from pair_id to the original pair for easy lookup
        pair_id_to_pair: Dict[int, ObjectPair] = {
            i: pair for i, pair in enumerate(pairs_to_verify)
        }

        # Prepare the prompt for the LLM
        prompt = self._create_verification_prompt(pairs_to_verify)

        # Count tokens to ensure we're within limits
        token_count = self._count_tokens(prompt)
        if token_count + self.reserved_tokens > self.max_tokens_per_batch:
            logger.warning(
                f"Prompt token count ({token_count}) is close to or exceeds the limit. Consider reducing batch size."
            )

        try:
            # Apply rate limiting using context manager
            async with self._rate_limiter.acquire_context(
                rate_limit=self._rate_limit,
                tokens=token_count
                + self.reserved_tokens,  # Include reserved tokens in the count
                reserve=True,
            ) as retry_after:
                # Implement retry logic for API calls
                response = None
                last_exception = None

                for attempt in range(self.max_retries):
                    try:
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
                            # After waiting, try to acquire the rate limit again
                            return await self._verify_objects_batch(batch)

                        if attempt > 0:
                            logger.info(
                                f"Retry attempt {attempt}/{self.max_retries} for OpenAI API call"
                            )
                            # Exponential backoff: wait 2^retry_count seconds before retrying
                            await asyncio.sleep(2**attempt)

                        # Call the OpenAI API with structured output
                        response = await asyncio.to_thread(
                            self.client.beta.chat.completions.parse,
                            model=self.model,
                            temperature=self.temperature,
                            response_format=LLMVerificationResponse,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that analyzes objects to determine if they are the same entity or concept.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                        )

                        # If we got here, the API call was successful
                        break

                    except Exception as e:
                        last_exception = e
                        logger.warning(
                            f"OpenAI API call failed (attempt {attempt}/{self.max_retries}): {str(e)}"
                        )

                # If we didn't get a response after all retries, raise the last exception
                if response is None:
                    if last_exception:
                        raise last_exception
                    else:
                        raise RuntimeError("Failed to get response from OpenAI API")

            # Process the results
            verification_results = response.choices[0].message.parsed
            logger.info(f"Verification results: {verification_results}")
            processed_results = []

            # Keep track of which pairs have been processed
            processed_pairs: Set[int] = set()

            # Process verified pairs
            for result in verification_results.results:
                pair_id = result.pair_id

                # Validate pair_id is in range
                if pair_id not in pair_id_to_pair:
                    logger.warning(f"Received result for invalid pair_id: {pair_id}")
                    continue

                pair = pair_id_to_pair[pair_id]
                processed_pairs.add(pair_id)

                processed_results.append(
                    ModelVerificationResult(
                        pair=pair,
                        are_same=result.are_same,
                        updated_name=result.updated_name,
                        updated_description=result.updated_description,
                    )
                )

            # Check if all pairs were processed
            if len(processed_pairs) < len(pairs_to_verify):
                missing_pairs = set(pair_id_to_pair.keys()) - processed_pairs
                logger.warning(
                    f"LLM failed to verify all pairs. Missing pairs: {missing_pairs}"
                )

            # Add results for input objects that were not processed
            for i, pair in enumerate(batch):
                if pair.similar_object is None or (
                    pair.similar_object is not None and i not in processed_pairs
                ):
                    processed_results.append(
                        ModelVerificationResult(
                            pair=pair,
                            are_same=False,
                            updated_name=None,
                            updated_description=None,
                        )
                    )

            return processed_results

        except Exception as e:
            logger.error(f"Error verifying objects with OpenAI: {e}")
            # Return default results for all pairs in case of error
            return [
                ModelVerificationResult(
                    pair=pair,
                    are_same=False,
                    updated_name=None,
                    updated_description=None,
                )
                for pair in batch
            ]

    async def close(self):
        """Close rate limiter resources."""
        await self._rate_limiter.close()
