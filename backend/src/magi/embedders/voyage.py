import asyncio
from datetime import datetime
from typing import List, Optional

import voyageai

from magi.config import VOYAGE_AI_CONFIG
from magi.services.rate_limiter import RateLimit, rate_limiter

from .base import EmbeddingProvider

from magi.utils import get_logger

logger = get_logger(__name__)


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Concrete implementation of EmbeddingProvider using Voyage AI's voyage-3-large."""

    def __init__(
        self,
        api_key: str = VOYAGE_AI_CONFIG.api_key,
        max_concurrent_requests: int = 40,
    ):
        self.client = voyageai.Client(api_key=api_key)
        self.rate_limiter = rate_limiter
        self.model = "voyage-3-large"
        self.rate_limit = RateLimit(
            name="voyage_3_large_embedding",
            rpm=2000,  # Requests per minute
            tpm=3_000_000,  # Tokens per minute
            window_size=60,
            num_shards=10,
            max_concurrent=max_concurrent_requests,
        )
        self.max_concurrent_requests = max_concurrent_requests

    @property
    def max_batch_size(self) -> int:
        """Maximum length of the input list to embed."""
        return 128

    @property
    def max_batch_tokens(self) -> int:
        """Maximum total number of tokens in the input list."""
        return 120_000

    async def embed(
        self,
        texts: List[str],
        truncation: bool = True,
        output_dimension: int = 1024,
        query_prompt: Optional[str] = None,
        embed_prompt: Optional[str] = None,
        current_max_batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        if query_prompt and embed_prompt:
            raise ValueError(
                "Both query_prompt and embed_prompt cannot be provided simultaneously."
            )

        # Use provided max_batch_size or default
        max_batch_size = current_max_batch_size or self.max_batch_size

        input_type = None
        if query_prompt:
            texts = [query_prompt + text for text in texts]
        elif embed_prompt:
            texts = [embed_prompt + text for text in texts]

        # Rate limiting logic
        total_tokens = self.client.count_tokens(texts, model=self.model)

        # Check if we need to chunk the requests based on batch size or token limits
        if len(texts) > max_batch_size or total_tokens > self.max_batch_tokens:
            # Split texts into smaller chunks respecting both max_batch_size and max_batch_tokens
            chunked_texts = []
            current_chunk = []
            current_tokens = 0
            current_batch_size = 0

            for text in texts:
                text_tokens = self.client.count_tokens([text], model=self.model)

                # If adding this text exceeds either limit, start a new chunk
                if (
                    current_batch_size + 1 > max_batch_size
                    or current_tokens + text_tokens > self.max_batch_tokens
                ):
                    if current_chunk:  # Only append if there's something to append
                        chunked_texts.append(current_chunk)
                    current_chunk = [text]
                    current_tokens = text_tokens
                    current_batch_size = 1
                else:
                    current_chunk.append(text)
                    current_tokens += text_tokens
                    current_batch_size += 1

            # Add the last chunk if it exists
            if current_chunk:
                chunked_texts.append(current_chunk)

            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            # Define a function to embed a single chunk with rate limiting
            async def embed_chunk(chunk, chunk_max_batch_size=max_batch_size):
                async with semaphore:
                    tokens = self.client.count_tokens(chunk, model=self.model)

                    # Log chunk size information for debugging
                    logger.debug(
                        f"Embedding chunk with {len(chunk)} texts and {tokens} tokens (max batch size: {chunk_max_batch_size})"
                    )

                    if len(chunk) > chunk_max_batch_size:
                        raise ValueError(
                            f"Chunk size {len(chunk)} exceeds max batch size {chunk_max_batch_size}"
                        )

                    if tokens > self.max_batch_tokens:
                        raise ValueError(
                            f"Chunk tokens {tokens} exceeds max batch tokens {self.max_batch_tokens}"
                        )

                    # Use context manager for rate limiting
                    async with self.rate_limiter.acquire_context(
                        self.rate_limit, tokens=tokens
                    ) as retry_after:
                        if retry_after:
                            # If rate limited, wait and then try again
                            wait_seconds = max(
                                0.0, retry_after - datetime.now().timestamp()
                            )
                            await asyncio.sleep(wait_seconds)
                            # Recursive call after waiting
                            return await embed_chunk(chunk, chunk_max_batch_size)

                        try:
                            # Use asyncio.to_thread to run the synchronous client.embed in a separate thread
                            result = await asyncio.to_thread(
                                self.client.embed,
                                chunk,
                                model=self.model,
                                input_type=input_type,
                                truncation=truncation,
                                output_dimension=output_dimension,
                            )
                            return result.embeddings
                        except Exception as e:
                            error_message = str(e)
                            if (
                                "Please resubmit with a smaller batch size"
                                in error_message
                            ):
                                logger.info(
                                    f"Batch size error detected. Halving batch size from {chunk_max_batch_size} to {chunk_max_batch_size // 2}"
                                )

                                # If we're already at a batch size of 1, we can't go smaller
                                if len(chunk) <= 1:
                                    raise ValueError(
                                        f"Cannot reduce batch size further, already at minimum with chunk size {len(chunk)}"
                                    )

                                # Split the chunk in half
                                half_size = max(1, len(chunk) // 2)
                                first_half = chunk[:half_size]
                                second_half = chunk[half_size:]

                                # Process each half with the reduced max batch size
                                new_max_batch_size = max(1, chunk_max_batch_size // 2)
                                first_result = await embed_chunk(
                                    first_half, new_max_batch_size
                                )
                                second_result = await embed_chunk(
                                    second_half, new_max_batch_size
                                )

                                # Combine results
                                return first_result + second_result
                            else:
                                # Re-raise other exceptions
                                raise

            # Create tasks for all chunks
            embedding_tasks = [embed_chunk(chunk) for chunk in chunked_texts]

            # Execute all tasks concurrently and gather results
            chunk_embeddings = await asyncio.gather(*embedding_tasks)

            # Flatten the results
            all_embeddings = [
                embedding
                for chunk_result in chunk_embeddings
                for embedding in chunk_result
            ]

            return all_embeddings
        else:
            # If within limits, proceed with a single request
            async with self.rate_limiter.acquire_context(
                self.rate_limit, tokens=total_tokens
            ) as retry_after:
                if retry_after:
                    # If rate limited, wait and then try again
                    wait_seconds = max(0.0, retry_after - datetime.now().timestamp())
                    await asyncio.sleep(wait_seconds)
                    # Recursive call after waiting
                    return await self.embed(
                        texts,
                        truncation=truncation,
                        output_dimension=output_dimension,
                        query_prompt=query_prompt,
                        embed_prompt=embed_prompt,
                        current_max_batch_size=max_batch_size,
                    )

                try:
                    # Use asyncio.to_thread to run the synchronous client.embed in a separate thread
                    result = await asyncio.to_thread(
                        self.client.embed,
                        texts,
                        model=self.model,
                        input_type=input_type,
                        truncation=truncation,
                        output_dimension=output_dimension,
                    )
                    return result.embeddings
                except Exception as e:
                    error_message = str(e)
                    if "Please resubmit with a smaller batch size" in error_message:
                        logger.info(
                            f"Batch size error detected. Halving batch size from {max_batch_size} to {max_batch_size // 2}"
                        )
                        # Retry with half the batch size
                        new_max_batch_size = max(1, max_batch_size // 2)
                        return await self.embed(
                            texts,
                            truncation=truncation,
                            output_dimension=output_dimension,
                            query_prompt=query_prompt,
                            embed_prompt=embed_prompt,
                            current_max_batch_size=new_max_batch_size,
                        )
                    else:
                        # Re-raise other exceptions
                        raise
