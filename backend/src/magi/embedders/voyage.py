from typing import List, Optional

import voyageai

from magi.config import VOYAGE_AI_CONFIG
from magi.services.rate_limiter import DistributedRateLimiter, RateLimit
from .base import EmbeddingProvider


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Concrete implementation of EmbeddingProvider using Voyage AI's voyage-3-large."""

    def __init__(self, api_key: str = VOYAGE_AI_CONFIG["api_key"]):
        self.client = voyageai.Client(api_key=api_key)
        self.rate_limiter = DistributedRateLimiter()
        self.model = "voyage-3-large"
        self.rate_limit = RateLimit(
            name="voyage_3_large_embedding",
            rpm=2000,  # Requests per minute
            tpm=3_000_000,  # Tokens per minute
            window_size=60,
            num_shards=10,
            max_concurrent=5,
        )

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
    ) -> List[List[float]]:
        if query_prompt and embed_prompt:
            raise ValueError(
                "Both query_prompt and embed_prompt cannot be provided simultaneously."
            )

        input_type = None
        if query_prompt:
            texts = [query_prompt + text for text in texts]
        elif embed_prompt:
            texts = [embed_prompt + text for text in texts]

        # Rate limiting logic
        total_tokens = self.client.count_tokens(texts, model=self.model)

        if total_tokens > self.max_batch_tokens:
            # Split texts into smaller chunks
            chunked_texts = []
            current_chunk = []
            current_tokens = 0

            for text in texts:
                text_tokens = self.client.count_tokens([text], model=self.model)
                if current_tokens + text_tokens > self.max_batch_tokens:
                    # If adding this text exceeds the limit, start a new chunk
                    chunked_texts.append(current_chunk)
                    current_chunk = [text]
                    current_tokens = text_tokens
                else:
                    current_chunk.append(text)
                    current_tokens += text_tokens

            # Add the last chunk if it exists
            if current_chunk:
                chunked_texts.append(current_chunk)

            # Embed each chunk and combine results
            all_embeddings = []
            for chunk in chunked_texts:
                tokens = self.client.count_tokens(chunk, model=self.model)
                await self.rate_limiter.acquire(self.rate_limit, tokens=tokens)

                result = self.client.embed(
                    chunk,
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation,
                    output_dimension=output_dimension,
                )
                all_embeddings.extend(result.embeddings)

            return all_embeddings
        else:
            # If within limits, proceed with a single request
            await self.rate_limiter.acquire(self.rate_limit, tokens=total_tokens)

            result = self.client.embed(
                texts,
                model=self.model,
                input_type=input_type,
                truncation=truncation,
                output_dimension=output_dimension,
            )
            return result.embeddings
