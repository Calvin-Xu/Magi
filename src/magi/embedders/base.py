from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Maximum length of the input list to embed."""
        pass

    @property
    @abstractmethod
    def max_batch_tokens(self) -> int:
        """Maximum total number of tokens in the input list."""
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        truncation: bool,
        output_dimension: int,
        query_prompt: Optional[str] = None,
        embed_prompt: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
