"""Base classes for schema builders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from magi.utils.logging import get_logger

logger = get_logger(__name__)


class SchemaBuilder(ABC):
    """Base abstract class for all schema builders.

    SchemaBuilder is responsible for extracting schemas from datasets,
    with concrete implementations handling the specifics of different
    extraction approaches and models.
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
    ):
        """Initialize the schema builder.

        Args:
            model: Model identifier
            max_retries: Maximum number of retries for API calls
        """
        self.model = model
        self.max_retries = max_retries

    @abstractmethod
    async def close(self):
        """Clean up any async resources."""
        pass

    @abstractmethod
    async def extract_schema(
        self,
        dataset_paths: List[str],
        user_prompt: str,
        support_documents: Optional[List[str]] = None,
        max_chunk_columns: int = 50,
    ) -> Any:
        """Extract schema from dataset files.

        Args:
            dataset_paths: List of paths to the dataset files
            user_prompt: User's prompt or description about the dataset
            support_documents: Optional list of paths to support documents
            max_chunk_columns: Maximum number of columns to provide to the LLM at once

        Returns:
            Extracted schema in implementation-specific format
        """
        pass

    @abstractmethod
    async def create_schema_graph(
        self,
        schema: Any,
        conn: Any,
        embedding_provider: Any,
    ) -> Dict[str, int]:
        """Create a schema graph in the database from the extracted schema.

        Args:
            schema: The extracted schema
            conn: Database connection
            embedding_provider: Provider for generating embeddings

        Returns:
            Dictionary with counts of created entities and relationships
        """
        pass
