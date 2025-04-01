import abc
from typing import List, Optional

import asyncpg

from magi.services.models import ExtractedRelationship
from magi.utils import get_logger

logger = get_logger(__name__)


class GraphAugmenter(abc.ABC):
    """Abstract base class for graph augmentation with domain knowledge.

    GraphAugmenters use domain-specific research to expand a schema graph
    with new entities and relationships discovered through research,
    creating a hybrid schema-knowledge graph.
    """

    @abc.abstractmethod
    async def create_context(self, conn: asyncpg.Connection) -> str:
        """Create a context string by analyzing the schema graph.

        Args:
            conn: Database connection to fetch schema entities and relationships

        Returns:
            A formatted context string describing the schema for research
        """
        pass

    @abc.abstractmethod
    async def get_augmented_relationships(
        self, context: str, user_instruction: Optional[str] = None, **kwargs
    ) -> List[ExtractedRelationship]:
        """Returns new relationships discovered through research.

        This method performs domain-specific research based on the schema context
        to discover new entities and relationships.

        Args:
            context: Schema context string created by create_context
            user_instruction: Optional user guidance for research focus

        Returns:
            List of new relationships discovered through research
        """
        pass
