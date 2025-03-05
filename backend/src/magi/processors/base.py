"""Base classes for document processors."""

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class DocumentProcessor(ABC):
    """Abstract base class for document processors that extract information."""

    @abstractmethod
    async def process(self, df: DataFrame) -> DataFrame:
        """
        Process a DataFrame of documents.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        pass
