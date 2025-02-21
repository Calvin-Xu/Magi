"""File processing utilities."""

import asyncio
import json
from typing import Optional
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ..extractors.gemini import GeminiExtractor


async def extract_relationships_from_text(
    text: str,
    extractor: GeminiExtractor,
    source_uri: str,
) -> list[dict]:
    """Extract relationships from text using the provided extractor."""
    relationships = []
    async for rel in extractor.extract_relationships(text):
        relationships.append(rel.model_dump())
    return relationships


# Cache extractor per process
def get_extractor(model: str) -> GeminiExtractor:
    """Get or create an extractor instance."""
    return GeminiExtractor(model=model)


def create_relationship_extractor_udf(model: str = "gemini-2.0-flash"):
    """Create a Spark UDF for relationship extraction."""

    def extract_relationships(text: str, uri: str) -> Optional[str]:
        """Wrapper for async extraction that returns JSON string."""
        # Create new event loop and extractor for this executor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Get cached extractor instance
            extractor = get_extractor(model)

            # Run extraction
            relationships = loop.run_until_complete(
                extract_relationships_from_text(text, extractor, uri)
            )

            # Always return a valid JSON array, even if empty
            return json.dumps(relationships or [])

        except Exception as e:
            print(f"Error in relationship extraction: {str(e)}")
            return json.dumps([])  # Return empty array instead of None
        finally:
            loop.close()

    return F.udf(extract_relationships, StringType())
