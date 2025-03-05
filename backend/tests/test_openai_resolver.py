"""
Test script for the OpenAIResolver implementation.

This script demonstrates how to use the OpenAIResolver for entity and relationship type resolution.
"""

import asyncio
from magi.services.create_tables import reset_database
from dotenv import load_dotenv
import asyncpg
import pytest

from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.resolvers.openai_resolver import OpenAIResolver
from magi.services.models import Entity, RelationshipType
from magi.config import POSTGRES_CONFIG, VOYAGE_AI_CONFIG, OPENAI_CONFIG


# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_entity_resolution():
    """Test entity resolution with OpenAIResolver."""
    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )

    await reset_database()

    try:
        # Initialize embedding provider
        embedding_provider = VoyageEmbeddingProvider(
            api_key=VOYAGE_AI_CONFIG["api_key"],
        )

        # Initialize OpenAI resolver for entities
        entity_resolver = OpenAIResolver(
            conn=conn,
            embedding_provider=embedding_provider,
            table_name="entities",
            reference_column="id",
            similarity_threshold=0.8,
            max_tokens_per_batch=4000,
            model="gpt-4o",
            temperature=0.0,
            api_key=OPENAI_CONFIG["api_key"],
        )

        # Create test entities
        entities_1 = [
            Entity(
                name="Gaius Julius Caesar Augustus",
                description="Gaius Julius Caesar Augustus (born Gaius Octavius; 63 BC - AD 14), also known as Octavian, was the first Roman emperor and founder of the Principate. His reign (27 BC - AD 14) initiated the Pax Romana, and he was succeeded by Tiberius.",
            ),
            Entity(
                name="Roman Empire",
                description="The Roman Empire (27 BC - AD 476 in the West, 27 BC - AD 1453 in the East) was a vast territorial empire that succeeded the Roman Republic. It was marked by imperial rule and significant territorial expansion.",
            ),
            Entity(
                name="Octavian",
                description="Octavian, later Augustus, was Caesar's adopted heir and Rome's first emperor.",
            ),
        ]

        # Resolve entities
        resolved_entities = await entity_resolver.resolve(entities_1)

        # Print results
        print("\nResolved Entities:")
        for i, entity in enumerate(resolved_entities):
            print(f"\nEntity {i + 1}:")
            print(f"  Original Name: {entities_1[i].name}")
            print(f"  Original Description: {entities_1[i].description}")
            print(f"  Resolved Name: {entity.name}")
            print(f"  Resolved Description: {entity.description}")
            print(f"  Database Reference: {entity.postgres_reference}")

        entities_2 = [
            Entity(
                name="Octavian",
                description="Octavian, later Augustus, was Caesar's adopted heir and Rome's first emperor.",
            ),
        ]

        # Resolve entities
        resolved_entities = await entity_resolver.resolve(entities_2)

        # Print results
        print("\nResolved Entities:")
        for i, entity in enumerate(resolved_entities):
            print(f"\nEntity {i + 1}:")
            print(f"  Original Name: {entities_2[i].name}")
            print(f"  Original Description: {entities_2[i].description}")
            print(f"  Resolved Name: {entity.name}")
            print(f"  Resolved Description: {entity.description}")
            print(f"  Database Reference: {entity.postgres_reference}")

    finally:
        await reset_database()
        await conn.close()


@pytest.mark.asyncio
async def test_relationship_type_resolution():
    """Test relationship type resolution with OpenAIResolver."""
    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )

    await reset_database()

    try:
        # Initialize embedding provider
        embedding_provider = VoyageEmbeddingProvider(
            api_key=VOYAGE_AI_CONFIG["api_key"],
        )

        # Initialize OpenAI resolver for relationship types
        rel_type_resolver = OpenAIResolver(
            conn=conn,
            embedding_provider=embedding_provider,
            table_name="relationship_types",
            reference_column="id",
            similarity_threshold=0.8,
            max_tokens_per_batch=4000,
            model="gpt-4o",
            temperature=0.0,
            api_key=OPENAI_CONFIG["api_key"],
        )

        # Create test relationship types
        rel_types = [
            RelationshipType(
                name="is a subsidiary of",
                description="Indicates that one company is owned and controlled by another company.",
            ),
            RelationshipType(
                name="competes with",
                description="Indicates that two companies are competitors in the same market.",
            ),
            RelationshipType(
                name="is owned by",
                description="Indicates that a company is owned by another company.",
            ),
        ]

        # Resolve relationship types
        resolved_rel_types = await rel_type_resolver.resolve(rel_types)

        # Print results
        print("\nResolved Relationship Types:")
        for i, rel_type in enumerate(resolved_rel_types):
            print(f"\nRelationship Type {i + 1}:")
            print(f"  Original Name: {rel_types[i].name}")
            print(f"  Original Description: {rel_types[i].description}")
            print(f"  Resolved Name: {rel_type.name}")
            print(f"  Resolved Description: {rel_type.description}")
            print(f"  Database Reference: {rel_type.postgres_reference}")

    finally:
        await reset_database()
        await conn.close()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_entity_resolution())
    # asyncio.run(test_relationship_type_resolution())
