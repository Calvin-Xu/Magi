"""
Test script for the OpenAIResolver implementation.

This script demonstrates how to use the OpenAIResolver for entity and relationship type resolution.
"""

import asyncio
from hashlib import md5

import asyncpg
import pytest
from dotenv import load_dotenv

from magi.config import OPENAI_CONFIG, POSTGRES_CONFIG, VOYAGE_AI_CONFIG
from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.resolvers.openai_resolver import OpenAIResolver
from magi.services.create_tables import reset_database
from magi.services.models import Entity

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

    try:
        await reset_database(conn)

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
                hash_key=md5("Gaius Julius Caesar Augustus".encode()).hexdigest(),
            ),
            Entity(
                name="Roman Empire",
                description="The Roman Empire (27 BC - AD 476 in the West, 27 BC - AD 1453 in the East) was a vast territorial empire that succeeded the Roman Republic. It was marked by imperial rule and significant territorial expansion.",
                hash_key=md5("The Roman Empire".encode()).hexdigest(),
            ),
            Entity(
                name="Octavian",
                description="Octavian, later Augustus, was Caesar's adopted heir and Rome's first emperor.",
                hash_key=md5("Octavian".encode()).hexdigest(),
            ),
        ]

        # Convert list to dictionary with hash keys
        entities_dict_1 = {f"entity_{i}": entity for i, entity in enumerate(entities_1)}

        # Resolve entities
        resolved_entities = await entity_resolver.resolve(entities_dict_1)

        # Print results
        print("\nResolved Entities:")
        for i, (hash_key, entity) in enumerate(resolved_entities.items()):
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
                hash_key=md5("Octavian".encode()).hexdigest(),
            ),
        ]

        # Convert list to dictionary with hash keys
        entities_dict_2 = {"entity_test": entities_2[0]}

        # Resolve entities
        resolved_entities = await entity_resolver.resolve(entities_dict_2)

        # Print results
        print("\nResolved Entities:")
        for i, (hash_key, entity) in enumerate(resolved_entities.items()):
            print(f"\nEntity {i + 1}:")
            print(f"  Original Name: {entities_2[i].name}")
            print(f"  Original Description: {entities_2[i].description}")
            print(f"  Resolved Name: {entity.name}")
            print(f"  Resolved Description: {entity.description}")
            print(f"  Database Reference: {entity.postgres_reference}")

    finally:
        # Close connection
        if not conn.is_closed():
            await conn.close()


# @pytest.mark.asyncio
# async def test_relationship_type_resolution():
#     """Test relationship type resolution with OpenAIResolver."""
#     # Connect to PostgreSQL
#     conn = await asyncpg.connect(
#         host=POSTGRES_CONFIG["host"],
#         port=POSTGRES_CONFIG["port"],
#         user=POSTGRES_CONFIG["user"],
#         password=POSTGRES_CONFIG["password"],
#         database=POSTGRES_CONFIG["database"],
#     )

#     try:
#         await reset_database(conn)

#         # Initialize embedding provider
#         embedding_provider = VoyageEmbeddingProvider(
#             api_key=VOYAGE_AI_CONFIG["api_key"],
#         )

#         # Initialize OpenAI resolver for relationship types
#         rel_type_resolver = OpenAIResolver(
#             conn=conn,
#             embedding_provider=embedding_provider,
#             table_name="relationship_types",
#             reference_column="id",
#             similarity_threshold=0.8,
#             max_tokens_per_batch=4000,
#             model="gpt-4o",
#             temperature=0.0,
#             api_key=OPENAI_CONFIG["api_key"],
#         )

#         # Create test relationship types
#         rel_types = [
#             RelationshipType(
#                 name="is a subsidiary of",
#                 description="Indicates that one company is owned and controlled by another company.",
#             ),
#             RelationshipType(
#                 name="competes with",
#                 description="Indicates that two companies are competitors in the same market.",
#             ),
#             RelationshipType(
#                 name="is owned by",
#                 description="Indicates that a company is owned by another company.",
#             ),
#         ]

#         # Convert list to dictionary with hash keys
#         rel_types_dict = {
#             f"rel_type_{i}": rel_type for i, rel_type in enumerate(rel_types)
#         }

#         # Resolve relationship types
#         resolved_rel_types = await rel_type_resolver.resolve(rel_types_dict)

#         # Print results
#         print("\nResolved Relationship Types:")
#         for i, (hash_key, rel_type) in enumerate(resolved_rel_types.items()):
#             print(f"\nRelationship Type {i + 1}:")
#             print(f"  Original Name: {rel_types[i].name}")
#             print(f"  Original Description: {rel_types[i].description}")
#             print(f"  Resolved Name: {rel_type.name}")
#             print(f"  Resolved Description: {rel_type.description}")
#             print(f"  Database Reference: {rel_type.postgres_reference}")

#         # Test rate limiter close method
#         await rel_type_resolver.close()

#     finally:
#         # Close connection
#         if not conn.is_closed():
#             await conn.close()


# @pytest.mark.asyncio
# async def test_rate_limiting():
#     """Test rate limiting functionality in OpenAIResolver."""
#     # Connect to PostgreSQL
#     conn = await asyncpg.connect(
#         host=POSTGRES_CONFIG["host"],
#         port=POSTGRES_CONFIG["port"],
#         user=POSTGRES_CONFIG["user"],
#         password=POSTGRES_CONFIG["password"],
#         database=POSTGRES_CONFIG["database"],
#     )

#     try:
#         await reset_database(conn)

#         # Initialize embedding provider
#         embedding_provider = VoyageEmbeddingProvider(
#             api_key=VOYAGE_AI_CONFIG["api_key"],
#         )

#         # Initialize OpenAI resolver with custom rate limits for testing
#         resolver = OpenAIResolver(
#             conn=conn,
#             embedding_provider=embedding_provider,
#             table_name="entities",
#             reference_column="id",
#             similarity_threshold=0.8,
#             max_tokens_per_batch=4000,
#             model="gpt-4o",
#             temperature=0.0,
#             api_key=OPENAI_CONFIG["api_key"],
#         )

#         # Override the rate limit for testing
#         test_rate_limit = RateLimit(
#             name="test-limit",
#             rpm=10,  # Very low limit for testing
#             tpm=1000,
#             window_size=60,
#             num_shards=1,
#             max_concurrent=1,
#         )
#         resolver._rate_limit = test_rate_limit

#         # Create a simple entity for testing
#         test_entity = Entity(
#             name="Test Entity",
#             description="This is a test entity for rate limiting.",
#         )

#         # Test that the resolver can handle rate limiting
#         # We're not actually testing the rate limiting itself (would be too slow)
#         # Just ensuring the code path works without errors
#         resolved = await resolver.resolve({"test_entity": test_entity})
#         assert len(resolved) == 1, "Should have resolved one entity"

#         # Clean up
#         await resolver.close()

#     finally:
#         # Close connection
#         if not conn.is_closed():
#             await conn.close()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_entity_resolution())
    # asyncio.run(test_relationship_type_resolution())
    # asyncio.run(test_rate_limiting())
