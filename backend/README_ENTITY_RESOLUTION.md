# Magi Entity Resolution System

The Magi Entity Resolution System is a robust framework for resolving entities and relationship types using embedding similarity and LLM verification. This document explains how to use the system and its key components.

## Overview

The entity resolution system consists of the following key components:

1. **Base Resolver Class**: A generic abstract base class that defines the interface for resolvers.
2. **OpenAI Resolver Implementation**: A concrete implementation using OpenAI's LLM capabilities.
3. **Entity and Relationship Type Models**: Data models for entities and relationship types.
4. **Processing Functions**: Functions for processing and storing resolved objects.

## Key Features

- **Embedding-based Similarity**: Uses vector embeddings to find similar objects in the database.
- **LLM Verification**: Uses LLMs to verify if objects are the same and generate updated information.
- **Batch Processing**: Handles large numbers of objects efficiently through batching.
- **Flexible Architecture**: Supports different resolver implementations through a common interface.
- **Type-safe Implementation**: Uses generic types for strong type checking.

## Usage

### Setting Up Resolvers

```python
from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.resolvers.openai_resolver import OpenAIResolver
from magi.services.models import Entity, RelationshipType

# Create embedding provider
embedding_provider = VoyageEmbeddingProvider(
    api_key="your_voyage_api_key",
    model="voyage-2",
)

# Create OpenAI resolver for entities
entity_resolver = OpenAIResolver(
    conn=conn,  # asyncpg connection
    embedding_provider=embedding_provider,
    table_name="entities",
    reference_column="id",
    similarity_threshold=0.8,
    max_tokens_per_batch=4000,
    model="gpt-4o",
    temperature=0.0,
    api_key="your_openai_api_key",
)

# Create OpenAI resolver for relationship types
rel_type_resolver = OpenAIResolver(
    conn=conn,  # asyncpg connection
    embedding_provider=embedding_provider,
    table_name="relationship_types",
    reference_column="id",
    similarity_threshold=0.8,
    max_tokens_per_batch=4000,
    model="gpt-4o",
    temperature=0.0,
    api_key="your_openai_api_key",
)
```

### Resolving Entities

```python
from magi.services.models import Entity

# Sample entities to resolve
entities = [
    Entity(
        name="Apple Inc.",
        description="A technology company that designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.",
    ),
    Entity(
        name="Microsoft",
        description="A technology company that develops, licenses, and supports software products, services, and devices.",
    ),
]

# Resolve entities
resolved_entities = await entity_resolver.resolve(entities)

# Access resolved entities with database references
for entity in resolved_entities:
    print(f"Name: {entity.name}")
    print(f"Description: {entity.description}")
    print(f"Database Reference: {entity.postgres_reference}")
```

### Resolving Relationship Types

```python
from magi.services.models import RelationshipType

# Sample relationship types to resolve
rel_types = [
    RelationshipType(
        name="manufactures",
        description="The subject entity produces or creates the object entity as a product.",
    ),
    RelationshipType(
        name="competes with",
        description="The subject entity is in direct competition with the object entity in one or more markets.",
    ),
]

# Resolve relationship types
resolved_rel_types = await rel_type_resolver.resolve(rel_types)

# Access resolved relationship types with database references
for rel_type in resolved_rel_types:
    print(f"Name: {rel_type.name}")
    print(f"Description: {rel_type.description}")
    print(f"Database Reference: {rel_type.postgres_reference}")
```

### Using in the Pipeline

```python
from magi.services.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline(spark, credentials, model="gpt-4o")

# Process and store relationships
relationships_with_refs, relationship_ids = await pipeline.process_and_store_relationships(
    extracted_relationships_df,
    embedding_provider,
    entity_resolver,
    rel_type_resolver,
    conn,
)
```

## How It Works

1. **Embedding Computation**: Computes embeddings for input objects using the format `{name}: {description}`.
2. **Similarity Search**: Uses pgvector to find similar objects in the database based on cosine similarity.
3. **Verification**: Groups object pairs into batches and uses an LLM to verify if they are the same.
4. **Resolution**: Updates existing objects or creates new ones based on verification results.
5. **Reference Linking**: Returns resolved objects with database references.

## Customization

You can create your own resolver implementations by inheriting from the `Resolver` abstract base class and implementing the `_verify_objects_batch` method:

```python
from magi.resolvers.base import Resolver
from magi.services.models import Entity

class CustomResolver(Resolver[Entity]):
    async def _verify_objects_batch(self, batch):
        # Implement your custom verification logic
        pass
```

## Testing

Run the test script to see the resolver in action:

```bash
cd backend
python -m tests.test_openai_resolver
```

## Requirements

- Python 3.8+
- asyncpg for database operations
- pgvector for vector similarity search
- OpenAI SDK (latest version)
- Voyage AI embedding provider
- Pydantic for data validation
