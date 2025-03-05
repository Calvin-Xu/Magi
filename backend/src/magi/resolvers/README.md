# Magi Entity Resolution System

The Magi entity resolution system is a robust and flexible framework for resolving entities and relationship types using advanced embedding and LLM techniques.

## Overview

The system uses a combination of embedding similarity and LLM verification to determine if two objects (entities or relationship types) represent the same real-world concept. The resolution process follows these steps:

1. Compute embeddings for input objects
2. Find similar objects in the database using vector similarity
3. Use an LLM to verify if the objects are the same and generate updated information
4. Update existing database entries or create new ones
5. Return resolved objects with database references

## Key Components

### Resolver

The `Resolver` is a generic abstract base class that defines the interface and common functionality for entity and relationship type resolution. It is parameterized with a type `T` that can be either `Entity` or `RelationshipType`.

```python
class Resolver(Generic[T], ABC):
    """
    Abstract base class for resolving objects (entities or relationship types).
    
    Generic type T must be either Entity or RelationshipType.
    """
```

### OpenAIResolver

The `OpenAIResolver` is a concrete implementation of the `Resolver` class that uses OpenAI's LLM capabilities to verify object similarity and generate updated information.

```python
class OpenAIResolver(Resolver[T]):
    """
    Resolver implementation using OpenAI's LLM for verification.
    """
```

## Usage Examples

### Entity Resolution

```python
# Create an embedding provider
embedding_provider = VoyageEmbeddingProvider()

# Create an OpenAI resolver for entities
entity_resolver = OpenAIResolver[Entity](
    conn=conn,
    embedding_provider=embedding_provider,
    table_name="entities",
    reference_column="id",
    similarity_threshold=0.8,
)

# Example entities to resolve
entities = [
    Entity(
        name="Apple Inc.",
        description="American multinational technology company headquartered in Cupertino, California.",
    ),
    Entity(
        name="Apple",
        description="A technology company that makes iPhones, Macs, and other consumer electronics.",
    ),
]

# Resolve the entities
resolved_entities = await entity_resolver.resolve(entities)
```

### Relationship Type Resolution

```python
# Create an OpenAI resolver for relationship types
rel_type_resolver = OpenAIResolver[RelationshipType](
    conn=conn,
    embedding_provider=embedding_provider,
    table_name="relationship_types",
    reference_column="id",
    similarity_threshold=0.8,
)

# Example relationship types to resolve
relationship_types = [
    RelationshipType(
        name="manufactures",
        description="A company produces or creates a product or service.",
    ),
    RelationshipType(
        name="produces",
        description="An entity creates or makes something as part of its operations.",
    ),
]

# Resolve the relationship types
resolved_rel_types = await rel_type_resolver.resolve(relationship_types)
```

### Integration with Pipeline

The entity resolution system integrates with the Magi pipeline through the `process_and_store_relationships` method:

```python
# Process and store relationships
relationships_with_refs, relationship_ids = await pipeline.process_and_store_relationships(
    extracted_relationships_df=relationships_df,
    embedding_provider=embedding_provider,
    entity_resolver=entity_resolver,
    rel_type_resolver=rel_type_resolver,
    conn=conn,
)
```

## Key Features

- **Generic Type Support**: Works with both `Entity` and `RelationshipType` objects
- **Embedding-based Similarity**: Uses vector similarity to find potential matches
- **LLM Verification**: Uses LLM to verify if objects are the same concept
- **Batch Processing**: Handles objects in batches for efficiency
- **Error Handling**: Robust error handling for LLM responses
- **Asynchronous Design**: Fully asynchronous implementation for performance

## Implementation Details

- Embedding input format: `{name}: {description}`
- Default similarity threshold: 0.8 (configurable)
- Batch size management to avoid token limits
- JSON parsing with error recovery
- Comprehensive type hinting
