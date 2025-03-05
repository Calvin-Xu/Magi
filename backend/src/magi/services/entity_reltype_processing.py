import pandas as pd
from ..embedders.voyage import VoyageEmbeddingProvider
from typing import List, Dict, Tuple, TypeVar
from ..services.models import Entity, RelationshipType, Relationship
from ..resolvers import Resolver
import asyncpg

T = TypeVar("T", Entity, RelationshipType)


async def compute_embeddings(
    descriptions: List[str], embedding_provider: VoyageEmbeddingProvider
) -> List[List[float]]:
    """Compute embeddings for a list of descriptions in batches."""
    embeddings = []
    for i in range(0, len(descriptions), embedding_provider.max_batch_size):
        batch = descriptions[i : i + embedding_provider.max_batch_size]
        batch_embeddings = await embedding_provider.embed(
            texts=batch,
            truncation=True,
        )
        embeddings.extend(batch_embeddings)
    return embeddings


async def create_extracted_entities_df(
    extracted_relationships_df: pd.DataFrame,
    embedding_provider: VoyageEmbeddingProvider,
    entity_resolver: Resolver[Entity] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Create extracted entities DataFrame from relationships and resolve against database.

    Args:
        extracted_relationships_df: DataFrame containing extracted relationships
        embedding_provider: Provider for computing embeddings
        entity_resolver: Optional resolver for entity resolution

    Returns:
        Tuple containing:
        - DataFrame of processed entities
        - Mapping from entity hash to database ID
    """
    # Extract unique from_entities
    from_entities = (
        extracted_relationships_df[
            [
                Relationship.FROM_ENTITY_COLUMN,
                Relationship.FROM_ENTITY_DESCRIPTION_COLUMN,
                Relationship.FROM_ENTITY_HASH_COLUMN,
            ]
        ]
        .rename(
            columns={
                Relationship.FROM_ENTITY_COLUMN: Entity.NAME_COLUMN,
                Relationship.FROM_ENTITY_DESCRIPTION_COLUMN: Entity.DESCRIPTION_COLUMN,
                Relationship.FROM_ENTITY_HASH_COLUMN: "hash",
            }
        )
        .drop_duplicates(subset=["hash"])
    )

    # Extract unique to_entities
    to_entities = (
        extracted_relationships_df[
            [
                Relationship.TO_ENTITY_COLUMN,
                Relationship.TO_ENTITY_DESCRIPTION_COLUMN,
                Relationship.TO_ENTITY_HASH_COLUMN,
            ]
        ]
        .rename(
            columns={
                Relationship.TO_ENTITY_COLUMN: Entity.NAME_COLUMN,
                Relationship.TO_ENTITY_DESCRIPTION_COLUMN: Entity.DESCRIPTION_COLUMN,
                Relationship.TO_ENTITY_HASH_COLUMN: "hash",
            }
        )
        .drop_duplicates(subset=["hash"])
    )

    # Combine and deduplicate entities
    all_entities = pd.concat([from_entities, to_entities]).drop_duplicates(
        subset=["hash"]
    )

    # Compute embeddings for all entities
    descriptions = all_entities[Entity.DESCRIPTION_COLUMN].tolist()
    embeddings = await compute_embeddings(descriptions, embedding_provider)

    # Create the DataFrame for entities
    extracted_entities_df = pd.DataFrame(
        {
            Entity.NAME_COLUMN: all_entities[Entity.NAME_COLUMN],
            Entity.DESCRIPTION_COLUMN: all_entities[Entity.DESCRIPTION_COLUMN],
            Entity.EMBEDDING_COLUMN: embeddings,
            Entity.POSTGRES_REFERENCE_COLUMN: None,  # Initially empty
            "hash": all_entities["hash"],
        }
    )

    # Create hash-to-reference mapping
    hash_to_reference = {}

    # If entity resolver is provided, perform entity resolution
    if entity_resolver:
        # Convert DataFrame to list of Entity objects for the resolver
        entities = []
        for _, row in extracted_entities_df.iterrows():
            entity = Entity(
                name=row[Entity.NAME_COLUMN],
                description=row[Entity.DESCRIPTION_COLUMN],
                embedding=row[Entity.EMBEDDING_COLUMN],
            )
            entities.append(entity)

        # Resolve entities
        resolved_entities = await entity_resolver.resolve(entities)

        # Update DataFrame with resolved entities
        for i, resolved_entity in enumerate(resolved_entities):
            extracted_entities_df.at[i, Entity.NAME_COLUMN] = resolved_entity.name
            extracted_entities_df.at[i, Entity.DESCRIPTION_COLUMN] = (
                resolved_entity.description
            )
            extracted_entities_df.at[i, Entity.EMBEDDING_COLUMN] = (
                resolved_entity.embedding
            )
            extracted_entities_df.at[i, Entity.POSTGRES_REFERENCE_COLUMN] = (
                resolved_entity.postgres_reference
            )

            # Update hash-to-reference mapping
            hash_to_reference[extracted_entities_df.at[i, "hash"]] = (
                resolved_entity.postgres_reference
            )

    return extracted_entities_df, hash_to_reference


async def create_extracted_rel_types_df(
    extracted_relationships_df: pd.DataFrame,
    embedding_provider: VoyageEmbeddingProvider,
    rel_type_resolver: Resolver[RelationshipType] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Create extracted relationship types DataFrame from relationships and resolve against database.

    Args:
        extracted_relationships_df: DataFrame containing extracted relationships
        embedding_provider: Provider for computing embeddings
        rel_type_resolver: Optional resolver for relationship type resolution

    Returns:
        Tuple containing:
        - DataFrame of processed relationship types
        - Mapping from relationship type hash to database ID
    """
    unique_rel_types = (
        extracted_relationships_df[
            [
                Relationship.RELATIONSHIP_TYPE_COLUMN,
                Relationship.RELATIONSHIP_DESCRIPTION_COLUMN,
                Relationship.RELATIONSHIP_TYPE_HASH_COLUMN,
            ]
        ]
        .rename(
            columns={
                Relationship.RELATIONSHIP_TYPE_COLUMN: RelationshipType.NAME_COLUMN,
                Relationship.RELATIONSHIP_DESCRIPTION_COLUMN: RelationshipType.DESCRIPTION_COLUMN,
                Relationship.RELATIONSHIP_TYPE_HASH_COLUMN: "hash",
            }
        )
        .drop_duplicates(subset=["hash"])
    )

    # Compute embeddings for unique relationship types
    descriptions = unique_rel_types[RelationshipType.DESCRIPTION_COLUMN].tolist()
    embeddings = await compute_embeddings(descriptions, embedding_provider)

    # Create the DataFrame for relationship types
    extracted_rel_types_df = pd.DataFrame(
        {
            RelationshipType.NAME_COLUMN: unique_rel_types[
                RelationshipType.NAME_COLUMN
            ],
            RelationshipType.DESCRIPTION_COLUMN: unique_rel_types[
                RelationshipType.DESCRIPTION_COLUMN
            ],
            RelationshipType.EMBEDDING_COLUMN: embeddings,
            RelationshipType.POSTGRES_REFERENCE_COLUMN: None,  # Initially empty
            "hash": unique_rel_types["hash"],
        }
    )

    # Create hash-to-reference mapping
    hash_to_reference = {}

    # If relationship type resolver is provided, perform resolution
    if rel_type_resolver:
        # Convert DataFrame to list of RelationshipType objects for the resolver
        rel_types = []
        for _, row in extracted_rel_types_df.iterrows():
            rel_type = RelationshipType(
                name=row[RelationshipType.NAME_COLUMN],
                description=row[RelationshipType.DESCRIPTION_COLUMN],
                embedding=row[RelationshipType.EMBEDDING_COLUMN],
            )
            rel_types.append(rel_type)

        # Resolve relationship types
        resolved_rel_types = await rel_type_resolver.resolve(rel_types)

        # Update DataFrame with resolved relationship types
        for i, resolved_rel_type in enumerate(resolved_rel_types):
            extracted_rel_types_df.at[i, RelationshipType.NAME_COLUMN] = (
                resolved_rel_type.name
            )
            extracted_rel_types_df.at[i, RelationshipType.DESCRIPTION_COLUMN] = (
                resolved_rel_type.description
            )
            extracted_rel_types_df.at[i, RelationshipType.EMBEDDING_COLUMN] = (
                resolved_rel_type.embedding
            )
            extracted_rel_types_df.at[i, RelationshipType.POSTGRES_REFERENCE_COLUMN] = (
                resolved_rel_type.postgres_reference
            )

            # Update hash-to-reference mapping
            hash_to_reference[extracted_rel_types_df.at[i, "hash"]] = (
                resolved_rel_type.postgres_reference
            )

    return extracted_rel_types_df, hash_to_reference


async def link_relationships_with_references(
    extracted_relationships_df: pd.DataFrame,
    entity_hash_to_reference: Dict[str, int],
    rel_type_hash_to_reference: Dict[str, int],
) -> pd.DataFrame:
    """
    Associate relationships with their corresponding entity and relationship type references.

    Args:
        extracted_relationships_df: DataFrame containing extracted relationships
        entity_hash_to_reference: Mapping from entity hash to database reference
        rel_type_hash_to_reference: Mapping from relationship type hash to database reference

    Returns:
        DataFrame with added reference columns
    """
    result_df = extracted_relationships_df.copy()

    # Add reference columns
    result_df["from_entity_reference"] = result_df[
        Relationship.FROM_ENTITY_HASH_COLUMN
    ].map(entity_hash_to_reference)
    result_df["to_entity_reference"] = result_df[
        Relationship.TO_ENTITY_HASH_COLUMN
    ].map(entity_hash_to_reference)
    result_df["relationship_type_reference"] = result_df[
        Relationship.RELATIONSHIP_TYPE_HASH_COLUMN
    ].map(rel_type_hash_to_reference)

    return result_df


async def save_relationships_to_db(
    relationships_df: pd.DataFrame,
    conn: asyncpg.Connection,
) -> List[int]:
    """
    Save relationships to the database using asyncpg.

    Args:
        relationships_df: DataFrame containing relationships with references
        conn: asyncpg connection

    Returns:
        List of database IDs for the saved relationships
    """
    relationship_ids = []

    # Start a transaction
    async with conn.transaction():
        for _, row in relationships_df.iterrows():
            # Insert relationship into database
            relationship_id = await conn.fetchval(
                """
                INSERT INTO relationships 
                (from_entity, to_entity, relationship_type, constraint_condition, reason, is_causal, source_document_uri)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                row[Relationship.FROM_ENTITY_REFERENCE_COLUMN],
                row[Relationship.TO_ENTITY_REFERENCE_COLUMN],
                row[Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN],
                row[Relationship.CONSTRAINT_CONDITION_COLUMN],
                row[Relationship.REASON_COLUMN],
                row[Relationship.IS_CAUSAL_COLUMN],
                row[Relationship.SOURCE_DOCUMENT_URI_COLUMN],
            )
            relationship_ids.append(relationship_id)

    return relationship_ids
