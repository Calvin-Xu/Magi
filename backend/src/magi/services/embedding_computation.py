import pandas as pd
from ..embedders.voyage import VoyageEmbeddingProvider
from typing import List
from ..services.models import Entity, RelationshipType, Relationship


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
            embed_prompt="dummy",
        )
        embeddings.extend(batch_embeddings)
    return embeddings


async def create_extracted_entities_df(
    extracted_relationships_df: pd.DataFrame,
    embedding_provider: VoyageEmbeddingProvider,
) -> pd.DataFrame:
    """Create extracted entities DataFrame from relationships."""
    # Collect all entities (from and to) and their descriptions
    from_entities = extracted_relationships_df[
        [
            Relationship.FROM_ENTITY_COLUMN,
            Relationship.FROM_ENTITY_DESCRIPTION_COLUMN,
            Relationship.FROM_ENTITY_HASH_COLUMN,
        ]
    ].rename(
        columns={
            Relationship.FROM_ENTITY_COLUMN: Entity.NAME_COLUMN,
            Relationship.FROM_ENTITY_DESCRIPTION_COLUMN: Entity.DESCRIPTION_COLUMN,
        }
    )

    to_entities = extracted_relationships_df[
        [
            Relationship.TO_ENTITY_COLUMN,
            Relationship.TO_ENTITY_DESCRIPTION_COLUMN,
            Relationship.TO_ENTITY_HASH_COLUMN,
        ]
    ].rename(
        columns={
            Relationship.TO_ENTITY_COLUMN: Entity.NAME_COLUMN,
            Relationship.TO_ENTITY_DESCRIPTION_COLUMN: Entity.DESCRIPTION_COLUMN,
        }
    )

    # Concatenate both DataFrames
    all_entities = pd.concat([from_entities, to_entities], ignore_index=True)

    # Deduplicate entities based on hashes
    unique_entities = all_entities.drop_duplicates(
        subset=[Entity.NAME_COLUMN, Entity.DESCRIPTION_COLUMN]
    )

    # Compute embeddings for unique entities
    descriptions = unique_entities[Entity.DESCRIPTION_COLUMN].tolist()
    embeddings = await compute_embeddings(descriptions, embedding_provider)

    # Create the DataFrame for entities
    extracted_entities_df = pd.DataFrame(
        {
            Entity.NAME_COLUMN: unique_entities[Entity.NAME_COLUMN],
            Entity.DESCRIPTION_COLUMN: unique_entities[Entity.DESCRIPTION_COLUMN],
            Entity.EMBEDDING_COLUMN: embeddings,
            Entity.POSTGRES_REFERENCE_COLUMN: None,  # Initially empty
            "hash": unique_entities["hash"],
        }
    )

    # TODO: add to postgres, disambiguation

    return extracted_entities_df


async def create_extracted_rel_types_df(
    extracted_relationships_df: pd.DataFrame,
    embedding_provider: VoyageEmbeddingProvider,
) -> pd.DataFrame:
    """Create extracted relationship types DataFrame from relationships."""
    unique_rel_types = extracted_relationships_df[
        [
            RelationshipType.NAME_COLUMN,
            RelationshipType.DESCRIPTION_COLUMN,
            RelationshipType.RELATIONSHIP_TYPE_HASH_COLUMN,
        ]
    ].drop_duplicates()

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
            "hash": unique_rel_types[RelationshipType.RELATIONSHIP_TYPE_HASH_COLUMN],
        }
    )

    # TODO: add to postgres, disambiguation

    return extracted_rel_types_df
