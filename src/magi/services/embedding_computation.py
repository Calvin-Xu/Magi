import pandas as pd
from ..embedders.voyage import VoyageEmbeddingProvider
from typing import List


async def create_extracted_entities_df(
    entities: List[dict], embedding_provider: VoyageEmbeddingProvider
) -> pd.DataFrame:
    """Create extracted entities DataFrame and compute embeddings."""
    # Define the DataFrame structure
    extracted_entities_df = pd.DataFrame(
        columns=["name", "description", "embedding", "postgres_reference"]
    )

    # Prepare batches for embedding
    for i in range(0, len(entities), embedding_provider.max_batch_size):
        batch = entities[i : i + embedding_provider.max_batch_size]
        names = [entity["name"] for entity in batch]
        descriptions = [entity["description"] for entity in batch]

        # Compute embeddings
        embeddings = await embedding_provider.embed(
            texts=descriptions,
            truncation=True,
            output_dimension=1024,  # Assuming 1024 is the desired output dimension
            embed_prompt="Use this prompt to embed the descriptions.",
        )

        # Populate the DataFrame
        for name, description, embedding in zip(names, descriptions, embeddings):
            extracted_entities_df = extracted_entities_df.append(
                {
                    "name": name,
                    "description": description,
                    "embedding": embedding,
                    "postgres_reference": None,  # Initially empty
                },
                ignore_index=True,
            )

    return extracted_entities_df


async def create_extracted_rel_types_df(
    relationship_types: List[dict], embedding_provider: VoyageEmbeddingProvider
) -> pd.DataFrame:
    """Create extracted relationship types DataFrame and compute embeddings."""
    # Define the DataFrame structure
    extracted_rel_types_df = pd.DataFrame(
        columns=["name", "description", "embedding", "postgres_reference"]
    )

    # Prepare batches for embedding
    for i in range(0, len(relationship_types), embedding_provider.max_batch_size):
        batch = relationship_types[i : i + embedding_provider.max_batch_size]
        names = [rel_type["name"] for rel_type in batch]
        descriptions = [rel_type["description"] for rel_type in batch]

        # Compute embeddings
        embeddings = await embedding_provider.embed(
            texts=descriptions,
            truncation=True,
            output_dimension=1024,  # Assuming 1024 is the desired output dimension
            embed_prompt="Use this prompt to embed the relationship type descriptions.",
        )

        # Populate the DataFrame
        for name, description, embedding in zip(names, descriptions, embeddings):
            extracted_rel_types_df = extracted_rel_types_df.append(
                {
                    "name": name,
                    "description": description,
                    "embedding": embedding,
                    "postgres_reference": None,  # Initially empty
                },
                ignore_index=True,
            )

    return extracted_rel_types_df
