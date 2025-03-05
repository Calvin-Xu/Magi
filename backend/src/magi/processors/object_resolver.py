"""Processor for entity and relationship resolution."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import asyncpg
import pandas as pd
from pyspark.sql import DataFrame

from magi.embedders.base import EmbeddingProvider
from magi.resolvers.base import Resolver
from magi.services.models import Entity, Relationship, RelationshipType
from magi.utils import get_logger, log_async_function_call
from .base import DocumentProcessor

# Create a logger for this module
logger = get_logger(__name__)

T = Union[Entity, RelationshipType]


@dataclass
class ObjectResolutionProcessor(DocumentProcessor):
    """
    Processor that resolves entities and relationship types from extracted relationships.

    This processor takes a DataFrame of extracted relationships, resolves the entities and
    relationship types against the database, and returns a DataFrame with database references
    instead of hash columns.
    """

    embedding_provider: EmbeddingProvider
    entity_resolver: Resolver[Entity]
    rel_type_resolver: Resolver[RelationshipType]
    conn: asyncpg.Connection

    @log_async_function_call()
    async def process(self, df: DataFrame) -> DataFrame:
        """
        Process a DataFrame of extracted relationships.

        Args:
            df: DataFrame containing extracted relationships with hash columns

        Returns:
            DataFrame with hash columns replaced by database reference columns
        """
        # Convert Spark DataFrame to Pandas DataFrame for easier processing
        pandas_df = df.toPandas()

        logger.info(f"Processing {len(pandas_df)} relationships")

        try:
            # Process entities and get hash-to-reference mapping
            _, entity_hash_to_reference = await self.create_extracted_entities_df(
                pandas_df
            )
            logger.info(
                f"Processed entities: {len(entity_hash_to_reference)} unique entities found"
            )

            # Process relationship types and get hash-to-reference mapping
            _, rel_type_hash_to_reference = await self.create_extracted_rel_types_df(
                pandas_df
            )
            logger.info(
                f"Processed relationship types: {len(rel_type_hash_to_reference)} unique types found"
            )

            # Associate relationships with entity and relationship type references
            relationships_with_refs = await self.link_relationships_with_references(
                pandas_df,
                entity_hash_to_reference,
                rel_type_hash_to_reference,
            )
            logger.info(
                f"Linked {len(relationships_with_refs)} relationships with references"
            )

            # Save relationships to database
            relationship_ids = await self.save_relationships_to_db(
                relationships_with_refs
            )
            logger.info(f"Saved {len(relationship_ids)} relationships to database")

            # Convert back to Spark DataFrame with the updated relationships_with_refs
            result_df = df.sparkSession.createDataFrame(relationships_with_refs)
            logger.info("Successfully converted back to Spark DataFrame")

            return result_df
        except Exception as e:
            logger.exception(f"Error in ObjectResolutionProcessor.process: {str(e)}")
            # Return the original DataFrame if there's an error
            return df

    @log_async_function_call()
    async def compute_embeddings(self, descriptions: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of descriptions in batches."""
        # Filter out empty descriptions to avoid errors
        valid_descriptions = [desc for desc in descriptions if desc and desc.strip()]

        if not valid_descriptions:
            logger.warning("No valid descriptions to compute embeddings for")
            return [[] for _ in range(len(descriptions))]

        logger.debug(f"Computing embeddings for {len(valid_descriptions)} descriptions")

        try:
            # Compute embeddings for valid descriptions
            embeddings = await self.embedding_provider.embed(valid_descriptions)

            # Create a mapping from original index to embedding
            result = []
            valid_idx = 0

            for desc in descriptions:
                if desc and desc.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # Use an empty list for invalid descriptions
                    result.append([])

            logger.debug(f"Successfully computed {len(embeddings)} embeddings")
            return result
        except Exception as e:
            logger.exception(f"Error computing embeddings: {str(e)}")
            # Return empty embeddings in case of error
            return [[] for _ in range(len(descriptions))]

    @log_async_function_call()
    async def create_extracted_entities_df(
        self, extracted_relationships_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Create a DataFrame of unique entities from extracted relationships.

        Args:
            extracted_relationships_df: DataFrame containing extracted relationships

        Returns:
            Tuple of (DataFrame of unique entities, hash-to-reference mapping)
        """
        # Extract unique entities from relationships
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
                Relationship.FROM_ENTITY_HASH_COLUMN: "hash",
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
                Relationship.TO_ENTITY_HASH_COLUMN: "hash",
            }
        )

        # Combine and drop duplicates
        all_entities = pd.concat([from_entities, to_entities]).drop_duplicates(
            subset=["hash"]
        )

        logger.debug(f"Extracted {len(all_entities)} unique entities")

        # Compute embeddings for all entities
        descriptions = all_entities[Entity.DESCRIPTION_COLUMN].tolist()
        embeddings = await self.compute_embeddings(descriptions)

        logger.debug(f"len(extracted_entities_df): {len(all_entities)}")
        logger.debug(f"len(embeddings): {len(embeddings)}")

        # Create the DataFrame for entities
        extracted_entities_df = pd.DataFrame(
            {
                Entity.NAME_COLUMN: all_entities[Entity.NAME_COLUMN],
                Entity.DESCRIPTION_COLUMN: all_entities[Entity.DESCRIPTION_COLUMN],
                "hash": all_entities["hash"],
            }
        )

        # Add embeddings as a separate step to avoid numpy array conversion issues
        for i, embedding in enumerate(embeddings):
            extracted_entities_df.at[i, Entity.EMBEDDING_COLUMN] = embedding

        # Initialize postgres reference column
        extracted_entities_df[Entity.POSTGRES_REFERENCE_COLUMN] = None

        logger.debug(
            f"Created entities DataFrame with {len(extracted_entities_df)} rows"
        )

        # Convert DataFrame to list of Entity objects for the resolver
        entities = []
        for _, row in extracted_entities_df.iterrows():
            # Make sure embedding is a list, not a numpy array
            embedding = row[Entity.EMBEDDING_COLUMN]
            if embedding is not None and not isinstance(embedding, list):
                embedding = (
                    row[Entity.EMBEDDING_COLUMN].tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )

            entity = Entity(
                name=row[Entity.NAME_COLUMN],
                description=row[Entity.DESCRIPTION_COLUMN],
                embedding=embedding,
            )
            entities.append(entity)

        logger.debug(f"Created {len(entities)} Entity objects")

        # Resolve entities against the database
        resolved_entities = await self.entity_resolver.resolve(entities)

        logger.info(f"Resolved {len(resolved_entities)} entities")

        # Create hash-to-reference mapping
        hash_to_reference = {}
        for i, entity in enumerate(resolved_entities):
            if entity.postgres_reference:
                hash_to_reference[all_entities.iloc[i]["hash"]] = (
                    entity.postgres_reference
                )

        logger.debug(
            f"Created hash-to-reference mapping with {len(hash_to_reference)} entries"
        )

        return extracted_entities_df, hash_to_reference

    @log_async_function_call()
    async def create_extracted_rel_types_df(
        self, extracted_relationships_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Create a DataFrame of unique relationship types from extracted relationships.

        Args:
            extracted_relationships_df: DataFrame containing extracted relationships

        Returns:
            Tuple of (DataFrame of unique relationship types, hash-to-reference mapping)
        """
        # Extract unique relationship types
        unique_rel_types = (
            extracted_relationships_df[
                [
                    Relationship.RELATIONSHIP_TYPE_COLUMN,
                    Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN,
                    Relationship.RELATIONSHIP_TYPE_HASH_COLUMN,
                ]
            ]
            .rename(
                columns={
                    Relationship.RELATIONSHIP_TYPE_COLUMN: RelationshipType.NAME_COLUMN,
                    Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN: RelationshipType.DESCRIPTION_COLUMN,
                    Relationship.RELATIONSHIP_TYPE_HASH_COLUMN: "hash",
                }
            )
            .drop_duplicates(subset=["hash"])
        )

        logger.debug(f"Extracted {len(unique_rel_types)} unique relationship types")

        # Compute embeddings for all relationship types
        descriptions = unique_rel_types[RelationshipType.DESCRIPTION_COLUMN].tolist()
        embeddings = await self.compute_embeddings(descriptions)

        logger.debug(f"Computed {len(embeddings)} embeddings for relationship types")

        # Create the DataFrame for relationship types
        extracted_rel_types_df = pd.DataFrame(
            {
                RelationshipType.NAME_COLUMN: unique_rel_types[
                    RelationshipType.NAME_COLUMN
                ],
                RelationshipType.DESCRIPTION_COLUMN: unique_rel_types[
                    RelationshipType.DESCRIPTION_COLUMN
                ],
                "hash": unique_rel_types["hash"],
            }
        )

        # Add embeddings as a separate step to avoid numpy array conversion issues
        for i, embedding in enumerate(embeddings):
            extracted_rel_types_df.at[i, RelationshipType.EMBEDDING_COLUMN] = embedding

        # Initialize postgres reference column
        extracted_rel_types_df[RelationshipType.POSTGRES_REFERENCE_COLUMN] = None

        logger.debug(
            f"Created relationship types DataFrame with {len(extracted_rel_types_df)} rows"
        )

        # Convert DataFrame to list of RelationshipType objects for the resolver
        rel_types = []
        for _, row in extracted_rel_types_df.iterrows():
            # Make sure embedding is a list, not a numpy array
            embedding = row[RelationshipType.EMBEDDING_COLUMN]
            if embedding is not None and not isinstance(embedding, list):
                embedding = (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )

            rel_type = RelationshipType(
                name=row[RelationshipType.NAME_COLUMN],
                description=row[RelationshipType.DESCRIPTION_COLUMN],
                embedding=embedding,
            )
            rel_types.append(rel_type)

        logger.debug(f"Created {len(rel_types)} RelationshipType objects")

        # Resolve relationship types against the database
        resolved_rel_types = await self.rel_type_resolver.resolve(rel_types)

        logger.info(f"Resolved {len(resolved_rel_types)} relationship types")

        # Create hash-to-reference mapping
        hash_to_reference = {}
        for i, rel_type in enumerate(resolved_rel_types):
            if rel_type.postgres_reference:
                hash_to_reference[unique_rel_types.iloc[i]["hash"]] = (
                    rel_type.postgres_reference
                )

        logger.debug(
            f"Created hash-to-reference mapping with {len(hash_to_reference)} entries"
        )

        return extracted_rel_types_df, hash_to_reference

    @log_async_function_call()
    async def link_relationships_with_references(
        self,
        relationships_df: pd.DataFrame,
        entity_hash_to_reference: Dict[str, str],
        rel_type_hash_to_reference: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Link relationships with entity and relationship type references.

        Args:
            relationships_df: DataFrame of relationships
            entity_hash_to_reference: Mapping from entity hash to database reference
            rel_type_hash_to_reference: Mapping from relationship type hash to database reference

        Returns:
            DataFrame with database references for entities and relationship types
        """
        logger.info(f"Linking {len(relationships_df)} relationships with references")
        logger.debug(
            f"Entity hash to reference map has {len(entity_hash_to_reference)} entries"
        )
        logger.debug(
            f"Relationship type hash to reference map has {len(rel_type_hash_to_reference)} entries"
        )

        # Create a copy of the DataFrame to avoid modifying the original
        result_df = relationships_df.copy()

        # Initialize reference columns with None
        result_df[Relationship.FROM_ENTITY_REFERENCE_COLUMN] = None
        result_df[Relationship.TO_ENTITY_REFERENCE_COLUMN] = None
        result_df[Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN] = None

        # Link entities and relationship types with their database references
        for idx, row in result_df.iterrows():
            from_entity_hash = row[Relationship.FROM_ENTITY_HASH_COLUMN]
            to_entity_hash = row[Relationship.TO_ENTITY_HASH_COLUMN]
            rel_type_hash = row[Relationship.RELATIONSHIP_TYPE_HASH_COLUMN]

            if from_entity_hash in entity_hash_to_reference:
                result_df.at[idx, Relationship.FROM_ENTITY_REFERENCE_COLUMN] = (
                    entity_hash_to_reference[from_entity_hash]
                )
            else:
                logger.warning(
                    f"No reference found for from_entity_hash: {from_entity_hash}"
                )

            if to_entity_hash in entity_hash_to_reference:
                result_df.at[idx, Relationship.TO_ENTITY_REFERENCE_COLUMN] = (
                    entity_hash_to_reference[to_entity_hash]
                )
            else:
                logger.warning(
                    f"No reference found for to_entity_hash: {to_entity_hash}"
                )

            if rel_type_hash in rel_type_hash_to_reference:
                result_df.at[idx, Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN] = (
                    rel_type_hash_to_reference[rel_type_hash]
                )
            else:
                logger.warning(f"No reference found for rel_type_hash: {rel_type_hash}")

        # Count how many relationships have all references
        complete_refs = result_df[
            result_df[Relationship.FROM_ENTITY_REFERENCE_COLUMN].notna()
            & result_df[Relationship.TO_ENTITY_REFERENCE_COLUMN].notna()
            & result_df[Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN].notna()
        ]

        logger.info(
            f"Successfully linked {len(complete_refs)} relationships with all references"
        )

        return result_df

    async def save_relationships_to_db(
        self,
        relationships_df: pd.DataFrame,
    ) -> List[int]:
        """
        Save relationships to the database using asyncpg.

        Args:
            relationships_df: DataFrame containing relationships with references

        Returns:
            List of database IDs for the saved relationships
        """
        relationship_ids = []

        logger.info(f"Saving {len(relationships_df)} relationships to database")

        try:
            async with self.conn.transaction():
                for _, row in relationships_df.iterrows():
                    # Extract fields from the relationship dictionary
                    from_entity_ref = row[Relationship.FROM_ENTITY_REFERENCE_COLUMN]
                    to_entity_ref = row[Relationship.TO_ENTITY_REFERENCE_COLUMN]
                    rel_type_ref = row[Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN]

                    if not all([from_entity_ref, to_entity_ref, rel_type_ref]):
                        logger.warning(
                            f"Skipping relationship with missing references: {row}"
                        )
                        continue

                    # Insert the relationship into the database
                    query = """
                    INSERT INTO relationships 
                    (from_entity, to_entity, relationship_type, constraint_condition, reason, is_causal, source_document_uri)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """

                    logger.debug(
                        f"Executing query with params: {from_entity_ref}, {to_entity_ref}, {rel_type_ref}"
                    )

                    result = await self.conn.fetchval(
                        query,
                        from_entity_ref,
                        to_entity_ref,
                        rel_type_ref,
                        row[Relationship.CONSTRAINT_CONDITION_COLUMN],
                        row[Relationship.REASON_COLUMN],
                        row[Relationship.IS_CAUSAL_COLUMN],
                        row[Relationship.SOURCE_DOCUMENT_URI_COLUMN],
                    )

                    relationship_ids.append(result)
                    logger.info(f"Inserted relationship with ID: {result}")

            logger.info(
                f"Successfully saved {len(relationship_ids)} relationships to database"
            )
            return relationship_ids
        except Exception as e:
            logger.exception(f"Error in save_relationships_to_db: {str(e)}")
            return relationship_ids
