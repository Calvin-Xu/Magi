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
            # Import necessary modules
            import asyncio
            import asyncpg
            from magi.config import POSTGRES_CONFIG

            # Create separate database connections for concurrent operations
            entity_conn = await asyncpg.connect(
                host=POSTGRES_CONFIG.host,
                port=POSTGRES_CONFIG.port,
                user=POSTGRES_CONFIG.user,
                password=POSTGRES_CONFIG.password,
                database=POSTGRES_CONFIG.database,
            )

            rel_type_conn = await asyncpg.connect(
                host=POSTGRES_CONFIG.host,
                port=POSTGRES_CONFIG.port,
                user=POSTGRES_CONFIG.user,
                password=POSTGRES_CONFIG.password,
                database=POSTGRES_CONFIG.database,
            )

            try:
                # Create temporary resolvers with separate connections
                entity_resolver_with_conn = type(self.entity_resolver)(
                    conn=entity_conn,
                    embedding_provider=self.entity_resolver.embedding_provider,
                    table_name=self.entity_resolver.table_name,
                    reference_column=self.entity_resolver.reference_column,
                    similarity_threshold=self.entity_resolver.similarity_threshold,
                    max_tokens_per_batch=self.entity_resolver.max_tokens_per_batch,
                )

                rel_type_resolver_with_conn = type(self.rel_type_resolver)(
                    conn=rel_type_conn,
                    embedding_provider=self.rel_type_resolver.embedding_provider,
                    table_name=self.rel_type_resolver.table_name,
                    reference_column=self.rel_type_resolver.reference_column,
                    similarity_threshold=self.rel_type_resolver.similarity_threshold,
                    max_tokens_per_batch=self.rel_type_resolver.max_tokens_per_batch,
                )

                # Create tasks for parallel execution with separate connections
                entities_task = self._create_extracted_entities_df_with_resolver(
                    pandas_df, entity_resolver_with_conn
                )
                rel_types_task = self._create_extracted_rel_types_df_with_resolver(
                    pandas_df, rel_type_resolver_with_conn
                )

                # Execute both tasks concurrently
                entities_result, rel_types_result = await asyncio.gather(
                    entities_task, rel_types_task
                )

                # Unpack results
                _, entity_hash_to_reference = entities_result
                _, rel_type_hash_to_reference = rel_types_result
            finally:
                # Close the temporary connections
                await entity_conn.close()
                await rel_type_conn.close()

            logger.info(
                f"Processed entities: {len(entity_hash_to_reference)} unique entities found"
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
        return await self._create_extracted_entities_df_with_resolver(
            extracted_relationships_df, self.entity_resolver
        )

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
        return await self._create_extracted_rel_types_df_with_resolver(
            extracted_relationships_df, self.rel_type_resolver
        )

    async def _create_extracted_entities_df_with_resolver(
        self, extracted_relationships_df: pd.DataFrame, resolver: Resolver
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Create a DataFrame of unique entities using the provided resolver.

        This is a wrapper around create_extracted_entities_df that allows using a custom resolver.

        Args:
            extracted_relationships_df: DataFrame containing extracted relationships
            resolver: The resolver to use for entity resolution

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

        # Create a separate Series for embeddings and then assign it to the DataFrame
        embedding_series = pd.Series(embeddings, index=extracted_entities_df.index)
        extracted_entities_df[Entity.EMBEDDING_COLUMN] = embedding_series

        # Initialize postgres reference column
        extracted_entities_df[Entity.POSTGRES_REFERENCE_COLUMN] = None

        logger.debug(
            f"Created entities DataFrame with {len(extracted_entities_df)} rows"
        )

        # Create a dictionary mapping from hash to Entity objects
        hash_to_entity = {}
        for i, row in extracted_entities_df.iterrows():
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
                hash_key=row["hash"],
            )
            hash_to_entity[row["hash"]] = entity

        logger.debug(f"Created dictionary with {len(hash_to_entity)} Entity objects")

        # Resolve entities against the database using the dictionary approach
        # Use the provided resolver instead of self.entity_resolver
        resolved_hash_to_entity = await resolver.resolve(hash_to_entity)

        logger.info(f"Resolved {len(resolved_hash_to_entity)} entities")

        # Update the DataFrame with postgres references from resolved entities
        for i, row in extracted_entities_df.iterrows():
            entity_hash = row["hash"]
            if entity_hash in resolved_hash_to_entity:
                extracted_entities_df.at[i, Entity.POSTGRES_REFERENCE_COLUMN] = (
                    resolved_hash_to_entity[entity_hash].postgres_reference
                )

        # Log how many entities have missing postgres references
        missing_refs = extracted_entities_df[
            extracted_entities_df[Entity.POSTGRES_REFERENCE_COLUMN].isna()
        ]
        if not missing_refs.empty:
            logger.warning(
                f"{len(missing_refs)} entities are missing postgres references after resolution"
            )

        # Create hash-to-reference mapping - only include entities with valid references
        hash_to_reference = {}
        for entity_hash, entity in resolved_hash_to_entity.items():
            if entity.postgres_reference:
                hash_to_reference[entity_hash] = entity.postgres_reference

        logger.debug(
            f"Created hash-to-reference mapping with {len(hash_to_reference)} entries"
        )

        return extracted_entities_df, hash_to_reference

    async def _create_extracted_rel_types_df_with_resolver(
        self, extracted_relationships_df: pd.DataFrame, resolver: Resolver
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Create a DataFrame of unique relationship types using the provided resolver.

        This is a wrapper around create_extracted_rel_types_df that allows using a custom resolver.

        Args:
            extracted_relationships_df: DataFrame containing extracted relationships
            resolver: The resolver to use for relationship type resolution

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

        # Create a separate Series for embeddings and then assign it to the DataFrame
        embedding_series = pd.Series(embeddings, index=extracted_rel_types_df.index)
        extracted_rel_types_df[RelationshipType.EMBEDDING_COLUMN] = embedding_series

        # Initialize postgres reference column
        extracted_rel_types_df[RelationshipType.POSTGRES_REFERENCE_COLUMN] = None

        logger.debug(
            f"Created relationship types DataFrame with {len(extracted_rel_types_df)} rows"
        )

        # Create a dictionary mapping from hash to RelationshipType objects
        hash_to_rel_type = {}
        for i, row in extracted_rel_types_df.iterrows():
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
                hash_key=row["hash"],
            )
            hash_to_rel_type[row["hash"]] = rel_type

        logger.debug(
            f"Created dictionary with {len(hash_to_rel_type)} RelationshipType objects"
        )

        # Resolve relationship types against the database using the dictionary approach
        # Use the provided resolver instead of self.rel_type_resolver
        resolved_hash_to_rel_type = await resolver.resolve(hash_to_rel_type)

        logger.info(f"Resolved {len(resolved_hash_to_rel_type)} relationship types")

        # Update the DataFrame with postgres references from resolved relationship types
        for i, row in extracted_rel_types_df.iterrows():
            rel_type_hash = row["hash"]
            if rel_type_hash in resolved_hash_to_rel_type:
                extracted_rel_types_df.at[
                    i, RelationshipType.POSTGRES_REFERENCE_COLUMN
                ] = resolved_hash_to_rel_type[rel_type_hash].postgres_reference

        # Log how many relationship types have missing postgres references
        missing_refs = extracted_rel_types_df[
            extracted_rel_types_df[RelationshipType.POSTGRES_REFERENCE_COLUMN].isna()
        ]
        if not missing_refs.empty:
            logger.warning(
                f"{len(missing_refs)} relationship types are missing postgres references after resolution"
            )

        # Create hash-to-reference mapping - only include relationship types with valid references
        hash_to_reference = {}
        for rel_type_hash, rel_type in resolved_hash_to_rel_type.items():
            if rel_type.postgres_reference:
                hash_to_reference[rel_type_hash] = rel_type.postgres_reference

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
        Save relationships to the PostgreSQL and Memgraph databases.

        Args:
            relationships_df: DataFrame containing relationships with references

        Returns:
            List of database IDs for the saved relationships
        """
        relationship_ids = []

        logger.info(f"Saving {len(relationships_df)} relationships to database")

        try:
            from gqlalchemy import Memgraph
            from magi.config import MEMGRAPH_CONFIG

            mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)

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

                    # Insert the relationship into PostgreSQL
                    query = """
                    INSERT INTO relationships 
                    (from_entity, to_entity, relationship_type, constraint_condition, reason, is_causal, source_document_uri)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """

                    logger.debug(
                        f"Executing query with params: {from_entity_ref}, {to_entity_ref}, {rel_type_ref}"
                    )

                    relationship_id = await self.conn.fetchval(
                        query,
                        from_entity_ref,
                        to_entity_ref,
                        rel_type_ref,
                        row[Relationship.CONSTRAINT_CONDITION_COLUMN],
                        row[Relationship.REASON_COLUMN],
                        row[Relationship.IS_CAUSAL_COLUMN],
                        row[Relationship.SOURCE_DOCUMENT_URI_COLUMN],
                    )

                    relationship_ids.append(relationship_id)
                    logger.info(f"Inserted relationship with ID: {relationship_id}")

                    from_entity_name = row[Relationship.FROM_ENTITY_COLUMN]
                    to_entity_name = row[Relationship.TO_ENTITY_COLUMN]
                    rel_type_name = row[Relationship.RELATIONSHIP_TYPE_COLUMN]

                    # Save to Memgraph
                    try:
                        query_str = f"""
                        MATCH (from_entity:Entity {{pg_id: {from_entity_ref}, name: "{from_entity_name}"}})
                        RETURN count(from_entity) AS count
                        """
                        result = mg.execute_and_fetch(query_str)
                        if next(result)["count"] == 0:
                            query_str = f"""
                            CREATE (e:Entity {{pg_id: {from_entity_ref}, name: "{from_entity_name}"}})
                            """
                            mg.execute(query_str)

                        query_str = f"""
                        MATCH (to_entity:Entity {{pg_id: {to_entity_ref}, name: "{to_entity_name}"}})
                        RETURN count(to_entity) AS count
                        """
                        result = mg.execute_and_fetch(query_str)
                        if next(result)["count"] == 0:
                            query_str = f"""
                            CREATE (e:Entity {{pg_id: {to_entity_ref}, name: "{to_entity_name}"}})
                            """
                            mg.execute(query_str)

                        # Create the relationship
                        # Convert relationship type to a valid Cypher identifier
                        # Replace spaces with underscores and remove special characters
                        import re

                        valid_rel_type = re.sub(r"[^a-zA-Z0-9_]", "_", rel_type_name)
                        valid_rel_type = valid_rel_type.upper()

                        query_str = f"""
                        MATCH (from_entity:Entity {{pg_id: {from_entity_ref}}})
                        MATCH (to_entity:Entity {{pg_id: {to_entity_ref}}})
                        CREATE (from_entity)-[r:{valid_rel_type} {{pg_id: {relationship_id}}}]->(to_entity)
                        RETURN r
                        """
                        mg.execute(query_str)

                        logger.info(
                            f"Inserted relationship into Memgraph: {from_entity_name} -[{valid_rel_type}]-> {to_entity_name}"
                        )
                    except Exception as e:
                        logger.exception(f"Error saving to Memgraph: {str(e)}")

            logger.info(
                f"Successfully saved {len(relationship_ids)} relationships to databases"
            )
            return relationship_ids
        except Exception as e:
            logger.exception(f"Error in save_relationships_to_db: {str(e)}")
            return relationship_ids
