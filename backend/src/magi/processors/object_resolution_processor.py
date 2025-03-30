"""Processor for entity and relationship resolution."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import asyncpg
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from magi.embedders.base import EmbeddingProvider
from magi.resolvers.base import Resolver
from magi.services.models import Entity, Relationship, RelationshipType
from magi.utils import get_logger, log_async_function_call

logger = get_logger(__name__)

T = Union[Entity, RelationshipType]


@dataclass
class ObjectResolutionProcessor:
    """
    Processor that resolves entities and relationship types from extracted relationships.

    This processor takes a Spark DataFrame of extracted relationships, resolves the entities and
    relationship types against the database, and returns a Spark DataFrame with database references
    instead of hash columns.
    """

    embedding_provider: EmbeddingProvider
    entity_resolver: Resolver[Entity]
    rel_type_resolver: Resolver[RelationshipType]
    conn: asyncpg.Connection

    @log_async_function_call()
    async def process(self, df: DataFrame) -> DataFrame:
        """
        Process a Spark DataFrame of extracted relationships.

        Steps:
          1. Extract and resolve unique entities & relationship types (async).
          2. Join references back into the main Spark DataFrame.
          3. Save the relationships to PostgreSQL & Memgraph.
          4. Return the enriched Spark DataFrame.

        Args:
            df: Spark DataFrame of extracted relationships

        Returns:
            Spark DataFrame with references (from_entity_reference, to_entity_reference, relationship_type_reference)
        """
        row_count = df.count()
        logger.info(f"Processing {row_count} relationships")

        # Setup parallel connections for resolution
        import asyncpg

        from magi.config import POSTGRES_CONFIG

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
            # Create temporary resolvers (one per connection)
            entity_resolver_with_conn = type(self.entity_resolver)(
                conn=entity_conn,
                embedding_provider=self.entity_resolver.embedding_provider,
                table_name=self.entity_resolver.table_name,
                reference_column=self.entity_resolver.reference_column,
                similarity_threshold=self.entity_resolver.similarity_threshold,
            )
            rel_type_resolver_with_conn = type(self.rel_type_resolver)(
                conn=rel_type_conn,
                embedding_provider=self.rel_type_resolver.embedding_provider,
                table_name=self.rel_type_resolver.table_name,
                reference_column=self.rel_type_resolver.reference_column,
                similarity_threshold=self.rel_type_resolver.similarity_threshold,
            )

            # Resolve entities and relationship types concurrently
            entities_task = self._create_extracted_entities_df_with_resolver(
                df, entity_resolver_with_conn
            )
            rel_types_task = self._create_extracted_rel_types_df_with_resolver(
                df, rel_type_resolver_with_conn
            )

            entities_result, rel_types_result = await asyncio.gather(
                entities_task, rel_types_task
            )

            # Unpack
            entity_ref_sdf, entity_hash_to_reference = entities_result
            rel_type_ref_sdf, rel_type_hash_to_reference = rel_types_result

            logger.info(
                f"Processed entities: {len(entity_hash_to_reference)} unique resolved"
            )
            logger.info(
                f"Processed relationship types: {len(rel_type_hash_to_reference)} unique resolved"
            )

        finally:
            await entity_conn.close()
            await rel_type_conn.close()

        # Join references back
        relationships_with_refs = await self.link_relationships_with_references(
            df, entity_hash_to_reference, rel_type_hash_to_reference
        )

        # Save relationships to DB
        relationship_ids = await self.save_relationships_to_db(relationships_with_refs)
        logger.info(f"Saved {len(relationship_ids)} relationships to DB")

        return relationships_with_refs

    async def compute_embeddings(self, descriptions: List[str]) -> List[List[float]]:
        """
        Compute embeddings in batches for a list of descriptions.
        Returns empty lists for invalid or empty descriptions.
        """
        valid_descriptions = [desc for desc in descriptions if desc and desc.strip()]
        if not valid_descriptions:
            logger.warning("No valid descriptions to compute embeddings for")
            return [[] for _ in range(len(descriptions))]

        logger.debug(f"Computing embeddings for {len(valid_descriptions)} descriptions")

        try:
            # Compute embeddings for valid descriptions
            embeddings = await self.embedding_provider.embed(valid_descriptions)

            # Map them back to the original indices
            result = []
            valid_idx = 0
            for desc in descriptions:
                if desc and desc.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result.append([])

            logger.debug(f"Computed {len(embeddings)} embeddings successfully")
            return result
        except Exception as e:
            logger.exception(f"Error computing embeddings: {str(e)}")
            return [[] for _ in range(len(descriptions))]

    @log_async_function_call()
    async def create_extracted_entities_df(
        self, extracted_relationships_df: DataFrame
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """
        Helper to create the extracted entities data, using the main entity_resolver.
        """
        return await self._create_extracted_entities_df_with_resolver(
            extracted_relationships_df, self.entity_resolver
        )

    @log_async_function_call()
    async def create_extracted_rel_types_df(
        self, extracted_relationships_df: DataFrame
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """
        Helper to create the extracted relationship types data, using the main rel_type_resolver.
        """
        return await self._create_extracted_rel_types_df_with_resolver(
            extracted_relationships_df, self.rel_type_resolver
        )

    async def _create_extracted_entities_df_with_resolver(
        self, relationships_df: DataFrame, resolver: Resolver
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """
        Identify unique entities from a Spark DF, compute embeddings, resolve them,
        and produce:
         - a small Spark DF mapping [hash -> reference]
         - a Python dict {hash: reference}
        """
        spark = relationships_df.sparkSession

        # Extract from-entities
        from_entities = relationships_df.select(
            F.col(Relationship.FROM_ENTITY_COLUMN).alias(Entity.NAME_COLUMN),
            F.col(Relationship.FROM_ENTITY_DESCRIPTION_COLUMN).alias(
                Entity.DESCRIPTION_COLUMN
            ),
            F.col(Relationship.FROM_ENTITY_HASH_COLUMN).alias("hash"),
        )

        # Extract to-entities
        to_entities = relationships_df.select(
            F.col(Relationship.TO_ENTITY_COLUMN).alias(Entity.NAME_COLUMN),
            F.col(Relationship.TO_ENTITY_DESCRIPTION_COLUMN).alias(
                Entity.DESCRIPTION_COLUMN
            ),
            F.col(Relationship.TO_ENTITY_HASH_COLUMN).alias("hash"),
        )

        # Union unique
        all_entities = from_entities.union(to_entities).drop_duplicates(["hash"])
        unique_count = all_entities.count()
        logger.debug(f"Extracted {unique_count} unique entity records")

        # Collect to the driver for embedding/resolution
        entity_rows = all_entities.collect()
        logger.debug(f"Collected {len(entity_rows)} entity rows locally")

        descriptions = [r[Entity.DESCRIPTION_COLUMN] for r in entity_rows]
        embeddings = await self.compute_embeddings(descriptions)

        # Build hash -> Entity object
        hash_to_entity: Dict[str, Entity] = {}
        for i, row in enumerate(entity_rows):
            entity_hash = row["hash"]
            name_val = row[Entity.NAME_COLUMN]
            desc_val = row[Entity.DESCRIPTION_COLUMN]
            emb_val = embeddings[i] if i < len(embeddings) else []
            hash_to_entity[entity_hash] = Entity(
                name=name_val,
                description=desc_val,
                embedding=emb_val,
                hash_key=entity_hash,
            )

        # Resolve them
        resolved_hash_to_entity = await resolver.resolve(hash_to_entity)
        logger.info(f"Resolved {len(resolved_hash_to_entity)} entities via resolver")

        # Debug: Check which entities have postgres_reference set
        entities_with_refs = sum(
            1
            for e in resolved_hash_to_entity.values()
            if e.postgres_reference is not None
        )
        entities_without_refs = sum(
            1 for e in resolved_hash_to_entity.values() if e.postgres_reference is None
        )
        logger.info(
            f"Entity resolution results: {entities_with_refs} with references, {entities_without_refs} without references"
        )

        # Build a final {hash -> reference}
        hash_to_reference = {}
        for h, e_obj in resolved_hash_to_entity.items():
            if e_obj.postgres_reference:
                hash_to_reference[h] = e_obj.postgres_reference
            else:
                logger.warning(
                    f"Entity missing postgres_reference after resolution: hash={h}, name={e_obj.name}"
                )

        logger.debug(
            f"Created hash->reference with {len(hash_to_reference)} resolved references"
        )

        # Build a Spark DF with [hash, from_entity_reference], etc.
        # We'll unify the column name to a single reference column, e.g. "entity_reference"
        # but in practice we'll rename it when we do broadcast joins.
        ref_schema = StructType(
            [
                StructField("hash", StringType(), True),
                StructField(Entity.POSTGRES_REFERENCE_COLUMN, IntegerType(), True),
            ]
        )
        ref_data = [(h, ref) for h, ref in hash_to_reference.items()]
        entity_ref_sdf = spark.createDataFrame(ref_data, schema=ref_schema)

        return entity_ref_sdf, hash_to_reference

    async def _create_extracted_rel_types_df_with_resolver(
        self, relationships_df: DataFrame, resolver: Resolver
    ) -> Tuple[DataFrame, Dict[str, str]]:
        """
        Identify unique relationship types from a Spark DF, compute embeddings, resolve them,
        and produce:
         - a small Spark DF mapping [hash -> reference]
         - a Python dict {hash: reference}
        """
        spark = relationships_df.sparkSession

        # Extract unique rel types
        unique_rel_types = relationships_df.select(
            F.col(Relationship.RELATIONSHIP_TYPE_COLUMN).alias(
                RelationshipType.NAME_COLUMN
            ),
            F.col(Relationship.RELATIONSHIP_TYPE_DESCRIPTION_COLUMN).alias(
                RelationshipType.DESCRIPTION_COLUMN
            ),
            F.col(Relationship.RELATIONSHIP_TYPE_HASH_COLUMN).alias("hash"),
        ).drop_duplicates(["hash"])

        unique_count = unique_rel_types.count()
        logger.debug(f"Extracted {unique_count} unique relationship types")

        # Collect
        rel_type_rows = unique_rel_types.collect()
        logger.debug(f"Collected {len(rel_type_rows)} rel-type rows locally")

        descriptions = [r[RelationshipType.DESCRIPTION_COLUMN] for r in rel_type_rows]
        embeddings = await self.compute_embeddings(descriptions)

        # Build hash->RelationshipType object
        hash_to_rel_type: Dict[str, RelationshipType] = {}
        for i, row in enumerate(rel_type_rows):
            r_hash = row["hash"]
            name_val = row[RelationshipType.NAME_COLUMN]
            desc_val = row[RelationshipType.DESCRIPTION_COLUMN]
            emb_val = embeddings[i] if i < len(embeddings) else []
            hash_to_rel_type[r_hash] = RelationshipType(
                name=name_val,
                description=desc_val,
                embedding=emb_val,
                hash_key=r_hash,
            )

        # Resolve
        resolved_hash_to_rel_type = await resolver.resolve(hash_to_rel_type)
        logger.info(f"Resolved {len(resolved_hash_to_rel_type)} relationship types")

        # Build final mapping
        hash_to_reference = {}
        for r_hash, rt_obj in resolved_hash_to_rel_type.items():
            if rt_obj.postgres_reference:
                hash_to_reference[r_hash] = rt_obj.postgres_reference

        logger.debug(
            f"Created hash->reference for relationship types with {len(hash_to_reference)} entries"
        )

        # Create a Spark DF
        ref_schema = StructType(
            [
                StructField("hash", StringType(), True),
                StructField(
                    RelationshipType.POSTGRES_REFERENCE_COLUMN, IntegerType(), True
                ),
            ]
        )
        ref_data = [(h, ref) for h, ref in hash_to_reference.items()]
        rel_type_ref_sdf = spark.createDataFrame(ref_data, schema=ref_schema)

        return rel_type_ref_sdf, hash_to_reference

    @log_async_function_call()
    async def link_relationships_with_references(
        self,
        relationships_df: DataFrame,
        entity_hash_to_reference: Dict[str, str],
        rel_type_hash_to_reference: Dict[str, str],
    ) -> DataFrame:
        """
        Use broadcast joins to attach from/to entity references and relationship-type references
        to the main Spark DF. The resulting DF will have columns:

          from_entity_reference
          to_entity_reference
          relationship_type_reference

        without extra "hash" columns.
        """
        spark = relationships_df.sparkSession
        row_count = relationships_df.count()

        logger.info(f"Linking references for {row_count} relationships")
        logger.debug(
            f"Entity hash->reference has {len(entity_hash_to_reference)} entries"
        )
        logger.debug(
            f"RelType hash->reference has {len(rel_type_hash_to_reference)} entries"
        )

        # Build broadcast DF for from-entity
        from_schema = StructType(
            [
                StructField("entity_hash", StringType(), True),
                StructField(
                    Relationship.FROM_ENTITY_REFERENCE_COLUMN, IntegerType(), True
                ),
            ]
        )
        from_data = [(h, ref) for h, ref in entity_hash_to_reference.items()]
        from_ref_df = spark.createDataFrame(from_data, schema=from_schema)

        # Build broadcast DF for to-entity
        to_schema = StructType(
            [
                StructField("entity_hash", StringType(), True),
                StructField(
                    Relationship.TO_ENTITY_REFERENCE_COLUMN, IntegerType(), True
                ),
            ]
        )
        to_data = [(h, ref) for h, ref in entity_hash_to_reference.items()]
        to_ref_df = spark.createDataFrame(to_data, schema=to_schema)

        # Build broadcast DF for relationship-types
        rt_schema = StructType(
            [
                StructField("type_hash", StringType(), True),
                StructField(
                    Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN, IntegerType(), True
                ),
            ]
        )
        rt_data = [(h, ref) for h, ref in rel_type_hash_to_reference.items()]
        rt_ref_df = spark.createDataFrame(rt_data, schema=rt_schema)

        # Left-join from_entity references
        df_with_from_ref = relationships_df.join(
            F.broadcast(from_ref_df),
            on=[F.col(Relationship.FROM_ENTITY_HASH_COLUMN) == F.col("entity_hash")],
            how="left",
        ).drop("entity_hash")

        # Left-join to_entity references
        df_with_to_ref = df_with_from_ref.join(
            F.broadcast(to_ref_df),
            on=[F.col(Relationship.TO_ENTITY_HASH_COLUMN) == F.col("entity_hash")],
            how="left",
        ).drop("entity_hash")

        # Left-join relationship_type references
        df_with_all_refs = df_with_to_ref.join(
            F.broadcast(rt_ref_df),
            on=[
                F.col(Relationship.RELATIONSHIP_TYPE_HASH_COLUMN) == F.col("type_hash")
            ],
            how="left",
        ).drop("type_hash")

        # Count how many have all references
        complete_refs_count = df_with_all_refs.filter(
            F.col(Relationship.FROM_ENTITY_REFERENCE_COLUMN).isNotNull()
            & F.col(Relationship.TO_ENTITY_REFERENCE_COLUMN).isNotNull()
            & F.col(Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN).isNotNull()
        ).count()

        logger.info(
            f"Successfully linked {complete_refs_count} relationships with all references"
        )
        return df_with_all_refs

    async def save_relationships_to_db(
        self,
        relationships_df: DataFrame,
    ) -> List[int]:
        """
        Collect the fully linked Spark DF to the driver and insert relationships into:
        - PostgreSQL
        - Memgraph

        Returns:
            A list of inserted relationship IDs from PostgreSQL.
        """
        from gqlalchemy import Memgraph

        from magi.config import MEMGRAPH_CONFIG

        row_count = relationships_df.count()
        logger.info(f"Saving {row_count} relationships to the database")

        # Collect to driver for insertion
        all_rows = relationships_df.collect()
        relationship_ids = []

        mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)

        # Create caches for resolved entity and relationship type names
        entity_name_cache = {}
        rel_type_name_cache = {}

        try:
            async with self.conn.transaction():
                # First, fetch all required entity and relationship type names in bulk
                entity_refs = set()
                rel_type_refs = set()

                for row in all_rows:
                    row_dict = row.asDict()
                    from_entity_ref = row_dict.get(
                        Relationship.FROM_ENTITY_REFERENCE_COLUMN
                    )
                    to_entity_ref = row_dict.get(
                        Relationship.TO_ENTITY_REFERENCE_COLUMN
                    )
                    rel_type_ref = row_dict.get(
                        Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN
                    )

                    if from_entity_ref:
                        entity_refs.add(from_entity_ref)
                    if to_entity_ref:
                        entity_refs.add(to_entity_ref)
                    if rel_type_ref:
                        rel_type_refs.add(rel_type_ref)

                # Bulk fetch entity names
                if entity_refs:
                    entity_query = """
                    SELECT id, name FROM entities WHERE id = ANY($1)
                    """
                    entity_rows = await self.conn.fetch(entity_query, list(entity_refs))
                    for er in entity_rows:
                        entity_name_cache[er["id"]] = er["name"]

                # Bulk fetch relationship type names
                if rel_type_refs:
                    rel_type_query = """
                    SELECT id, name FROM relationship_types WHERE id = ANY($1)
                    """
                    rel_type_rows = await self.conn.fetch(
                        rel_type_query, list(rel_type_refs)
                    )
                    for rtr in rel_type_rows:
                        rel_type_name_cache[rtr["id"]] = rtr["name"]

                logger.info(
                    f"Cached {len(entity_name_cache)} entity names and {len(rel_type_name_cache)} relationship type names"
                )

                # Now process each relationship
                for row in all_rows:
                    # Convert the Row to a dict so we can do row_dict.get(...)
                    row_dict = row.asDict()

                    from_entity_ref = row_dict.get(
                        Relationship.FROM_ENTITY_REFERENCE_COLUMN
                    )
                    to_entity_ref = row_dict.get(
                        Relationship.TO_ENTITY_REFERENCE_COLUMN
                    )
                    rel_type_ref = row_dict.get(
                        Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN
                    )

                    # If references are missing, we skip
                    if not all([from_entity_ref, to_entity_ref, rel_type_ref]):
                        logger.warning(
                            f"Skipping relationship due to missing references: {row_dict}"
                        )
                        continue

                    # Insert the relationship in PostgreSQL
                    query = """
                    INSERT INTO relationships
                    (from_entity, to_entity, relationship_type, constraint_condition, reason, is_causal, source_document_uri)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """
                    relationship_id = await self.conn.fetchval(
                        query,
                        from_entity_ref,
                        to_entity_ref,
                        rel_type_ref,
                        row_dict.get(Relationship.CONSTRAINT_CONDITION_COLUMN),
                        row_dict.get(Relationship.REASON_COLUMN),
                        row_dict.get(Relationship.IS_CAUSAL_COLUMN),
                        row_dict.get(Relationship.SOURCE_DOCUMENT_URI_COLUMN),
                    )
                    relationship_ids.append(relationship_id)
                    logger.debug(
                        f"Inserted relationship with ID {relationship_id} in Postgres"
                    )

                    # Get resolved entity and relationship type names from cache
                    from_entity_name = entity_name_cache.get(from_entity_ref)
                    to_entity_name = entity_name_cache.get(to_entity_ref)
                    rel_type_name = rel_type_name_cache.get(rel_type_ref)

                    if not all([from_entity_name, to_entity_name, rel_type_name]):
                        logger.warning(
                            f"Skipping Memgraph update due to missing resolved names: "
                            f"from={from_entity_ref}/{from_entity_name}, "
                            f"to={to_entity_ref}/{to_entity_name}, "
                            f"rel_type={rel_type_ref}/{rel_type_name}"
                        )
                        continue

                    # Memgraph: create the nodes and relationship
                    try:
                        # Check or create the "from" entity
                        check_from = f"""
                        MATCH (fe:Entity {{pg_id: {from_entity_ref}, name: "{from_entity_name}"}})
                        RETURN count(fe) AS cnt
                        """
                        result_from = mg.execute_and_fetch(check_from)
                        if next(result_from)["cnt"] == 0:
                            create_from = f"""
                            CREATE (e:Entity {{pg_id: {from_entity_ref}, name: "{from_entity_name}"}})
                            """
                            mg.execute(create_from)

                        # Check or create the "to" entity
                        check_to = f"""
                        MATCH (te:Entity {{pg_id: {to_entity_ref}, name: "{to_entity_name}"}})
                        RETURN count(te) AS cnt
                        """
                        result_to = mg.execute_and_fetch(check_to)
                        if next(result_to)["cnt"] == 0:
                            create_to = f"""
                            CREATE (e:Entity {{pg_id: {to_entity_ref}, name: "{to_entity_name}"}})
                            """
                            mg.execute(create_to)

                        # Create the relationship
                        import re

                        valid_rel_type = re.sub(r"[^a-zA-Z0-9_]", "_", rel_type_name)
                        valid_rel_type = valid_rel_type.upper()

                        create_rel = f"""
                        MATCH (f:Entity {{pg_id: {from_entity_ref}}})
                        MATCH (t:Entity {{pg_id: {to_entity_ref}}})
                        CREATE (f)-[r:{valid_rel_type} {{pg_id: {relationship_id}}}]->(t)
                        RETURN r
                        """
                        mg.execute(create_rel)

                        logger.debug(
                            f"Created Memgraph relationship: "
                            f"{from_entity_name} -[{valid_rel_type}]-> {to_entity_name}"
                        )
                    except Exception as me:
                        logger.exception(f"Error saving to Memgraph: {str(me)}")

            logger.info(
                f"Successfully saved {len(relationship_ids)} total relationships to DB"
            )
        except Exception as e:
            logger.exception(f"Error in save_relationships_to_db: {str(e)}")

        return relationship_ids
