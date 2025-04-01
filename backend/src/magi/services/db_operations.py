"""
Database operations for entities, relationship types, and relationships.

This module provides reusable functions for database operations with PostgreSQL and Memgraph.
"""

from typing import Dict, List, Optional, Any

import asyncpg

from magi.config import MEMGRAPH_CONFIG
from magi.services.models import Entity, Relationship, RelationshipType
from magi.utils import get_logger

logger = get_logger(__name__)

# PostgreSQL operations


async def insert_entity(
    conn: asyncpg.Connection,
    entity: Entity,
) -> int:
    """
    Insert an entity into the PostgreSQL database.

    Args:
        conn: PostgreSQL connection
        entity: Entity to insert

    Returns:
        ID of the inserted entity
    """
    fields = ["name", "description", "embedding", "from_imported_schema"]
    values = [
        entity.name,
        entity.description,
        str(entity.embedding),
        entity.from_imported_schema,
    ]
    placeholders = [f"${i + 1}" for i in range(len(values))]

    # Vector type needs special handling
    vector_index = fields.index("embedding")
    placeholders[vector_index] += "::vector"

    query = f"""
    INSERT INTO entities ({", ".join(fields)})
    VALUES ({", ".join(placeholders)})
    RETURNING id
    """

    try:
        new_id = await conn.fetchval(query, *values)
        return new_id
    except Exception as e:
        logger.error(f"Error inserting entity {entity.name}: {e}")
        raise


async def insert_relationship_type(
    conn: asyncpg.Connection,
    relationship_type: RelationshipType,
) -> int:
    """
    Insert a relationship type into the PostgreSQL database.

    Args:
        conn: PostgreSQL connection
        relationship_type: RelationshipType to insert

    Returns:
        ID of the inserted relationship type
    """
    fields = ["name", "description", "embedding", "from_imported_schema"]
    values = [
        relationship_type.name,
        relationship_type.description,
        str(relationship_type.embedding),
        relationship_type.from_imported_schema,
    ]
    placeholders = [f"${i + 1}" for i in range(len(values))]

    # Vector type needs special handling
    vector_index = fields.index("embedding")
    placeholders[vector_index] += "::vector"

    query = f"""
    INSERT INTO relationship_types ({", ".join(fields)})
    VALUES ({", ".join(placeholders)})
    RETURNING id
    """

    try:
        new_id = await conn.fetchval(query, *values)
        return new_id
    except Exception as e:
        logger.error(f"Error inserting relationship type {relationship_type.name}: {e}")
        raise


async def insert_relationship(
    conn: asyncpg.Connection,
    relationship: Relationship,
) -> int:
    """
    Insert a relationship into the PostgreSQL database.

    Args:
        conn: PostgreSQL connection
        relationship: Relationship to insert

    Returns:
        ID of the inserted relationship
    """
    fields = [
        "from_entity",
        "relationship_type",
        "to_entity",
        "constraint_condition",
        "reason",
        "is_causal",
        "source_uri",
        "from_imported_schema",
    ]

    values = [
        relationship.from_entity.postgres_reference,
        relationship.relationship_type.postgres_reference,
        relationship.to_entity.postgres_reference,
        relationship.constraint_condition,
        relationship.reason,
        relationship.is_causal,
        relationship.source_uri,
        relationship.from_imported_schema,
    ]

    placeholders = [f"${i + 1}" for i in range(len(values))]

    query = f"""
    INSERT INTO relationships ({", ".join(fields)})
    VALUES ({", ".join(placeholders)})
    RETURNING id
    """

    try:
        new_id = await conn.fetchval(query, *values)
        return new_id
    except Exception as e:
        logger.error(f"Error inserting relationship: {e}")
        raise


async def update_entity(
    conn: asyncpg.Connection,
    entity_id: int,
    updates: Dict[str, Any],
) -> None:
    """
    Update an entity in the PostgreSQL database.

    Args:
        conn: PostgreSQL connection
        entity_id: ID of the entity to update
        updates: Dictionary of field names and values to update
    """
    if not updates:
        return

    set_clauses = []
    values = [entity_id]
    param_index = 2

    for field, value in updates.items():
        if field == "embedding" and isinstance(value, list):
            set_clauses.append(f"{field} = ${param_index}::vector")
            values.append(str(value))
        else:
            set_clauses.append(f"{field} = ${param_index}")
            values.append(value)
        param_index += 1

    set_clause = ", ".join(set_clauses)
    query = f"UPDATE entities SET {set_clause} WHERE id=$1"

    try:
        await conn.execute(query, *values)
    except Exception as e:
        logger.error(f"Error updating entity {entity_id}: {e}")
        raise


async def update_relationship_type(
    conn: asyncpg.Connection,
    relationship_type_id: int,
    updates: Dict[str, Any],
) -> None:
    """
    Update a relationship type in the PostgreSQL database.

    Args:
        conn: PostgreSQL connection
        relationship_type_id: ID of the relationship type to update
        updates: Dictionary of field names and values to update
    """
    if not updates:
        return

    set_clauses = []
    values = [relationship_type_id]
    param_index = 2

    for field, value in updates.items():
        if field == "embedding" and isinstance(value, list):
            set_clauses.append(f"{field} = ${param_index}::vector")
            values.append(str(value))
        else:
            set_clauses.append(f"{field} = ${param_index}")
            values.append(value)
        param_index += 1

    set_clause = ", ".join(set_clauses)
    query = f"UPDATE relationship_types SET {set_clause} WHERE id=$1"

    try:
        await conn.execute(query, *values)
    except Exception as e:
        logger.error(f"Error updating relationship type {relationship_type_id}: {e}")
        raise


async def find_entity_by_name(
    conn: asyncpg.Connection,
    name: str,
) -> Optional[Dict[str, Any]]:
    """
    Find an entity by name.

    Args:
        conn: PostgreSQL connection
        name: Entity name to search for

    Returns:
        Entity record as a dictionary, or None if not found
    """
    query = "SELECT * FROM entities WHERE name=$1 LIMIT 1"

    try:
        row = await conn.fetchrow(query, name)
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Error finding entity {name}: {e}")
        raise


async def find_relationship_type_by_name(
    conn: asyncpg.Connection,
    name: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a relationship type by name.

    Args:
        conn: PostgreSQL connection
        name: Relationship type name to search for

    Returns:
        Relationship type record as a dictionary, or None if not found
    """
    query = "SELECT * FROM relationship_types WHERE name=$1 LIMIT 1"

    try:
        row = await conn.fetchrow(query, name)
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Error finding relationship type {name}: {e}")
        raise


async def find_similar_by_embeddings_batch(
    conn: asyncpg.Connection,
    table_name: str,
    query_embeddings: List[List[float]],
    threshold: float,
    limit_per_query: int = 1,
    batch_size: int = 50,
) -> List[List[Dict[str, Any]]]:
    """
    Find similar embeddings for a batch of query embeddings.

    Args:
        conn: Database connection
        table_name: Table to search in
        query_embeddings: List of embedding vectors to search for
        threshold: Similarity threshold (0-1)
        limit_per_query: Maximum number of results per query embedding
        batch_size: Number of embeddings to process in a single database query

    Returns:
        List of lists of matching records
    """
    if not query_embeddings:
        return []

    results = [[] for _ in range(len(query_embeddings))]

    # Process in batches to avoid overwhelming the database
    for batch_start in range(0, len(query_embeddings), batch_size):
        batch_end = min(batch_start + batch_size, len(query_embeddings))
        batch = query_embeddings[batch_start:batch_end]

        # Create a single query that uses unnest() to process multiple embeddings
        query = f"""
        WITH numbered_embeddings AS (
            SELECT 
                unnest($1::int[]) AS query_idx,
                unnest($2::vector[]) AS query_embedding
        )
        SELECT 
            t.id,
            t.name,
            t.description,
            1 - (t.embedding <=> ne.query_embedding) AS similarity,
            ne.query_idx
        FROM 
            {table_name} t,
            numbered_embeddings ne
        WHERE 
            1 - (t.embedding <=> ne.query_embedding) > $3
        ORDER BY 
            ne.query_idx, similarity DESC
        """

        # Prepare parameters
        query_indices = list(range(batch_start, batch_end))
        embedding_strings = [str(emb) for emb in batch]

        # Execute the query
        rows = await conn.fetch(query, query_indices, embedding_strings, threshold)

        # Process results - group by query_idx and limit to top results per query
        current_idx = None
        count = 0

        for row in rows:
            query_idx = row["query_idx"]

            # If we've moved to a new query index, reset the counter
            if current_idx != query_idx:
                current_idx = query_idx
                count = 0

            # Only add up to limit_per_query results per query
            if count < limit_per_query:
                results[query_idx].append(
                    {
                        "reference_id": row["id"],
                        "name": row["name"],
                        "description": row["description"],
                        "similarity": row["similarity"],
                    }
                )
                count += 1

    return results


# Batch operations


async def save_relationships_to_db(
    conn: asyncpg.Connection,
    relationships_data: List[Dict[str, Any]],
) -> List[int]:
    """
    Save a batch of relationships to both PostgreSQL and Memgraph.

    Args:
        conn: PostgreSQL connection
        relationships_data: List of dictionaries containing relationship data
            Each dict must have:
                - from_entity_reference
                - to_entity_reference
                - relationship_type_reference
            And may optionally have:
                - constraint_condition
                - reason
                - is_causal
                - source_uri

    Returns:
        A list of inserted relationship IDs from PostgreSQL.
    """
    from gqlalchemy import Memgraph

    relationship_ids = []

    mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)

    # Create caches for resolved entity and relationship type names
    entity_name_cache = {}
    rel_type_name_cache = {}

    try:
        async with conn.transaction():
            # First, fetch all required entity and relationship type names in bulk
            entity_refs = set()
            rel_type_refs = set()

            for row_dict in relationships_data:
                from_entity_ref = row_dict.get(
                    Relationship.FROM_ENTITY_REFERENCE_COLUMN
                )
                to_entity_ref = row_dict.get(Relationship.TO_ENTITY_REFERENCE_COLUMN)
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
                entity_rows = await conn.fetch(entity_query, list(entity_refs))
                for er in entity_rows:
                    entity_name_cache[er["id"]] = er["name"]

            # Bulk fetch relationship type names
            if rel_type_refs:
                rel_type_query = """
                SELECT id, name FROM relationship_types WHERE id = ANY($1)
                """
                rel_type_rows = await conn.fetch(rel_type_query, list(rel_type_refs))
                for rtr in rel_type_rows:
                    rel_type_name_cache[rtr["id"]] = rtr["name"]

            logger.info(
                f"Cached {len(entity_name_cache)} entity names and {len(rel_type_name_cache)} relationship type names"
            )

            # Now process each relationship
            for row_dict in relationships_data:
                from_entity_ref = row_dict.get(
                    Relationship.FROM_ENTITY_REFERENCE_COLUMN
                )
                to_entity_ref = row_dict.get(Relationship.TO_ENTITY_REFERENCE_COLUMN)
                rel_type_ref = row_dict.get(
                    Relationship.RELATIONSHIP_TYPE_REFERENCE_COLUMN
                )

                # If references are missing, we skip
                if not all([from_entity_ref, to_entity_ref, rel_type_ref]):
                    logger.warning(
                        f"Skipping relationship due to missing references: {row_dict}"
                    )
                    continue

                # Check if the relationship already exists in the database
                check_query = """
                SELECT id, source_uris FROM relationships
                WHERE from_entity = $1 AND to_entity = $2 AND relationship_type = $3
                """
                existing_relationship = await conn.fetchrow(
                    check_query, from_entity_ref, to_entity_ref, rel_type_ref
                )

                # Get the source URI from the current relationship
                source_uri = row_dict.get(Relationship.SOURCE_URI_COLUMN)
                relationship_id = None

                if existing_relationship:
                    # Relationship exists, check if we need to append the source URI
                    relationship_id = existing_relationship["id"]
                    existing_source_uris = existing_relationship["source_uris"] or []

                    if source_uri and source_uri not in existing_source_uris:
                        # Add the new source URI to the list
                        updated_source_uris = existing_source_uris + [source_uri]

                        # Update the existing relationship
                        update_query = """
                        UPDATE relationships
                        SET source_uris = $1
                        WHERE id = $2
                        """
                        await conn.execute(
                            update_query, updated_source_uris, relationship_id
                        )
                        logger.debug(
                            f"Updated relationship {relationship_id} with new source URI: {source_uri}"
                        )
                    else:
                        logger.debug(
                            f"Source URI already exists or is empty for relationship {relationship_id}"
                        )
                else:
                    # Insert new relationship with source_uri as an array
                    initial_source_uris = [source_uri] if source_uri else []

                    query = """
                    INSERT INTO relationships
                    (from_entity, to_entity, relationship_type, constraint_condition, reason, is_causal, source_uris, from_imported_schema, confidence)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                    """
                    relationship_id = await conn.fetchval(
                        query,
                        from_entity_ref,
                        to_entity_ref,
                        rel_type_ref,
                        row_dict.get(Relationship.CONSTRAINT_CONDITION_COLUMN),
                        row_dict.get(Relationship.REASON_COLUMN),
                        row_dict.get(Relationship.IS_CAUSAL_COLUMN),
                        initial_source_uris,
                        row_dict.get(Relationship.FROM_IMPORTED_SCHEMA_COLUMN),
                        row_dict.get(Relationship.CONFIDENCE_COLUMN),
                    )
                    logger.debug(
                        f"Inserted new relationship with ID {relationship_id} in Postgres"
                    )

                # Only append to our result list if this is a new ID
                if relationship_id and relationship_id not in relationship_ids:
                    relationship_ids.append(relationship_id)

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

                # Insert into Memgraph
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
