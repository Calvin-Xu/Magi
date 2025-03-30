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
        "source_document_uri",
        "from_imported_schema",
    ]

    values = [
        relationship.from_entity.postgres_reference,
        relationship.relationship_type.postgres_reference,
        relationship.to_entity.postgres_reference,
        relationship.constraint_condition,
        relationship.reason,
        relationship.is_causal,
        relationship.source_document_uri,
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


# Memgraph operations


async def insert_entity_to_memgraph(
    entity_id: int,
    entity_name: str,
    entity_description: str,
    from_imported_schema: bool = False,
) -> None:
    """
    Insert an entity into Memgraph.

    Args:
        entity_id: Entity ID in PostgreSQL
        entity_name: Entity name
        entity_description: Entity description
        from_imported_schema: Whether this entity is from an imported schema
    """
    from gqlalchemy import Memgraph

    try:
        mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)

        # Escape quotes in strings for Cypher
        entity_name_escaped = entity_name.replace("'", "\\'")
        entity_description_escaped = entity_description.replace("'", "\\'")

        # Create entity node
        query = f"""
        MERGE (e:Entity {{id: {entity_id}, name: '{entity_name_escaped}', 
                         description: '{entity_description_escaped}', 
                         from_imported_schema: {str(from_imported_schema).lower()}}})
        """

        mg.execute(query)
        logger.debug(f"Created/updated entity in Memgraph: {entity_id}, {entity_name}")
    except Exception as e:
        logger.error(f"Error inserting entity {entity_id} into Memgraph: {e}")
        # Continue execution, as this is non-critical


async def insert_relationship_to_memgraph(
    relationship_id: int,
    from_entity_id: int,
    relationship_type_id: int,
    to_entity_id: int,
    relationship_type_name: str,
    constraint_condition: Optional[str] = None,
    reason: Optional[str] = None,
    is_causal: bool = False,
    from_imported_schema: bool = False,
) -> None:
    """
    Insert a relationship into Memgraph.

    Args:
        relationship_id: Relationship ID in PostgreSQL
        from_entity_id: From entity ID
        relationship_type_id: Relationship type ID
        to_entity_id: To entity ID
        relationship_type_name: Relationship type name
        constraint_condition: Constraint condition
        reason: Reason for the relationship
        is_causal: Whether the relationship is causal
        from_imported_schema: Whether this relationship is from an imported schema
    """
    from gqlalchemy import Memgraph

    try:
        mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)

        # Escape quotes in strings for Cypher
        relationship_name_escaped = relationship_type_name.replace("'", "\\'")
        constraint_condition_escaped = (
            constraint_condition.replace("'", "\\'") if constraint_condition else ""
        )
        reason_escaped = reason.replace("'", "\\'") if reason else ""

        # Ensure both entity nodes exist
        query = f"""
        MATCH (from:Entity {{id: {from_entity_id}}}), (to:Entity {{id: {to_entity_id}}})
        MERGE (from)-[r:{relationship_name_escaped} {{
            id: {relationship_id},
            relationship_type_id: {relationship_type_id},
            constraint_condition: '{constraint_condition_escaped}',
            reason: '{reason_escaped}',
            is_causal: {str(is_causal).lower()},
            from_imported_schema: {str(from_imported_schema).lower()}
        }}]->(to)
        """

        mg.execute(query)
        logger.debug(f"Created relationship in Memgraph: {relationship_id}")
    except Exception as e:
        logger.error(
            f"Error inserting relationship {relationship_id} into Memgraph: {e}"
        )
        # Continue execution, as this is non-critical


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
