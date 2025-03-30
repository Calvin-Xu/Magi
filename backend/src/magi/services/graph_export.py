"""Graph export service for Magi knowledge graph.

This module provides functionality to export the knowledge graph from PostgreSQL
to various formats including GraphML.
"""

import io
from typing import Dict, Tuple, List, Any

import asyncpg
import networkx as nx
import pandas as pd

from magi.utils import get_logger

logger = get_logger(__name__)


async def fetch_entities(
    conn: asyncpg.Connection, include_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """Fetch all entities from the database.

    Args:
        conn: PostgreSQL connection
        include_embeddings: Whether to include embedding vectors in the result

    Returns:
        List of entity dictionaries
    """
    query = """
    SELECT 
        id, 
        name, 
        description
        {embedding_field}
    FROM entities
    """

    # Conditionally include the embedding field
    embedding_field = ", embedding" if include_embeddings else ""
    formatted_query = query.format(embedding_field=embedding_field)

    entities = await conn.fetch(formatted_query)

    # Convert row objects to dictionaries
    result = []
    for entity in entities:
        entity_dict = dict(entity)

        # Convert embedding from PostgreSQL vector to Python list if present
        if include_embeddings and entity_dict.get("embedding"):
            # Remove curly braces and split by commas
            embedding_str = entity_dict["embedding"]
            if (
                isinstance(embedding_str, str)
                and embedding_str.startswith("{")
                and embedding_str.endswith("}")
            ):
                embedding_values = embedding_str[1:-1].split(",")
                entity_dict["embedding"] = [float(val) for val in embedding_values]

        result.append(entity_dict)

    return result


async def fetch_relationship_types(
    conn: asyncpg.Connection, include_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """Fetch all relationship types from the database.

    Args:
        conn: PostgreSQL connection
        include_embeddings: Whether to include embedding vectors in the result

    Returns:
        List of relationship type dictionaries
    """
    query = """
    SELECT 
        id, 
        name, 
        description
        {embedding_field}
    FROM relationship_types
    """

    # Conditionally include the embedding field
    embedding_field = ", embedding" if include_embeddings else ""
    formatted_query = query.format(embedding_field=embedding_field)

    rel_types = await conn.fetch(formatted_query)

    # Convert row objects to dictionaries
    result = []
    for rel_type in rel_types:
        rel_type_dict = dict(rel_type)

        # Convert embedding from PostgreSQL vector to Python list if present
        if include_embeddings and rel_type_dict.get("embedding"):
            # Remove curly braces and split by commas
            embedding_str = rel_type_dict["embedding"]
            if (
                isinstance(embedding_str, str)
                and embedding_str.startswith("{")
                and embedding_str.endswith("}")
            ):
                embedding_values = embedding_str[1:-1].split(",")
                rel_type_dict["embedding"] = [float(val) for val in embedding_values]

        result.append(rel_type_dict)

    return result


async def fetch_relationships(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
    """Fetch all relationships with full entity and relationship type information.

    Args:
        conn: PostgreSQL connection

    Returns:
        List of relationship dictionaries
    """
    query = """
    SELECT 
        r.id, 
        r.from_entity, 
        r.to_entity, 
        r.relationship_type,
        r.constraint_condition,
        r.reason,
        r.is_causal,
        r.source_document_uri,
        e_from.name as from_entity_name,
        e_to.name as to_entity_name,
        rt.name as relationship_type_name,
        rt.description as relationship_type_description
    FROM relationships r
    JOIN entities e_from ON r.from_entity = e_from.id
    JOIN entities e_to ON r.to_entity = e_to.id
    JOIN relationship_types rt ON r.relationship_type = rt.id
    """
    relationships = await conn.fetch(query)
    return [dict(rel) for rel in relationships]


async def export_graph_to_graphml(
    conn: asyncpg.Connection, include_embeddings: bool = False
) -> Tuple[str, bytes]:
    """Export the knowledge graph to GraphML format.

    Args:
        conn: PostgreSQL connection
        include_embeddings: Whether to include embedding vectors in the export

    Returns:
        Tuple containing filename and the GraphML content as bytes
    """
    logger.info(
        f"Exporting knowledge graph to GraphML format (include_embeddings={include_embeddings})"
    )

    # Create a directed graph
    G = nx.DiGraph()

    # Fetch data
    entities = await fetch_entities(conn, include_embeddings)
    relationships = await fetch_relationships(conn)

    # Add nodes to the graph
    for entity in entities:
        node_attrs = {
            "label": entity["name"],
            "description": entity["description"] or "",
            "node_type": "entity",
        }

        # Add embedding if available
        if include_embeddings and "embedding" in entity:
            node_attrs["embedding"] = str(entity["embedding"])

        G.add_node(entity["id"], **node_attrs)

    # Add edges to the graph
    for rel in relationships:
        G.add_edge(
            rel["from_entity"],
            rel["to_entity"],
            key=rel["id"],
            label=rel["relationship_type_name"],
            relationship_type=rel["relationship_type"],
            relationship_type_name=rel["relationship_type_name"],
            relationship_type_description=rel["relationship_type_description"] or "",
            constraint_condition=rel["constraint_condition"] or "",
            reason=rel["reason"] or "",
            is_causal=rel["is_causal"] or False,
            source_document_uri=rel["source_document_uri"] or "",
        )

    # Export to GraphML
    output = io.BytesIO()
    nx.write_graphml(G, output)
    output.seek(0)

    # Get timestamp for filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"magi_knowledge_graph_{timestamp}.graphml"

    node_count = len(G.nodes)
    edge_count = len(G.edges)
    logger.info(f"Graph export complete: {node_count} nodes, {edge_count} edges")

    if node_count == 0 and edge_count == 0:
        logger.warning("Exported graph is empty")

    return filename, output.getvalue()


async def export_graph_to_json(
    conn: asyncpg.Connection, include_embeddings: bool = False
) -> Tuple[str, bytes]:
    """Export the knowledge graph to JSON format.

    Args:
        conn: PostgreSQL connection
        include_embeddings: Whether to include embedding vectors in the export

    Returns:
        Tuple containing filename and the JSON content as bytes
    """
    logger.info(
        f"Exporting knowledge graph to JSON format (include_embeddings={include_embeddings})"
    )

    # Fetch data using the helper functions
    entities = await fetch_entities(conn, include_embeddings)
    rel_types = await fetch_relationship_types(conn, include_embeddings)
    relationships = await fetch_relationships(conn)

    # Create a dictionary with all data
    graph_data = {
        "entities": entities,
        "relationship_types": rel_types,
        "relationships": relationships,
    }

    # Convert to JSON
    import json

    json_data = json.dumps(graph_data, indent=2)

    # Get timestamp for filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"magi_knowledge_graph_{timestamp}.json"

    entity_count = len(entities)
    rel_count = len(relationships)
    logger.info(
        f"JSON export complete: {entity_count} entities, {rel_count} relationships"
    )

    if entity_count == 0 and rel_count == 0:
        logger.warning("Exported JSON is empty")

    return filename, json_data.encode("utf-8")


async def export_graph_to_csv(
    conn: asyncpg.Connection, include_embeddings: bool = False
) -> Tuple[str, bytes]:
    """Export the knowledge graph to CSV format.

    Args:
        conn: PostgreSQL connection
        include_embeddings: Whether to include embedding vectors in the export

    Returns:
        Tuple containing filename and the CSV content as bytes (as a ZIP file)
    """
    logger.info(
        f"Exporting knowledge graph to CSV format (include_embeddings={include_embeddings})"
    )

    # Fetch data using the helper functions
    entities = await fetch_entities(conn, include_embeddings)
    rel_types = await fetch_relationship_types(conn, include_embeddings)
    relationships = await fetch_relationships(conn)

    # Convert to pandas DataFrames
    entities_df = pd.DataFrame(entities)
    rel_types_df = pd.DataFrame(rel_types)
    relationships_df = pd.DataFrame(relationships)

    # Create a ZIP file containing the CSVs
    import io
    import zipfile
    from datetime import datetime

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Write entities CSV
        if not entities_df.empty:
            entities_csv = entities_df.to_csv(index=False)
            zip_file.writestr("entities.csv", entities_csv)

        # Write relationship types CSV
        if not rel_types_df.empty:
            rel_types_csv = rel_types_df.to_csv(index=False)
            zip_file.writestr("relationship_types.csv", rel_types_csv)

        # Write relationships CSV
        if not relationships_df.empty:
            relationships_csv = relationships_df.to_csv(index=False)
            zip_file.writestr("relationships.csv", relationships_csv)

    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"magi_knowledge_graph_{timestamp}.zip"

    entity_count = len(entities)
    rel_type_count = len(rel_types)
    rel_count = len(relationships)
    logger.info(
        f"CSV export complete: {entity_count} entities, {rel_type_count} relationship types, {rel_count} relationships"
    )

    if entity_count == 0 and rel_count == 0:
        logger.warning("Exported CSV is empty")

    zip_buffer.seek(0)
    return filename, zip_buffer.getvalue()


async def export_graph(
    conn: asyncpg.Connection, format_type: str, include_embeddings: bool = False
) -> Tuple[str, bytes]:
    """Export the knowledge graph in the specified format.

    Args:
        conn: PostgreSQL connection
        format_type: The export format type ('graphml', 'json', or 'csv')
        include_embeddings: Whether to include embedding vectors in the export

    Returns:
        Tuple containing filename and the file content as bytes
    """
    logger.info(
        f"Exporting graph in {format_type} format (include_embeddings={include_embeddings})"
    )

    if format_type == "graphml":
        return await export_graph_to_graphml(conn, include_embeddings)
    elif format_type == "json":
        return await export_graph_to_json(conn, include_embeddings)
    elif format_type == "csv":
        return await export_graph_to_csv(conn, include_embeddings)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
