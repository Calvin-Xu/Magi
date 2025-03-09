"""Graph export service for Magi knowledge graph.

This module provides functionality to export the knowledge graph from PostgreSQL
to various formats including GraphML.
"""

import io
from typing import Dict, Tuple, List, Any, Optional

import asyncpg
import networkx as nx
import pandas as pd

from magi.utils import get_logger

logger = get_logger(__name__)


async def fetch_entities(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
    """Fetch all entities from the database.

    Args:
        conn: PostgreSQL connection

    Returns:
        List of entity dictionaries
    """
    query = """
    SELECT 
        id, 
        name, 
        description 
    FROM entities
    """
    entities = await conn.fetch(query)
    return [dict(entity) for entity in entities]


async def fetch_relationship_types(conn: asyncpg.Connection) -> List[Dict[str, Any]]:
    """Fetch all relationship types from the database.

    Args:
        conn: PostgreSQL connection

    Returns:
        List of relationship type dictionaries
    """
    query = """
    SELECT 
        id, 
        name, 
        description 
    FROM relationship_types
    """
    rel_types = await conn.fetch(query)
    return [dict(rel_type) for rel_type in rel_types]


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


async def export_graph_to_graphml(conn: asyncpg.Connection) -> Tuple[str, bytes]:
    """Export the knowledge graph to GraphML format.

    Args:
        conn: PostgreSQL connection

    Returns:
        Tuple containing filename and the GraphML content as bytes
    """
    logger.info("Exporting knowledge graph to GraphML format")

    # Create a directed graph
    G = nx.DiGraph()

    # Fetch data
    entities = await fetch_entities(conn)
    relationships = await fetch_relationships(conn)

    # Add nodes to the graph
    for entity in entities:
        G.add_node(
            entity["id"],
            label=entity["name"],
            description=entity["description"] or "",
            node_type="entity",
        )

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


async def export_graph_to_json(conn: asyncpg.Connection) -> Tuple[str, bytes]:
    """Export the knowledge graph to JSON format.

    Args:
        conn: PostgreSQL connection

    Returns:
        Tuple containing filename and the JSON content as bytes
    """
    logger.info("Exporting knowledge graph to JSON format")

    # Fetch data using the helper functions
    entities = await fetch_entities(conn)
    rel_types = await fetch_relationship_types(conn)
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


async def export_graph_to_csv(conn: asyncpg.Connection) -> Tuple[str, bytes]:
    """Export the knowledge graph to CSV format (nodes and edges files zipped).

    Args:
        conn: PostgreSQL connection

    Returns:
        Tuple containing filename and the ZIP content as bytes
    """
    logger.info("Exporting knowledge graph to CSV format")

    # Fetch data
    entities = await fetch_entities(conn)
    rel_types = await fetch_relationship_types(conn)
    relationships = await fetch_relationships(conn)

    # Create DataFrames
    entities_df = pd.DataFrame(entities)
    rel_types_df = pd.DataFrame(rel_types)
    relationships_df = pd.DataFrame(relationships)

    # Create CSV content
    import io
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add entities CSV
        if not entities_df.empty:
            entities_csv = entities_df.to_csv(index=False)
            zip_file.writestr("entities.csv", entities_csv)
        else:
            zip_file.writestr("entities.csv", "id,name,description")

        # Add relationship types CSV
        if not rel_types_df.empty:
            rel_types_csv = rel_types_df.to_csv(index=False)
            zip_file.writestr("relationship_types.csv", rel_types_csv)
        else:
            zip_file.writestr("relationship_types.csv", "id,name,description")

        # Add relationships CSV
        if not relationships_df.empty:
            relationships_csv = relationships_df.to_csv(index=False)
            zip_file.writestr("relationships.csv", relationships_csv)
        else:
            zip_file.writestr(
                "relationships.csv",
                "id,from_entity,to_entity,relationship_type,constraint_condition,reason,is_causal,source_document_uri",
            )

    # Get timestamp for filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"magi_knowledge_graph_{timestamp}.zip"

    entity_count = len(entities)
    rel_count = len(relationships)
    logger.info(
        f"CSV export complete: {entity_count} entities, {rel_count} relationships"
    )

    if entity_count == 0 and rel_count == 0:
        logger.warning("Exported CSV is empty")

    return filename, zip_buffer.getvalue()


async def export_graph(conn: asyncpg.Connection, format_type: str) -> Tuple[str, bytes]:
    """Export the knowledge graph in the specified format.

    Args:
        conn: PostgreSQL connection
        format_type: The export format type ('graphml', 'json', or 'csv')

    Returns:
        Tuple containing filename and the file content as bytes
    """
    export_functions = {
        "graphml": export_graph_to_graphml,
        "json": export_graph_to_json,
        "csv": export_graph_to_csv,
    }

    if format_type not in export_functions:
        raise ValueError(f"Unsupported export format: {format_type}")

    export_function = export_functions[format_type]
    return await export_function(conn)
