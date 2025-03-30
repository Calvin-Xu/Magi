#!/usr/bin/env python
"""
Build a knowledge graph from FHIR resource schemas.

This script extracts entities and relationships from FHIR resource schemas
and imports them into the Magi knowledge graph database.
"""

import asyncio
import importlib
import inspect
import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Type

import asyncpg
from pydantic import Field

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fhir.resources import domainresource
from src.magi.config import (
    PostgresConfig,
    VoyageAIConfig,
    MemgraphConfig,
    _load_env_file,
)
from src.magi.embedders.voyage import VoyageEmbeddingProvider
from src.magi.services.db_operations import (
    insert_entity,
    insert_relationship_type,
    save_relationships_to_db,
)
from src.magi.services.models import Entity, RelationshipType
from src.magi.utils import get_logger

# Load environment variables
_load_env_file()

# Configure logging to reduce verbosity
logger = get_logger(__name__)
# Disable overly verbose loggers
logging.getLogger("voyageai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class FHIRProperty:
    """Represents a property of a FHIR resource."""

    name: str
    description: str
    type_name: str
    is_reference: bool = False
    reference_types: List[str] = None
    entity: Entity = None


@dataclass
class FHIRResource:
    """Represents a FHIR resource with its properties."""

    name: str
    description: str
    properties: List[FHIRProperty] = None


async def get_description_from_field(field: Field) -> str:
    """Extract description from a pydantic Field."""
    if not field:
        return ""

    # Check if field has description directly
    if hasattr(field, "description") and field.description:
        return field.description

    # Check if field has description in json_schema_extra
    if hasattr(field, "json_schema_extra") and field.json_schema_extra:
        extra = field.json_schema_extra
        if isinstance(extra, dict) and "description" in extra:
            return extra["description"]

    # Check for title as fallback
    if hasattr(field, "title") and field.title:
        return field.title

    return ""


async def get_reference_types_from_field(field: Field) -> List[str]:
    """Extract reference types from a field that is a ReferenceType."""
    if not field:
        return []

    if hasattr(field, "json_schema_extra") and field.json_schema_extra:
        extra = field.json_schema_extra
        if isinstance(extra, dict) and "enum_reference_types" in extra:
            return extra["enum_reference_types"]

    return []


async def extract_fhir_resource(
    resource_class: Type, resource_entities: Dict[str, Entity]
) -> FHIRResource:
    """Extract a FHIR resource class into a FHIRResource object."""
    resource_name = resource_class.__name__
    doc = inspect.cleandoc(resource_class.__doc__)
    if split := doc.split("\n\n"):
        if len(split) > 1 and split[0].startswith("Disclaimer:"):
            doc = f"{resource_name} (FHIR resource): " + "".join(split[1:]).replace(
                "\n", " "
            )
    resource_description = doc if doc else f"FHIR resource: {resource_name}"

    # Create the resource
    resource = FHIRResource(
        name=resource_name, description=resource_description, properties=[]
    )

    # Get properties from elements_sequence
    if hasattr(resource_class, "elements_sequence") and callable(
        resource_class.elements_sequence
    ):
        elements = resource_class.elements_sequence()

        for prop_name in elements:
            # Skip extension properties
            if prop_name.endswith("__ext"):
                continue

            # Get field metadata using multiple approaches
            prop_description = ""
            field = None

            # Try to get field from annotations
            if hasattr(resource_class, "__annotations__"):
                if prop_name in resource_class.__annotations__:
                    field = getattr(resource_class, prop_name, None)

            # Extract property description using multiple approaches

            # Check model_fields if available (Pydantic v2 style)
            if (
                hasattr(resource_class, "model_fields")
                and prop_name in resource_class.model_fields
            ):
                field_info = resource_class.model_fields[prop_name]
                if hasattr(field_info, "description") and field_info.description:
                    prop_description = field_info.description

            # If we still don't have a description, try to get it from the field's docstring
            if (
                not prop_description
                and field
                and hasattr(field, "__doc__")
                and field.__doc__
            ):
                prop_description = field.__doc__.strip()

            # If still no description, use a generic one
            if not prop_description:
                prop_description = f"FHIR property: {prop_name}"
            else:
                prop_description = f"{prop_name} (FHIR property): " + prop_description

            # Determine property type
            type_name = "unknown"
            is_reference = False
            reference_types = []

            if (
                hasattr(resource_class, "model_fields")
                and prop_name in resource_class.model_fields
            ):
                field_info = resource_class.model_fields[prop_name]
                if (
                    hasattr(field_info, "json_schema_extra")
                    and field_info.json_schema_extra
                ):
                    json_schema_extra = field_info.json_schema_extra
                    if (
                        isinstance(json_schema_extra, dict)
                        and "enum_reference_types" in json_schema_extra
                    ):
                        reference_types = json_schema_extra["enum_reference_types"]
                        is_reference = True
                        type_name = "Reference"

            # Create property entity
            property_entity = Entity(
                name=f"{resource_name}.{prop_name}",
                description=prop_description,
                from_imported_schema=True,
            )

            # Add property to resource
            resource.properties.append(
                FHIRProperty(
                    name=prop_name,
                    description=prop_description,
                    type_name=type_name,
                    is_reference=is_reference,
                    reference_types=reference_types,
                    entity=property_entity,
                )
            )

    return resource


async def discover_fhir_resources() -> List[Type]:
    """Discover all FHIR resource classes in the fhir.resources package."""
    resources = []

    # Get a list of all modules in the fhir.resources package
    import pkgutil
    import fhir.resources

    # Get the path to the fhir.resources package
    package_path = fhir.resources.__path__

    # Iterate through all modules in the package
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.startswith("_"):
            continue

        try:
            # Import the module
            module = importlib.import_module(f"fhir.resources.{module_name}")

            # Find classes that inherit from DomainResource
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and hasattr(domainresource, "DomainResource")
                    and hasattr(obj, "__mro__")
                    and domainresource.DomainResource in obj.__mro__
                    and obj != domainresource.DomainResource
                ):
                    resources.append(obj)
                    logger.debug(f"Found FHIR resource: {obj.__name__}")
        except ImportError as e:
            logger.warning(
                f"Could not import module: fhir.resources.{module_name}: {e}"
            )
        except Exception as e:
            logger.warning(
                f"Error processing module: fhir.resources.{module_name}: {e}"
            )

    return resources


async def compute_embeddings(
    embedding_provider: VoyageEmbeddingProvider, descriptions: List[str]
) -> List[List[float]]:
    """Compute embeddings for a list of descriptions."""
    if not descriptions:
        return []

    try:
        # VoyageEmbeddingProvider.embed already handles batching internally
        embeddings = await embedding_provider.embed(descriptions)
        return embeddings
    except Exception as e:
        logger.warning(f"Error computing embeddings: {e}")
        return [[] for _ in descriptions]


async def create_schema_relationship_types(conn: asyncpg.Connection) -> Dict[str, int]:
    """Create relationship types used for the schema graph."""
    relationship_types = {
        "has FHIR property": "A relationship indicating that a resource has a specific property.",
        "of FHIR resource type": "A relationship indicating that a property is of a specific resource type.",
    }

    relationship_type_ids = {}

    embedding_provider = VoyageEmbeddingProvider()
    descriptions = list(relationship_types.values())
    embeddings = await compute_embeddings(embedding_provider, descriptions)

    for i, (name, description) in enumerate(relationship_types.items()):
        rel_type = RelationshipType(
            name=name,
            description=description,
            embedding=embeddings[i] if i < len(embeddings) else [],
            from_imported_schema=True,
        )

        rel_type_id = await insert_relationship_type(conn, rel_type)
        relationship_type_ids[name] = rel_type_id
        logger.info(f"Created relationship type: {name} (ID: {rel_type_id})")

    return relationship_type_ids


async def insert_entities_batch(
    conn: asyncpg.Connection, entities: List[Entity]
) -> Dict[Entity, int]:
    """
    Insert a batch of entities into the database.

    Args:
        conn: Database connection
        entities: List of entities to insert

    Returns:
        Dictionary mapping each entity to its database ID
    """
    if not entities:
        return {}

    entity_to_id = {}

    try:
        # Use a more sequential approach to avoid transaction conflicts
        # Process in smaller batches to maintain some parallelism
        BATCH_SIZE = 10

        for i in range(0, len(entities), BATCH_SIZE):
            batch = entities[i : i + BATCH_SIZE]

            # Process each entity in the batch sequentially to avoid conflicts
            for entity in batch:
                try:
                    # Use the individual insert function which is more reliable
                    entity_id = await insert_entity(conn, entity)
                    entity_to_id[entity] = entity_id
                except Exception as e:
                    logger.warning(f"Error inserting entity {entity.name}: {e}")

        return entity_to_id
    except Exception as e:
        logger.error(f"Error in batch entity insertion: {e}")

        # Fall back to fully sequential insertion if batching fails
        for entity in entities:
            try:
                entity_id = await insert_entity(conn, entity)
                entity_to_id[entity] = entity_id
            except Exception as e2:
                logger.error(f"Error inserting entity {entity.name}: {e2}")

        return entity_to_id


async def build_fhir_schema_graph(
    postgres_config: PostgresConfig,
    memgraph_config: MemgraphConfig,
    voyage_config: VoyageAIConfig,
):
    """Build a knowledge graph from FHIR resource schemas."""
    try:
        # Connect to PostgreSQL
        logger.info(
            f"Connecting to PostgreSQL at {postgres_config.host}:{postgres_config.port}..."
        )
        conn = await asyncpg.connect(
            host=postgres_config.host,
            port=postgres_config.port,
            user=postgres_config.user,
            password=postgres_config.password,
            database=postgres_config.database,
        )

        # Create relationship types
        logger.info("Creating relationship types...")
        relationship_types = await create_schema_relationship_types(conn)

        # Initialize embedding provider
        embedding_provider = VoyageEmbeddingProvider(voyage_config.api_key)

        # Discover FHIR resource classes
        logger.info("Discovering FHIR resource classes...")
        resource_classes = await discover_fhir_resources()
        logger.info(f"Found {len(resource_classes)} FHIR resource classes")

        # Track all entities to avoid duplicates
        resource_entities = {}  # Maps resource name to Entity
        property_entities = {}  # Maps property name to Entity

        # Track all "has FHIR property" and "of FHIR resource type" relationships
        has_property_relationships = []
        of_type_relationships = []

        # Process FHIR resources
        logger.info("Extracting FHIR resources and preparing batch operations...")
        for resource_class in resource_classes:
            # Extract resource and its properties
            fhir_resource = await extract_fhir_resource(
                resource_class, resource_entities
            )

            # Create resource description
            resource_description = (
                fhir_resource.description
                if fhir_resource.description
                else f"FHIR resource: {fhir_resource.name}"
            )

            # Create resource entity if it doesn't exist
            if fhir_resource.name not in resource_entities:
                resource_entities[fhir_resource.name] = Entity(
                    name=fhir_resource.name,
                    description=resource_description,
                    from_imported_schema=True,
                )

            # Process properties
            for prop in fhir_resource.properties:
                # Store property entity
                property_name = f"{fhir_resource.name}.{prop.name}"
                property_entities[property_name] = prop.entity

                # Create "has FHIR property" relationship
                has_property_relationships.append(
                    {
                        "from_entity": resource_entities[fhir_resource.name],
                        "to_entity": prop.entity,
                        "relationship_type_id": relationship_types["has FHIR property"],
                        "from_imported_schema": True,
                    }
                )

                # Track reference types for "of FHIR resource type" relationships
                if prop.is_reference and prop.reference_types:
                    for ref_type_name in prop.reference_types:
                        # Store reference to create "of FHIR resource type" relationship later
                        of_type_relationships.append(
                            {
                                "from_entity": prop.entity,
                                "to_resource_name": ref_type_name,
                                "relationship_type_id": relationship_types[
                                    "of FHIR resource type"
                                ],
                                "from_imported_schema": True,
                            }
                        )

        # Compute embeddings for resource entities
        logger.info(f"Generating embeddings for {len(resource_entities)} resources...")
        resource_descriptions = [
            entity.description for entity in resource_entities.values()
        ]
        resource_embeddings = await compute_embeddings(
            embedding_provider, resource_descriptions
        )

        # Assign embeddings to resource entities
        for i, entity in enumerate(resource_entities.values()):
            entity.embedding = resource_embeddings[i]

        # Insert resource entities
        logger.info(f"Inserting {len(resource_entities)} resource entities...")
        resource_entity_map = await insert_entities_batch(
            conn, list(resource_entities.values())
        )

        # Update resource entities with their database IDs
        for entity, entity_id in resource_entity_map.items():
            entity.postgres_reference = entity_id

        # Compute embeddings for property entities
        all_property_entities = list(property_entities.values())
        logger.info(
            f"Generating embeddings for {len(all_property_entities)} properties..."
        )

        property_descriptions = [entity.description for entity in all_property_entities]
        property_embeddings = await compute_embeddings(
            embedding_provider, property_descriptions
        )

        # Assign embeddings to property entities
        for i, entity in enumerate(all_property_entities):
            entity.embedding = property_embeddings[i]

        # Insert property entities
        logger.info(f"Inserting {len(all_property_entities)} property entities...")
        property_entity_map = await insert_entities_batch(conn, all_property_entities)

        # Update property entities with their database IDs
        for entity, entity_id in property_entity_map.items():
            entity.postgres_reference = entity_id

        # Prepare "has FHIR property" relationship data
        logger.info(
            f"Creating {len(has_property_relationships)} 'has FHIR property' relationships..."
        )
        has_property_dicts = []
        for rel_data in has_property_relationships:
            has_property_dicts.append(
                {
                    "from_entity_reference": rel_data["from_entity"].postgres_reference,
                    "to_entity_reference": rel_data["to_entity"].postgres_reference,
                    "relationship_type_reference": rel_data["relationship_type_id"],
                    "from_imported_schema": rel_data["from_imported_schema"],
                }
            )

        # Save "has FHIR property" relationships
        await save_relationships_to_db(conn, has_property_dicts)
        logger.info(
            f"Saved {len(has_property_dicts)} 'has FHIR property' relationships"
        )

        # Prepare "of FHIR resource type" relationship data
        logger.info(
            f"Creating {len(of_type_relationships)} 'of FHIR resource type' relationships..."
        )
        of_type_dicts = []
        for rel_data in of_type_relationships:
            # Find the resource entity by name
            if rel_data["to_resource_name"] in resource_entities:
                to_entity = resource_entities[rel_data["to_resource_name"]]

                # Only create relationship if both entities have postgres references
                if (
                    rel_data["from_entity"].postgres_reference
                    and to_entity.postgres_reference
                ):
                    of_type_dicts.append(
                        {
                            "from_entity_reference": rel_data[
                                "from_entity"
                            ].postgres_reference,
                            "to_entity_reference": to_entity.postgres_reference,
                            "relationship_type_reference": rel_data[
                                "relationship_type_id"
                            ],
                            "from_imported_schema": rel_data["from_imported_schema"],
                        }
                    )

        # Save "of FHIR resource type" relationships
        if of_type_dicts:
            await save_relationships_to_db(conn, of_type_dicts)
            logger.info(
                f"Saved {len(of_type_dicts)} 'of FHIR resource type' relationships"
            )
        else:
            logger.warning("No 'of FHIR resource type' relationships were created")

        # Log summary
        logger.info(f"Successfully processed {len(resource_entities)} resources")
        logger.info(f"Created {len(property_entities)} property entities")
        logger.info(
            f"Created {len(has_property_relationships)} 'has FHIR property' relationships"
        )
        logger.info(
            f"Created {len(of_type_dicts)} 'of FHIR resource type' relationships"
        )

    except Exception as e:
        logger.error(f"Error building FHIR schema graph: {e}")
        raise
    finally:
        if "conn" in locals():
            await conn.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from FHIR resource schemas."
    )
    parser.add_argument(
        "--local", action="store_true", help="Run with local database configuration"
    )
    parser.add_argument("--host", type=str, help="PostgreSQL host")
    parser.add_argument("--port", type=int, help="PostgreSQL port")
    parser.add_argument("--db", type=str, help="PostgreSQL database name")
    parser.add_argument("--user", type=str, help="PostgreSQL user")
    parser.add_argument("--password", type=str, help="PostgreSQL password")
    parser.add_argument("--voyage-api-key", type=str, help="Voyage AI API key")
    parser.add_argument("--memgraph-host", type=str, help="Memgraph host")
    parser.add_argument("--memgraph-port", type=int, help="Memgraph port")

    return parser.parse_args()


async def main():
    """Main entry point for the script."""
    args = parse_args()

    # Load default configurations
    default_postgres_config = PostgresConfig()
    default_voyage_config = VoyageAIConfig()
    default_memgraph_config = MemgraphConfig()

    # Override with command line arguments if provided
    postgres_config = PostgresConfig(
        host=args.host or ("localhost" if args.local else default_postgres_config.host),
        port=args.port or default_postgres_config.port,
        database=args.db or default_postgres_config.database,
        user=args.user or default_postgres_config.user,
        password=args.password or default_postgres_config.password,
    )

    voyage_config = VoyageAIConfig(
        api_key=args.voyage_api_key or default_voyage_config.api_key,
    )

    memgraph_config = MemgraphConfig(
        host=args.memgraph_host
        or ("localhost" if args.local else default_memgraph_config.host),
        port=args.memgraph_port or default_memgraph_config.port,
    )

    # Validate required configurations
    if not voyage_config.api_key:
        logger.error(
            "Voyage AI API key is required. Set it using --voyage-api-key or VOYAGE_AI_API_KEY environment variable."
        )
        sys.exit(1)

    try:
        await build_fhir_schema_graph(postgres_config, memgraph_config, voyage_config)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
