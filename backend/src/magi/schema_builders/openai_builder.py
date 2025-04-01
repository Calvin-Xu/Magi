"""OpenAI-based schema builder."""

import os
from typing import Any, Dict, List, Optional
import json

from markitdown import MarkItDown

from magi.config import OPENAI_CONFIG
from magi.services.db_operations import (
    insert_entity,
    insert_relationship_type,
    save_relationships_to_db,
)
from magi.services.models import Entity, RelationshipType
from magi.services.openai import (
    call_openai_with_backoff,
    create_openai_client,
    get_model_limits,
)
from magi.services.rate_limiter import RateLimit
from magi.embedders.voyage import VoyageEmbeddingProvider

from .base import SchemaBuilder
from .models import RelationalDatasetSchema
from .prompts import SCHEMA_EXTRACTION_PROMPT
from magi.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class OpenAISchemaBuilder(SchemaBuilder):
    """Extract schema from dataset files using OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        max_retries: int = 3,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the schema builder.

        Args:
            model: OpenAI model to use
            max_retries: Maximum number of retries for API calls
            openai_api_key: OpenAI API key (defaults to env var)
        """
        super().__init__(model=model, max_retries=max_retries)

        # Use provided key or fall back to config
        openai_api_key = openai_api_key or OPENAI_CONFIG.api_key
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Create OpenAI client
        self._client = create_openai_client(openai_api_key)

        # Get model-specific limits
        limits = get_model_limits(model)

        # Set up rate limiting
        self._rate_limit = RateLimit(
            name=model,
            rpm=limits.rpm,
            tpm=limits.tpm,
            window_size=60,
            num_shards=10,
            max_concurrent=limits.max_concurrent,
        )

    async def close(self):
        """Clean up any async resources."""
        # No async resources to clean up for now
        pass

    async def _read_dataset_sample(self, file_path: str, num_lines: int = 3) -> str:
        """Read the header and first few lines of a dataset file.

        Args:
            file_path: Path to the dataset file
            num_lines: Number of lines to read (including header)

        Returns:
            String containing the header and sample lines
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Determine the delimiter based on file extension
        delimiter = "\t" if file_path.endswith((".tsv", ".tab")) else ","

        lines = []
        filename = os.path.basename(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Try to use csv reader for proper handling of quotes and escapes
                import csv

                reader = csv.reader(file, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i >= num_lines:
                        break
                    lines.append(delimiter.join(row))
        except Exception as e:
            # Fallback to simple line reading if csv reader fails
            logger.warning(
                f"Error using CSV reader: {e}. Falling back to simple lines."
            )
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    for i, line in enumerate(file):
                        if i >= num_lines:
                            break
                        lines.append(line.strip())
            except Exception as e2:
                logger.error(f"Error reading dataset file {file_path}: {e2}")
                raise

        sample = "\n".join(lines)
        return f"File: {filename}\n{sample}"

    async def _read_dataset_files(
        self, dataset_paths: List[str], num_lines: int = 3
    ) -> str:
        """Read samples from multiple dataset files and concatenate them.

        Args:
            dataset_paths: Paths to the dataset files
            num_lines: Number of lines to read from each file

        Returns:
            String containing samples from all files
        """
        file_samples = []

        for path in dataset_paths:
            try:
                sample = await self._read_dataset_sample(path, num_lines)
                file_samples.append(sample)
            except Exception as e:
                logger.error(f"Error reading dataset file {path}: {e}")

        return "\n\n".join(file_samples)

    async def _read_support_documents(
        self, document_paths: List[str], max_chars: int = 10000
    ) -> str:
        """Read support documents and concatenate their contents.

        Args:
            document_paths: List of paths to support documents
            max_chars: Maximum number of characters to read per document

        Returns:
            Concatenated document contents
        """
        all_docs = []

        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"Support document not found: {path}")
                continue
            try:
                md = MarkItDown(enable_plugins=False)
                result = md.convert(path)
                doc_name = os.path.basename(path)
                all_docs.append(f"--- {doc_name} ---\n{result.text_content}\n")
            except Exception as e:
                logger.error(f"Error reading support document {path}: {e}")

        return "\n\n".join(all_docs)

    def create_schema_json_format(self, model_class):
        """
        Create a proper JSON schema format for OpenAI API from a Pydantic model.

        Args:
            model_class: The Pydantic model class

        Returns:
            Dict containing the proper format for OpenAI's response_format
        """
        return {"type": "json_object"}

    async def extract_schema(
        self,
        dataset_paths: List[str],
        user_prompt: str,
        support_documents: List[str] = None,
    ) -> RelationalDatasetSchema:
        """Extract schema from multiple dataset files.

        Args:
            dataset_paths: List of paths to the dataset files
            user_prompt: User's prompt or description about the dataset
            support_documents: Optional list of paths to support documents

        Returns:
            Extracted relational dataset schema
        """
        logger.info(f"Extracting schema from {len(dataset_paths)} dataset files")

        # Read dataset files and support documents
        dataset_files_text = await self._read_dataset_files(dataset_paths)
        support_docs_text = ""
        if support_documents:
            support_docs_text = await self._read_support_documents(support_documents)

        # Format the extraction prompt
        prompt = (
            SCHEMA_EXTRACTION_PROMPT.replace("{dataset_files}", dataset_files_text)
            .replace("{user_prompt}", user_prompt)
            .replace("{support_documents}", support_docs_text)
        )

        try:
            # Prepare message for OpenAI API
            messages = [{"role": "user", "content": prompt}]

            # Call OpenAI API with backoff and rate limiting
            completion = await call_openai_with_backoff(
                client=self._client,
                model=self.model,
                messages=messages,
                response_format=self.create_schema_json_format(RelationalDatasetSchema),
                rate_limit=self._rate_limit,
                max_retries=self.max_retries,
            )

            # Extract the parsed result
            if not completion:
                logger.warning("No valid response for dataset files")
                return None

            # Get the response message
            message = completion.choices[0].message

            # Check for refusal
            if hasattr(message, "refusal") and message.refusal:
                logger.warning(f"Model refused to answer. Reason: {message.refusal}")
                return None

            # Parse the content as JSON and convert to RelationalDatasetSchema
            if hasattr(message, "content") and message.content:
                try:
                    logger.info(f"Response content: {message.content}")
                    # Parse JSON content into our schema model
                    schema_json = json.loads(message.content)
                    schema = RelationalDatasetSchema.model_validate(schema_json)

                    logger.info(
                        f"Extracted schema with {len(schema.tables)} tables and "
                        f"{sum(len(table.properties) for table in schema.tables.values())} properties"
                    )
                    return schema
                except Exception as e:
                    logger.error(f"Error parsing schema JSON: {e}")
                    return None
            else:
                logger.warning("No content in response message")
                return None

        except Exception as e:
            logger.exception(f"Error extracting schema: {str(e)}")
            return None

    async def create_schema_graph(
        self,
        source_uri: str,
        schema: RelationalDatasetSchema,
        conn: Any,
        embedding_provider: VoyageEmbeddingProvider,
    ) -> Dict[str, int]:
        """Create a schema graph in the database from the extracted schema.

        Args:
            schema: The extracted relational dataset schema
            conn: Database connection
            embedding_provider: Provider for generating embeddings

        Returns:
            Dictionary with counts of created entities and relationships
        """
        if not schema:
            logger.error("Cannot create schema graph from empty schema")
            return {}

        try:
            # Step 1: Create entities and relationship types
            entities: Dict[str, Entity] = {}
            has_property_rel_type = None
            references_rel_type = None

            # Create relationship types if they don't exist yet
            has_property_rel_type = RelationshipType(
                name="has property",
                description="A relationship indicating that a table has a property.",
                from_imported_schema=True,
            )

            references_rel_type = RelationshipType(
                name="references",
                description="A relationship indicating that a property references another table.",
                from_imported_schema=True,
            )

            # Create table entities
            for table_name, table_schema in schema.tables.items():
                # Create entity for the table
                table_entity = Entity(
                    name=table_name,
                    description=table_schema.description,
                    from_imported_schema=True,
                )
                entities[table_name] = table_entity

                # Create entities for each property
                for prop_name, prop_schema in table_schema.properties.items():
                    property_entity = Entity(
                        name=f"{table_name}.{prop_name}",
                        description=prop_schema.description,
                        from_imported_schema=True,
                    )
                    entities[f"{table_name}.{prop_name}"] = property_entity

            # Step 2: Generate embeddings for entities and relationship types
            rel_types = [has_property_rel_type, references_rel_type]
            entity_list = list(entities.values())

            # Prepare a single list of all descriptions for efficient batched embedding
            all_descriptions = [entity.description for entity in entity_list] + [
                rel_type.description for rel_type in rel_types
            ]

            # Generate embeddings for all entities and relationship types in a single call
            all_embeddings = await embedding_provider.embed(all_descriptions)

            # Distribute embeddings back to entities and relationship types
            entity_count = len(entity_list)
            for i, entity in enumerate(entity_list):
                entity.embedding = all_embeddings[i]

            for i, rel_type in enumerate(rel_types):
                rel_type.embedding = all_embeddings[entity_count + i]

            # Step 3: Insert entities and relationship types into the database
            entity_ids = {}
            for entity in entity_list:
                entity_id = await insert_entity(conn, entity)
                entity.postgres_reference = entity_id
                entity_ids[entity.name] = entity_id

            # Insert relationship types
            has_property_id = await insert_relationship_type(
                conn, has_property_rel_type
            )
            has_property_rel_type.postgres_reference = has_property_id

            references_id = await insert_relationship_type(conn, references_rel_type)
            references_rel_type.postgres_reference = references_id

            # Step 4: Create relationships based on schema
            relationship_dicts = []

            # Create "has property" relationships
            for table_name, table_schema in schema.tables.items():
                table_entity_id = entity_ids[table_name]

                for prop_name in table_schema.properties:
                    property_entity_id = entity_ids[f"{table_name}.{prop_name}"]

                    # Add "has property" relationship
                    relationship_dicts.append(
                        {
                            "from_entity_reference": table_entity_id,
                            "to_entity_reference": property_entity_id,
                            "relationship_type_reference": has_property_id,
                            "from_imported_schema": True,
                            "is_causal": False,
                            "source_uri": source_uri,
                        }
                    )

            # Create "references" relationships for foreign keys
            references_count = 0
            for table_name, table_schema in schema.tables.items():
                for prop_name, prop_schema in table_schema.properties.items():
                    # Check if this property references another table
                    if prop_schema.reference is not False:
                        referenced_table = prop_schema.reference

                        # Make sure the referenced table exists in our schema
                        if referenced_table in entity_ids:
                            property_entity_id = entity_ids[f"{table_name}.{prop_name}"]
                            referenced_table_id = entity_ids[referenced_table]

                            # Add "references" relationship
                            relationship_dicts.append(
                                {
                                    "from_entity_reference": property_entity_id,
                                    "to_entity_reference": referenced_table_id,
                                    "relationship_type_reference": references_id,
                                    "from_imported_schema": True,
                                    "is_causal": False,
                                    "source_uri": source_uri,
                                }
                            )
                            references_count += 1
                        else:
                            logger.warning(
                                f"Referenced table {referenced_table} not found in schema"
                            )

            # Step 5: Save relationships to the database
            await save_relationships_to_db(conn, relationship_dicts)

            return {
                "tables_created": len(schema.tables),
                "properties_created": sum(
                    len(table.properties) for table in schema.tables.values()
                ),
                "entities_created": len(entity_list),
                "has_property_relationships": len(relationship_dicts)
                - references_count,
                "references_relationships": references_count,
                "total_relationships": len(relationship_dicts),
                "entities": len(entity_list),
                "rel_types": 2,  # has_property and references
                "relationships": len(relationship_dicts),
            }

        except Exception as e:
            logger.exception(f"Error creating schema graph: {str(e)}")
            raise
