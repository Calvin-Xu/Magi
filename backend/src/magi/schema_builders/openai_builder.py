"""OpenAI-based schema builder."""

import os
from typing import Any, Dict, List, Optional, Tuple
import json
import csv
import asyncio

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
from .models import (
    RelationalDatasetSchema,
    TableSchema,
    PropertySchema,
)
from .prompts import SCHEMA_EXTRACTION_PROMPT
from magi.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAISchemaBuilder(SchemaBuilder):
    """Extract schema from dataset files using OpenAI, with column chunking."""

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

    async def close(self) -> None:
        """Clean up any async resources."""
        # No async resources to clean up for now
        pass

    async def _read_csv_header_and_rows(
        self,
        file_path: str,
        num_data_rows: int = 2,
    ) -> Tuple[List[str], List[List[str]]]:
        """Read the header row and the first few data rows from a CSV/TSV file.

        Args:
            file_path: Path to the dataset file
            num_data_rows: Number of data rows to read

        Returns:
            A tuple of (header_columns, data_rows)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Determine the delimiter based on file extension
        delimiter = "\t" if file_path.endswith((".tsv", ".tab")) else ","

        header_columns: List[str] = []
        data_rows: List[List[str]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i == 0:
                    header_columns = row
                else:
                    data_rows.append(row)
                if i >= num_data_rows:
                    break

        return header_columns, data_rows

    async def _read_support_documents(
        self, document_paths: List[str], max_chars: Optional[int] = None
    ) -> str:
        """Read support documents and concatenate their contents as text.

        Args:
            document_paths: List of paths to support documents
            max_chars: Maximum number of characters to read per document

        Returns:
            Concatenated document contents
        """
        all_docs = []

        for path in document_paths or []:
            if not os.path.exists(path):
                logger.warning(f"Support document not found: {path}")
                continue
            try:
                md = MarkItDown(enable_plugins=False)
                result = md.convert(path)
                doc_name = os.path.basename(path)
                # Trim content to max_chars if needed
                text_content = (
                    result.text_content[:max_chars]
                    if max_chars
                    else result.text_content
                )
                all_docs.append(f"--- {doc_name} ---\n{text_content}\n")
            except Exception as e:
                logger.error(f"Error reading support document {path}: {e}")

        return "\n\n".join(all_docs)

    def _create_schema_json_format(self) -> Dict[str, str]:
        """
        Minimal placeholder for instructing the LLM to return valid JSON.
        In some OpenAI endpoints, you can pass a function calling or JSON schema.
        For older endpoints, you just parse the raw text.
        """
        return {"type": "json_object"}

    async def _extract_schema_chunk(
        self,
        table_name: str,
        chunk_header: List[str],
        chunk_data: List[List[str]],
        chunk_start_index: int,
        chunk_end_index: int,
        total_columns: int,
        user_prompt: str,
        support_docs_text: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Given a chunk of columns from one table, call the LLM to extract partial schema.
        Returns a dictionary of the form:
        {
            "properties": {
                "col1": {...},
                "col2": {...}
            }
        }
        or None if extraction fails.
        """
        # Format the chunk sample as a small textual representation for the prompt
        # We'll show the chunk header plus the first few rows for that chunk
        chunk_text_lines = []
        # Build a simple table-like string for the chunk
        # Header
        chunk_text_lines.append(",".join(chunk_header))
        # Data
        for row in chunk_data:
            # Make sure we only take the same slice from the row
            chunk_text_lines.append(",".join(row))

        chunk_text = "\n".join(chunk_text_lines)

        # Prepare the final prompt
        prompt = SCHEMA_EXTRACTION_PROMPT.format(
            total_columns=total_columns,
            chunk_start_index=chunk_start_index
            + 1,  # 1-based index in user-friendly format
            chunk_end_index=chunk_end_index,
            table_name=table_name,
            table_chunk_sample=chunk_text,
            user_prompt="USER CONTEXT:\n" + user_prompt if user_prompt else "",
            support_documents=(
                "SUPPORT DOCUMENTS:\n" + support_docs_text if support_docs_text else ""
            ),
        )

        # Call the LLM
        try:
            messages = [{"role": "user", "content": prompt}]

            completion = await call_openai_with_backoff(
                client=self._client,
                model=self.model,
                messages=messages,
                response_format=self._create_schema_json_format(),
                rate_limit=self._rate_limit,
                max_retries=self.max_retries,
            )

            if not completion:
                logger.warning("No valid completion returned for schema chunk.")
                return None

            # Attempt to parse JSON content from the message
            message = completion.choices[0].message
            if not hasattr(message, "content") or not message.content:
                logger.warning("No content in the chunk response message.")
                return None

            logger.debug(f"Chunk response content: {message.content}")

            # Attempt to load JSON
            return json.loads(message.content)

        except Exception as e:
            logger.error(f"Error extracting schema for chunk: {e}")
            return None

    def _merge_partial_schema(
        self, existing_schema: Dict[str, PropertySchema], partial_schema: Dict[str, Any]
    ) -> Dict[str, PropertySchema]:
        """
        Merge a partial schema chunk result into the existing schema dictionary.

        We expect `partial_schema` to be of the form:
        {
          "properties": {
            "col_name": {
              "description": str,
              "type": str,
              "is_primary_key": bool,
              "reference": str|false
            },
            ...
          }
        }
        """
        props = partial_schema.get("properties", {})
        for col_name, col_data in props.items():
            # Convert to PropertySchema (with validation) if possible
            try:
                p_schema = PropertySchema(**col_data)
                existing_schema[col_name] = p_schema
            except Exception as e:
                logger.warning(f"Failed to parse property schema for '{col_name}': {e}")
        return existing_schema

    async def _extract_schema_for_table_in_chunks(
        self,
        table_name: str,
        file_path: str,
        user_prompt: str,
        support_docs_text: str,
        max_chunk_columns: int,
    ) -> Optional[TableSchema]:
        """
        Extract the schema for a single table by chunking its columns.
        """
        # Read header and a couple of data rows
        header, data_rows = await self._read_csv_header_and_rows(file_path)
        total_cols = len(header)

        if total_cols == 0:
            logger.warning(f"No columns found in table {table_name}. Skipping.")
            return None

        # We'll store the final merged schema for this table
        merged_properties: Dict[str, PropertySchema] = {}

        # Prepare tasks for concurrent execution
        chunk_tasks = []
        chunk_ranges = []

        # We iterate in increments of max_chunk_columns over the header
        start = 0
        while start < total_cols:
            end = min(start + max_chunk_columns, total_cols)
            chunk_header = header[start:end]

            # Build chunk_data by slicing the same columns in data_rows
            chunk_data = []
            for row in data_rows:
                # pad row if needed to avoid IndexError
                # because CSV might have fewer columns in short lines
                row_padded = row + ([""] * (len(header) - len(row)))
                chunk_data.append(row_padded[start:end])

            # Create a task for extracting this chunk
            task = self._extract_schema_chunk(
                table_name=table_name,
                chunk_header=chunk_header,
                chunk_data=chunk_data,
                chunk_start_index=start,
                chunk_end_index=end,
                total_columns=total_cols,
                user_prompt=user_prompt,
                support_docs_text=support_docs_text,
            )

            chunk_tasks.append(task)
            chunk_ranges.append((start, end))
            start += max_chunk_columns

        # Execute all chunk tasks concurrently
        if chunk_tasks:
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            # Process results, handling any exceptions
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    start, end = chunk_ranges[i]
                    logger.error(
                        f"Error extracting schema chunk {start}-{end}: {result}"
                    )
                    continue

                if result:
                    # Merge partial schema into final
                    merged_properties = self._merge_partial_schema(
                        merged_properties, result
                    )

        # Build final table schema
        # For demonstration, we use the file name as table description.
        # In reality, you'd want to prompt the LLM for a short table-level description
        # or parse user_prompt for that info. For now, we set a placeholder.
        if not merged_properties:
            return None

        return TableSchema(
            properties=merged_properties,
            description=f"Table '{table_name}' extracted from file '{os.path.basename(file_path)}'.",
        )

    async def extract_schema(
        self,
        dataset_paths: List[str],
        user_prompt: str,
        support_documents: Optional[List[str]] = None,
        max_chunk_columns: int = 50,
    ) -> Optional[RelationalDatasetSchema]:
        """Extract schema from multiple dataset files, chunking columns to avoid context overflow.

        Args:
            dataset_paths: List of paths to the dataset files
            user_prompt: User's prompt or description about the dataset
            support_documents: Optional list of paths to support documents
            max_chunk_columns: Maximum number of columns to provide to the LLM at once

        Returns:
            Extracted relational dataset schema or None if extraction fails
        """
        logger.info(f"Extracting schema from {len(dataset_paths)} dataset files.")
        support_docs_text = await self._read_support_documents(support_documents)

        # Build a combined schema across multiple dataset files
        combined_schema = RelationalDatasetSchema(tables={})
        for file_path in dataset_paths:
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            logger.info(f"Processing table: {table_name}")

            table_schema = await self._extract_schema_for_table_in_chunks(
                table_name=table_name,
                file_path=file_path,
                user_prompt=user_prompt,
                support_docs_text=support_docs_text,
                max_chunk_columns=max_chunk_columns,
            )
            if table_schema:
                combined_schema.tables[table_name] = table_schema

        if not combined_schema.tables:
            logger.warning("No tables extracted from the dataset files.")
            return None

        return combined_schema

    async def create_schema_graph(
        self,
        source_uri: str,
        schema: RelationalDatasetSchema,
        conn: Any,
        embedding_provider: VoyageEmbeddingProvider,
    ) -> Dict[str, int]:
        """Create a schema graph in the database from the extracted schema.

        Args:
            source_uri: A URI or path pointing to the data source
            schema: The extracted relational dataset schema
            conn: Database connection
            embedding_provider: Provider for generating embeddings

        Returns:
            Dictionary with counts of created entities and relationships
        """
        if not schema:
            logger.error("Cannot create schema graph from empty schema.")
            return {}

        try:
            # Step 1: Create entities and relationship types
            entities: Dict[str, Entity] = {}
            # RelationshipTypes
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

            # Step 2: Generate embeddings
            rel_types = [has_property_rel_type, references_rel_type]
            entity_list = list(entities.values())

            # Prepare a single list of all descriptions for efficient batched embedding
            all_descriptions = [entity.description for entity in entity_list] + [
                rt.description for rt in rel_types
            ]
            # Generate embeddings in a single call
            all_embeddings = await embedding_provider.embed(all_descriptions)

            # Assign embeddings back
            entity_count = len(entity_list)
            for i, entity in enumerate(entity_list):
                entity.embedding = all_embeddings[i]
            for i, rt in enumerate(rel_types):
                rt.embedding = all_embeddings[entity_count + i]

            # Step 3: Insert Entities and RelationshipTypes
            entity_ids = {}
            for e in entity_list:
                eid = await insert_entity(conn, e)
                e.postgres_reference = eid
                entity_ids[e.name] = eid

            has_property_id = await insert_relationship_type(
                conn, has_property_rel_type
            )
            has_property_rel_type.postgres_reference = has_property_id

            references_id = await insert_relationship_type(conn, references_rel_type)
            references_rel_type.postgres_reference = references_id

            # Step 4: Create relationships
            relationship_dicts = []
            references_count = 0

            for table_name, table_schema in schema.tables.items():
                table_entity_id = entity_ids[table_name]

                # "has property" relationships
                for prop_name, prop_schema in table_schema.properties.items():
                    property_entity_id = entity_ids[f"{table_name}.{prop_name}"]
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

                    # "references" relationships for foreign keys
                    if prop_schema.reference and prop_schema.reference is not False:
                        ref_table = prop_schema.reference
                        if ref_table in entity_ids:
                            relationship_dicts.append(
                                {
                                    "from_entity_reference": property_entity_id,
                                    "to_entity_reference": entity_ids[ref_table],
                                    "relationship_type_reference": references_id,
                                    "from_imported_schema": True,
                                    "is_causal": False,
                                    "source_uri": source_uri,
                                }
                            )
                            references_count += 1
                        else:
                            logger.warning(
                                f"Referenced table '{ref_table}' not found among extracted entities."
                            )

            # Step 5: Persist relationships
            await save_relationships_to_db(conn, relationship_dicts)

            return {
                "tables_created": len(schema.tables),
                "properties_created": sum(
                    len(tbl.properties) for tbl in schema.tables.values()
                ),
                "entities_created": len(entity_list),
                "has_property_relationships": len(relationship_dicts)
                - references_count,
                "references_relationships": references_count,
                "total_relationships": len(relationship_dicts),
            }

        except Exception as e:
            logger.exception(f"Error creating schema graph: {str(e)}")
            raise
