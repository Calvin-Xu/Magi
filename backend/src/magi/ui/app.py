import logging
import os
import tempfile
from typing import Optional, Tuple, List, Any

import asyncpg
import gradio as gr
import pandas as pd
from pyspark.sql import SparkSession

from magi.config import POSTGRES_CONFIG
from magi.processors.relationship_extractor import AVAILABLE_MODELS, DEFAULT_MODEL
from magi.schema_builder import OpenAISchemaBuilder
from magi.schema_builder.models import RelationalDatasetSchema
from magi.services.aws import AWSCredentials
from magi.services.checks import run_health_checks
from magi.embedders.voyage import VoyageEmbeddingProvider
from magi.services.graph_export import export_graph as export_graph_func
from magi.services.pipeline import Pipeline
from magi.services.s3 import list_s3_objects
from magi.ui.formatters import all_services_ok, format_status_markdown
from magi.utils import disable_logging, get_logger, set_global_log_level

# Create a logger for this module
logger = get_logger(__name__)

# Constants for display
MAX_ROWS_DISPLAY = 500  # Number of rows to show in UI

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize Spark
    spark: SparkSession = (
        SparkSession.builder.appName("magi").master("local[*]").getOrCreate()
    )

    # Configure logging
    set_global_log_level(DEFAULT_LOG_LEVEL)
    logger.info("Starting Magi UI application")

    with gr.Blocks(title="Magi System Status", theme=gr.themes.Base()) as ui:
        # Service Status Section
        gr.Markdown("# Magi System Status")
        status_title = gr.Markdown("⚪ Checking Services...")
        with gr.Accordion("Service Details", open=False):
            status_md = gr.Markdown("Checking services...")
            refresh_btn = gr.Button("Refresh Status", variant="primary", size="sm")

            # Moved Logging Configuration inside Service Details
            gr.Markdown("### Logging Configuration")
            with gr.Row():
                log_level = gr.Dropdown(
                    label="Log Level",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    value="INFO",
                )
                logging_enabled = gr.Checkbox(
                    label="Enable Logging",
                    value=True,
                )

            with gr.Row():
                apply_log_settings_btn = gr.Button(
                    "Apply Log Settings", variant="primary", size="sm"
                )

        # S3 Browser Section
        gr.Markdown("## S3 Data Source")
        with gr.Row():
            s3_uri = gr.Textbox(
                label="S3 URI",
                placeholder="s3://bucket/prefix/path",
            )
            aws_key_id = gr.Textbox(
                label="AWS Access Key ID",
                placeholder="Optional - will use environment variables if not provided",
            )
            aws_secret = gr.Textbox(
                label="AWS Secret Access Key",
                placeholder="Optional - will use environment variables if not provided",
                type="password",
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL,
            )

        with gr.Row():
            browse_btn = gr.Button("Browse S3", variant="primary")
            process_btn = gr.Button("Ingest", variant="primary")

        output_md = gr.Markdown("")
        output_df = gr.Dataframe()

        # Schema Building Section
        gr.Markdown("## Schema Builder")
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Data Input"):
                    dataset_files = gr.File(
                        label="Upload Dataset Files",
                        file_count="multiple",
                        file_types=None,
                    )
                    support_docs = gr.File(
                        label="Upload Supporting Documents (Optional)",
                        file_count="multiple",
                        file_types=None,
                    )

                    # Build schema button row
                    with gr.Column():
                        schema_model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=["gpt-4o-2024-11-20", "o3-mini-2025-01-31"],
                            value="gpt-4o-2024-11-20",
                        )
                        build_schema_btn = gr.Button("Build Schema", variant="primary")

                with gr.Tab("Schema Visualization"):
                    schema_tables_df = gr.Dataframe(
                        label="Tables", headers=["Table Name", "Description"]
                    )
                    schema_properties_df = gr.Dataframe(
                        label="Properties",
                        headers=[
                            "Table",
                            "Property",
                            "Type",
                            "Description",
                            "Is Primary Key",
                            "References",
                        ],
                    )

                    # Dataset URI input field
                    dataset_uri = gr.Textbox(
                        label="Dataset URI",
                        placeholder="Enter the source URI for this dataset (e.g., https://doi.org/10.13026/07hj-2a80)",
                        value="",
                    )

                    # Save schema to database button
                    save_schema_btn = gr.Button(
                        "Embed and Save Schema", variant="primary"
                    )
                    save_schema_output = gr.Markdown("")

            # Chat-like interface on the right
            with gr.Column(scale=2):
                chat_interface = gr.Chatbot(label="Schema Assistant", height=500)
                with gr.Row():
                    user_prompt = gr.Textbox(
                        label="Describe your dataset",
                        placeholder="E.g., This dataset contains information about movies and actors...",
                        lines=3,
                    )
                    # clear_chat_btn = gr.Button("Clear", variant="secondary")

        # Database Exploration Section
        gr.Markdown("## Database Exploration")
        with gr.Row():
            refresh_data_btn = gr.Button("Refresh Data", variant="primary")

        entities_df = gr.Dataframe(label="Entities Table")  # For entities table
        relationship_types_df = gr.Dataframe(
            label="Relationship Types Table"
        )  # For relationship types table
        relationships_df = gr.Dataframe(
            label="Relationships Table"
        )  # For relationships table

        wipe_btn = gr.Button("Wipe All Data", variant="stop")
        wipe_output = gr.Markdown("")

        gr.Markdown("## Export Graph")
        with gr.Row():
            export_format = gr.Radio(
                label="Export Format",
                choices=["GraphML", "JSON", "CSV"],
                value="GraphML",
                info="Select the format to export the graph",
            )
            include_embeddings = gr.Checkbox(
                label="Include Embeddings",
                value=False,
                info="Include embedding vectors in the export (increases file size)",
            )
            export_graph_btn = gr.Button("Export Graph", variant="primary")

        with gr.Row():
            export_file = gr.File(label="Exported Graph", interactive=True)

        export_graph_output = gr.Markdown("")

        async def browse_s3(
            uri: str,
            key_id: Optional[str],
            secret: Optional[str],
        ) -> str:
            """Browse S3 and return formatted results."""
            if not uri:
                return "Please enter an S3 URI"

            try:
                credentials = (
                    AWSCredentials(key_id, secret) if key_id or secret else None
                )

                objects = []
                async for obj_uri in list_s3_objects(uri, credentials):
                    objects.append(obj_uri)

                if not objects:
                    return f"No objects found at {uri}"

                # Format results as markdown
                result = f"Found {len(objects)} objects:\n\n"
                for obj in objects:
                    result += f"- `{obj}`\n"
                return result

            except Exception as e:
                return f"Error: {str(e)}"

        async def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            conn = await asyncpg.connect(
                host=POSTGRES_CONFIG.host,
                port=POSTGRES_CONFIG.port,
                user=POSTGRES_CONFIG.user,
                password=POSTGRES_CONFIG.password,
                database=POSTGRES_CONFIG.database,
            )

            entities = await conn.fetch(
                "SELECT id, name, description FROM entities LIMIT 10;"
            )
            relationship_types = await conn.fetch(
                "SELECT id, name, description FROM relationship_types LIMIT 10;"
            )
            relationships = await conn.fetch(
                """
                SELECT 
                    r.id, 
                    e_from.name || ' [' || e_from.id || ']' AS from_entity, 
                    rt.name || ' [' || rt.id || ']' AS relationship_type, 
                    e_to.name || ' [' || e_to.id || ']' AS to_entity, 
                    r.reason, 
                    r.source_document_uri
                FROM relationships r
                JOIN entities e_from ON r.from_entity = e_from.id
                JOIN entities e_to ON r.to_entity = e_to.id
                JOIN relationship_types rt ON r.relationship_type = rt.id
                LIMIT 10;
                """
            )
            if not entities or not relationship_types or not relationships:
                return (
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                )
            entities_df = pd.DataFrame(entities)
            entities_df.columns = ["id", "name", "description"]
            relationship_types_df = pd.DataFrame(relationship_types)
            relationship_types_df.columns = ["id", "name", "description"]
            relationships_df = pd.DataFrame(relationships)
            relationships_df.columns = [
                "id",
                "from_entity",
                "relationship_type",
                "to_entity",
                "reason",
                "source_document_uri",
            ]

            await conn.close()

            return (
                entities_df,
                relationship_types_df,
                relationships_df,
            )

        async def refresh_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (
                entities_data,
                relationship_types_data,
                relationships_data,
            ) = await load_data()
            entities_df.value = entities_data  # Update the entities DataFrame
            relationship_types_df.value = (
                relationship_types_data  # Update the relationship types DataFrame
            )
            relationships_df.value = (
                relationships_data  # Update the relationships DataFrame
            )
            # Return the data for the output components
            return entities_data, relationship_types_data, relationships_data

        async def process_files(
            uri: str,
            key_id: Optional[str],
            secret: Optional[str],
            model: str,
        ) -> pd.DataFrame:
            """Process text files and extract relationships."""

            conn = await asyncpg.connect(
                host=POSTGRES_CONFIG.host,
                port=POSTGRES_CONFIG.port,
                user=POSTGRES_CONFIG.user,
                password=POSTGRES_CONFIG.password,
                database=POSTGRES_CONFIG.database,
            )

            if not uri:
                return "Please enter an S3 URI"

            try:
                credentials = (
                    AWSCredentials(key_id, secret) if key_id or secret else None
                )
                pipeline = Pipeline(spark, conn, model=model, credentials=credentials)

                total_documents = 0
                first_df = None

                # Process documents
                async for df in pipeline.process_documents(uri):
                    if first_df is None:
                        first_df = df.limit(MAX_ROWS_DISPLAY)
                    total_documents += df.count()

                if total_documents == 0:
                    return "No text documents found to process"

                if first_df is None:
                    return "No DataFrames were processed"

                # Convert the PySpark DataFrame to a Pandas DataFrame
                result_df = first_df.toPandas()  # Convert to Pandas DataFrame

                # Refresh the data in the DataFrame components
                refreshed_data = (
                    await refresh_data()
                )  # Call the refresh function to get updated data

                # Update the UI components with the refreshed data
                entities_df.value = refreshed_data[0]
                relationship_types_df.value = refreshed_data[1]
                relationships_df.value = refreshed_data[2]

                return result_df

            except Exception as e:
                print(f"Pipeline error: {str(e)}")  # Debug
                import traceback

                print(f"Traceback: {traceback.format_exc()}")  # Debug full traceback
                return f"Error processing files: {str(e)}"

        async def wipe_all_data() -> str:
            """Wipe all data from entities, relationship_types, and relationships tables."""
            conn = await asyncpg.connect(
                host=POSTGRES_CONFIG.host,
                port=POSTGRES_CONFIG.port,
                user=POSTGRES_CONFIG.user,
                password=POSTGRES_CONFIG.password,
                database=POSTGRES_CONFIG.database,
            )

            await conn.execute(
                "TRUNCATE TABLE relationships, relationship_types, entities CASCADE;"
            )
            await conn.close()

            from gqlalchemy import Memgraph

            from magi.config import MEMGRAPH_CONFIG

            mg = Memgraph(host=MEMGRAPH_CONFIG.host, port=MEMGRAPH_CONFIG.port)
            mg.execute("MATCH (n) DETACH DELETE n;")

            return "All data wiped from tables."

        async def wipe_data() -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            result = await wipe_all_data()
            # Get refreshed data after wiping
            refreshed_data = await refresh_data()
            return result, *refreshed_data

        async def apply_log_settings(log_level: str, logging_enabled: bool) -> None:
            if logging_enabled:
                set_global_log_level(log_level)
                logger.info(f"Logging level set to {log_level}")
            else:
                disable_logging()
                logger.info("Logging disabled")

        async def export_graph(
            format_type: str, include_embeddings: bool
        ) -> Tuple[str, str]:
            try:
                # Convert UI format selection to lowercase format type
                format_map = {"GraphML": "graphml", "JSON": "json", "CSV": "csv"}
                export_format = format_map.get(format_type, "graphml")

                conn = await asyncpg.connect(
                    host=POSTGRES_CONFIG.host,
                    port=POSTGRES_CONFIG.port,
                    user=POSTGRES_CONFIG.user,
                    password=POSTGRES_CONFIG.password,
                    database=POSTGRES_CONFIG.database,
                )

                # Export the graph using the selected format
                filename, content = await export_graph_func(
                    conn, export_format, include_embeddings
                )

                await conn.close()

                # Create temporary file for download
                import tempfile
                import os

                # Create file
                file_path = os.path.join(tempfile.gettempdir(), filename)
                with open(file_path, "wb") as f:
                    f.write(content)

                # Return file for download
                return (
                    f"Graph exported successfully as {format_type}. Click to download.",
                    file_path,
                )
            except Exception as e:
                logger.exception(f"Error exporting graph as {format_type}")
                return f"Error exporting graph: {str(e)}", None

        async def process_uploaded_files(files: List[Any], temp_dir: str) -> List[str]:
            """Save uploaded files to temporary directory and return paths."""
            file_paths = []
            for file in files:
                # Handle different types of file objects from Gradio
                if isinstance(file, str):
                    # If it's already a file path
                    file_paths.append(file)
                else:
                    # For Gradio file objects
                    file_name = file.name if hasattr(file, "name") else "uploaded_file"
                    file_path = os.path.join(temp_dir, file_name)

                    # Get file content
                    if hasattr(file, "read"):
                        # If it has a read method (file-like object)
                        content = file.read()
                    elif hasattr(file, "value"):
                        # If it has a value attribute (newer Gradio versions)
                        content = file.value
                    else:
                        # Assume it's the file content directly
                        content = file

                    # Write to file
                    with open(file_path, "wb") as f:
                        if isinstance(content, str):
                            f.write(content.encode("utf-8"))
                        else:
                            f.write(content)

                    file_paths.append(file_path)

            return file_paths

        async def build_schema(
            dataset_files: List[Any],
            support_docs: List[Any],
            user_prompt_text: str,
            model: str,
        ) -> Tuple[
            RelationalDatasetSchema, pd.DataFrame, pd.DataFrame, List[List[Any]]
        ]:
            """Build schema from uploaded files and user prompt."""
            if not dataset_files:
                return (
                    [],
                    pd.DataFrame(),
                    pd.DataFrame(),
                    [[None, "Please upload dataset files to analyze"]],
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Process uploaded files
                    dataset_paths = await process_uploaded_files(
                        dataset_files, temp_dir
                    )
                    support_paths = (
                        await process_uploaded_files(support_docs, temp_dir)
                        if support_docs
                        else []
                    )

                    # Initialize the schema builder
                    schema_builder = OpenAISchemaBuilder(model)

                    # Update chat with status
                    chat_history = []
                    chat_history.append(
                        [
                            None,
                            f"Analyzing {len(dataset_paths)} dataset files with model: {model}...",
                        ]
                    )

                    # Extract schema
                    schema = await schema_builder.extract_schema(
                        dataset_paths,
                        user_prompt_text,
                        support_paths,
                    )

                    logger.info(f"Extracted schema: {schema}")

                    if not schema:
                        chat_history.append(
                            [None, "Failed to extract schema. Please try again."]
                        )
                        return [], pd.DataFrame(), pd.DataFrame(), chat_history

                    # Convert schema to DataFrames for display
                    tables_data = []
                    for table_name, table in schema.tables.items():
                        tables_data.append([table_name, table.description])

                    properties_data = []
                    for table_name, table in schema.tables.items():
                        for prop_name, prop in table.properties.items():
                            properties_data.append(
                                [
                                    table_name,
                                    prop_name,
                                    prop.type,
                                    prop.description,
                                    "Yes" if prop.is_primary_key else "No",
                                    prop.references if prop.references else "",
                                ]
                            )

                    tables_df = pd.DataFrame(
                        tables_data, columns=["Table Name", "Description"]
                    )
                    properties_df = pd.DataFrame(
                        properties_data,
                        columns=[
                            "Table",
                            "Property",
                            "Type",
                            "Description",
                            "Is Primary Key",
                            "References",
                        ],
                    )

                    # Update chat with success message
                    chat_history.append(
                        [
                            None,
                            f"Successfully extracted schema with {len(schema.tables)} tables and "
                            f"{sum(len(table.properties) for table in schema.tables.values())} properties.",
                        ]
                    )

                    # Store the schema for later use
                    # We'll store it in a global variable or return it and keep track in a state variable
                    return schema, tables_df, properties_df, chat_history

                except Exception as e:
                    logger.exception(f"Error building schema: {str(e)}")
                    return (
                        [],
                        pd.DataFrame(),
                        pd.DataFrame(),
                        [[None, f"Error: {str(e)}"]],
                    )

        async def save_schema_to_db(
            source_document_uri: str, schema: RelationalDatasetSchema
        ) -> str:
            """Save the extracted schema to the database."""
            if not schema:
                return "No schema to save. Please build a schema first."

            try:
                # Connect to the database
                conn = await asyncpg.connect(
                    host=POSTGRES_CONFIG.host,
                    port=POSTGRES_CONFIG.port,
                    user=POSTGRES_CONFIG.user,
                    password=POSTGRES_CONFIG.password,
                    database=POSTGRES_CONFIG.database,
                )

                # Initialize the embedding provider
                embedding_provider = VoyageEmbeddingProvider()

                # Create the schema graph
                schema_builder = OpenAISchemaBuilder()
                result = await schema_builder.create_schema_graph(
                    source_document_uri, schema, conn, embedding_provider
                )

                # Clean up
                await conn.close()
                await schema_builder.close()

                # Format the result
                entities_count = result.get("entities", 0)
                rel_types_count = result.get("rel_types", 0)
                relationships_count = result.get("relationships", 0)

                return (
                    f"✅ Schema saved to database successfully!\n\n"
                    f"Created {entities_count} entities, {rel_types_count} relationship types, "
                    f"and {relationships_count} relationships."
                )

            except Exception as e:
                logger.exception(f"Error saving schema to database: {str(e)}")
                return f"❌ Error saving schema to database: {str(e)}"

        async def clear_chat() -> List[List[Any]]:
            """Clear the chat history."""
            return []

        # Create a session state to store the current schema
        current_schema = gr.State(value=None)

        # Service status update handlers
        async def async_update() -> Tuple[str, str]:
            await run_health_checks()
            ok = all_services_ok()
            status_emoji = "✅" if ok else "⚠️"
            title = (
                f"{status_emoji} All Services OK"
                if ok
                else f"{status_emoji} Some Service(s) Need Attention"
            )
            return title, format_status_markdown()

        browse_btn.click(
            fn=browse_s3,
            inputs=[s3_uri, aws_key_id, aws_secret],
            outputs=[output_md],
        )

        process_btn.click(
            fn=process_files,
            inputs=[s3_uri, aws_key_id, aws_secret, model_dropdown],
            outputs=[output_df],
        )

        wipe_btn.click(
            fn=wipe_data,
            outputs=[wipe_output, entities_df, relationship_types_df, relationships_df],
        )

        apply_log_settings_btn.click(
            fn=apply_log_settings,
            inputs=[log_level, logging_enabled],
            outputs=[],
        )

        # Connect refresh_tables_btn to refresh_data function
        refresh_data_btn.click(
            fn=refresh_data,
            outputs=[entities_df, relationship_types_df, relationships_df],
        )

        refresh_btn.click(
            fn=async_update,
            outputs=[status_title, status_md],
        )

        # Connect UI elements to functions
        build_schema_btn.click(
            fn=build_schema,
            inputs=[dataset_files, support_docs, user_prompt, schema_model_dropdown],
            outputs=[
                current_schema,
                schema_tables_df,
                schema_properties_df,
                chat_interface,
            ],
        )

        save_schema_btn.click(
            fn=save_schema_to_db,
            inputs=[dataset_uri, current_schema],
            outputs=[save_schema_output],
        )

        # clear_chat_btn.click(fn=clear_chat, inputs=[], outputs=[chat_interface])

        # Load data when UI loads
        async def load_initial_data() -> Tuple[
            str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame
        ]:
            status_result = await async_update()
            data_result = await load_data()
            return (*status_result, *data_result)

        ui.load(
            fn=load_initial_data,
            outputs=[
                status_title,
                status_md,
                entities_df,
                relationship_types_df,
                relationships_df,
            ],
        )

        export_graph_btn.click(
            fn=export_graph,
            inputs=[export_format, include_embeddings],
            outputs=[export_graph_output, export_file],
        )

    return ui
