import logging
from typing import Optional, Tuple

import asyncpg
import gradio as gr
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from magi.config import POSTGRES_CONFIG
from magi.services.aws import AWSCredentials
from magi.services.checks import run_health_checks
from magi.services.pipeline import Pipeline
from magi.services.s3 import list_s3_objects
from magi.utils import disable_logging, get_logger, set_global_log_level
from magi.ui.formatters import all_services_ok, format_status_markdown

# Create a logger for this module
logger = get_logger(__name__)

# Constants for display
MAX_ROWS_DISPLAY = 500  # Number of rows to show in UI

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize Spark
    spark = SparkSession.builder.appName("magi").master("local[*]").getOrCreate()

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

        with gr.Row():
            browse_btn = gr.Button("Browse S3", variant="primary")
            process_btn = gr.Button("Ingest", variant="primary")

        output_md = gr.Markdown("")
        output_df = gr.Dataframe()

        # Create DataFrame components for displaying tables
        gr.Markdown("## Database Tables")
        with gr.Row():
            refresh_tables_btn = gr.Button(
                "Refresh Tables", variant="primary", size="sm"
            )

        entities_df = gr.Dataframe(label="Entities Table")  # For entities table
        relationship_types_df = gr.Dataframe(
            label="Relationship Types Table"
        )  # For relationship types table
        relationships_df = gr.Dataframe(
            label="Relationships Table"
        )  # For relationships table

        wipe_btn = gr.Button("Wipe All Data", variant="stop")
        wipe_output = gr.Markdown("")

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

        async def load_data():
            conn = await asyncpg.connect(
                host=POSTGRES_CONFIG["host"],
                port=POSTGRES_CONFIG["port"],
                user=POSTGRES_CONFIG["user"],
                password=POSTGRES_CONFIG["password"],
                database=POSTGRES_CONFIG["database"],
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

        async def refresh_data():
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
        ) -> DataFrame:
            """Process text files and extract relationships."""

            conn = await asyncpg.connect(
                host=POSTGRES_CONFIG["host"],
                port=POSTGRES_CONFIG["port"],
                user=POSTGRES_CONFIG["user"],
                password=POSTGRES_CONFIG["password"],
                database=POSTGRES_CONFIG["database"],
            )

            if not uri:
                return "Please enter an S3 URI"

            try:
                credentials = (
                    AWSCredentials(key_id, secret) if key_id or secret else None
                )
                pipeline = Pipeline(spark, conn, credentials=credentials)

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
                host=POSTGRES_CONFIG["host"],
                port=POSTGRES_CONFIG["port"],
                user=POSTGRES_CONFIG["user"],
                password=POSTGRES_CONFIG["password"],
                database=POSTGRES_CONFIG["database"],
            )

            await conn.execute(
                "TRUNCATE TABLE relationships, relationship_types, entities CASCADE;"
            )
            await conn.close()

            from gqlalchemy import Memgraph
            from magi.config import MEMGRAPH_CONFIG

            mg = Memgraph(host=MEMGRAPH_CONFIG["host"], port=MEMGRAPH_CONFIG["port"])
            mg.execute("MATCH (n) DETACH DELETE n;")

            return "All data wiped from tables."

        async def wipe_data():
            result = await wipe_all_data()
            # Get refreshed data after wiping
            refreshed_data = await refresh_data()
            return result, *refreshed_data

        async def apply_log_settings(log_level: str, logging_enabled: bool):
            if logging_enabled:
                set_global_log_level(log_level)
                logger.info(f"Logging level set to {log_level}")
            else:
                disable_logging()
                logger.info("Logging disabled")

        browse_btn.click(
            fn=browse_s3,
            inputs=[s3_uri, aws_key_id, aws_secret],
            outputs=[output_md],
        )

        process_btn.click(
            fn=process_files,
            inputs=[s3_uri, aws_key_id, aws_secret],
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
        refresh_tables_btn.click(
            fn=refresh_data,
            outputs=[entities_df, relationship_types_df, relationships_df],
        )

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

        refresh_btn.click(
            fn=async_update,
            outputs=[status_title, status_md],
        )

        # Load data when UI loads
        async def load_initial_data():
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

    return ui
