from typing import Tuple, Optional
import gradio as gr
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .formatters import format_status_markdown, all_services_ok
from ..services.checks import run_health_checks
from ..services.s3 import list_s3_objects
from ..services.aws import AWSCredentials
from ..services.pipeline import Pipeline


# Constants for display
MAX_FIELD_LENGTH = 100  # Maximum length for content display
MAX_ROWS_DISPLAY = 10  # Number of rows to show in UI
MAX_RELATIONSHIPS_DISPLAY = 3  # Maximum number of relationships to show


def truncate_field(text: str, max_length: int = MAX_FIELD_LENGTH) -> str:
    """Truncate text field for display."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_dataframe_sample(df, num_rows: int = MAX_ROWS_DISPLAY) -> str:
    """Format a sample of the DataFrame for display."""
    # Select needed columns and limit before converting
    df = df.select("uri", "content", "relationships_json").limit(num_rows)
    pdf = df.toPandas()

    # Truncate both content and relationships
    pdf["content"] = pdf["content"].apply(truncate_field)
    pdf["relationships_json"] = pdf["relationships_json"].apply(truncate_field)

    return pdf.to_string()


def format_relationships(df) -> str:
    """Format extracted relationships for display."""
    # Apply limit before expensive operations
    df = df.limit(MAX_ROWS_DISPLAY)

    # Select only needed columns and explode
    rel_df = (
        df.select("uri", "extracted_relationships")
        .withColumn("relationship", F.explode("extracted_relationships"))
        .limit(MAX_RELATIONSHIPS_DISPLAY)
    )

    # Convert to pandas more efficiently
    pdf = rel_df.toPandas()

    # Format each relationship
    formatted_rels = []
    for _, row in pdf.iterrows():
        rel = row["relationship"]
        formatted_rels.append(
            f"- {rel['from_entity']} → {rel['relationship_type']} → {rel['to_entity']}\n"
            f"  Description: {rel['relationship_description']}\n"
            f"  Reason: {truncate_field(rel['reason'])}\n"
            f"  From: {truncate_field(rel['from_entity_description'])}\n"
            f"  To: {truncate_field(rel['to_entity_description'])}\n"
            f"  Constraint: {rel['constraint_condition']}\n"
            f"  Is causal: {rel['is_causal']}\n"
            f"  From document: {row['uri']}\n"
        )

    return "\n".join(formatted_rels)


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio interface."""
    # Initialize Spark
    spark = SparkSession.builder.appName("magi").getOrCreate()

    with gr.Blocks(title="Magi System Status", theme=gr.themes.Base()) as ui:
        # Service Status Section
        gr.Markdown("# Magi System Status")
        status_title = gr.Markdown("⚪ Checking Services...")
        with gr.Accordion("Service Details", open=False):
            status_md = gr.Markdown("Checking services...")
            refresh_btn = gr.Button("Refresh Status", variant="primary", size="sm")

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
            process_btn = gr.Button("Ingest Files", variant="primary")

        output_md = gr.Markdown("")

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

        async def process_files(
            uri: str,
            key_id: Optional[str],
            secret: Optional[str],
        ) -> str:
            """Process text files and extract relationships."""
            if not uri:
                return "Please enter an S3 URI"

            try:
                credentials = (
                    AWSCredentials(key_id, secret) if key_id or secret else None
                )
                pipeline = Pipeline(credentials=credentials)

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

                # Format results with error handling
                try:
                    sample_str = format_dataframe_sample(first_df)
                    rels_str = format_relationships(first_df)
                except Exception as e:
                    return f"Error formatting results: {str(e)}"

                result = [
                    f"Processed {total_documents} text documents\n",
                    "Sample of documents:",
                    "```",
                    sample_str,
                    "```\n",
                    "Sample of extracted relationships:",
                    "```",
                    rels_str,
                    "```",
                ]

                return "\n".join(result)

            except Exception as e:
                print(f"Pipeline error: {str(e)}")  # Debug
                import traceback

                print(f"Traceback: {traceback.format_exc()}")  # Debug full traceback
                return f"Error processing files: {str(e)}"

        browse_btn.click(
            fn=browse_s3,
            inputs=[s3_uri, aws_key_id, aws_secret],
            outputs=[output_md],
        )

        process_btn.click(
            fn=process_files,
            inputs=[s3_uri, aws_key_id, aws_secret],
            outputs=[output_md],
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

        ui.load(
            fn=async_update,
            outputs=[status_title, status_md],
        )

    return ui
