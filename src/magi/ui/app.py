from typing import Tuple
import gradio as gr
from .formatters import format_status_markdown, all_services_ok
from ..services.checks import run_health_checks


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Magi System Status", theme=gr.themes.Base()) as ui:
        gr.Markdown("# Magi System Status")

        # Status title above accordion
        status_title = gr.Markdown("⚪ Checking Services...")

        # Create accordion with fixed title
        with gr.Accordion("Service Details", open=False):
            status_md = gr.Markdown("Checking services...")
            refresh_btn = gr.Button("Refresh Status", variant="primary", size="sm")

        async def async_update() -> Tuple[str, str]:
            """Update both status components."""
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

        # Initial status check
        ui.load(
            fn=async_update,
            outputs=[status_title, status_md],
        )

    return ui
