from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from ..ui.app import create_gradio_app


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Magi")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Gradio app
    app = gr.mount_gradio_app(app, create_gradio_app(), path="/")

    return app
