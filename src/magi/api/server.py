from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from ..ui.app import create_gradio_app
from ..services.create_tables import create_tables
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    await create_tables()
    yield
    # Clean up on shutdown (if needed)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Magi", lifespan=lifespan)

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
