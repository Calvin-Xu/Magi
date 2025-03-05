from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from ..ui.app import create_gradio_app
from ..services.create_tables import create_tables, reset_database
from contextlib import asynccontextmanager
from ..config import POSTGRES_CONFIG
import asyncpg


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = await asyncpg.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )

    # Check if we should reset the database
    should_reset = os.getenv("MAGI_RESET_DB", "").lower() == "true"

    if should_reset:
        print("Resetting database...")
        await reset_database(conn)
    else:
        # Create tables on startup
        await create_tables(conn)

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
