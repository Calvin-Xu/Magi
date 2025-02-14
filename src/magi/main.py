# src/magi/main.py
from dataclasses import dataclass
from enum import Enum, auto
import threading
from typing import Dict, Optional, NamedTuple
import gradio as gr
import requests
from gqlalchemy import Memgraph
import psycopg2
from pyspark.sql import SparkSession
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


class ServiceState(Enum):
    """Possible states for a service."""

    UNKNOWN = auto()
    OK = auto()
    ERROR = auto()

    def to_emoji(self) -> str:
        """Convert state to an emoji representation."""
        return {
            ServiceState.OK: "✅",
            ServiceState.ERROR: "❌",
            ServiceState.UNKNOWN: "⚪",
        }[self]


class ServiceInfo(NamedTuple):
    """Information about a service's status."""

    state: ServiceState
    message: str


class ServiceUI(NamedTuple):
    """UI information for a service."""

    name: str
    url: str


@dataclass
class ServiceStatus:
    """Thread-safe service status tracker."""

    _lock: threading.Lock
    _status: Dict[str, ServiceInfo]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = {
            "memgraph": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "postgres": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "spark_master": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "spark_worker": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
            "memgraph_lab": ServiceInfo(ServiceState.UNKNOWN, "Not checked yet"),
        }

    def update(self, service: str, state: ServiceState, message: str) -> None:
        """Update the status of a service."""
        with self._lock:
            self._status[service] = ServiceInfo(state, message)

    def get_all(self) -> Dict[str, ServiceInfo]:
        """Get a copy of all service statuses."""
        with self._lock:
            return dict(self._status)


# Initialize global service status
service_status = ServiceStatus()

# Define service UI configurations
SERVICE_UIS: Dict[str, Optional[ServiceUI]] = {
    "memgraph": None,
    "memgraph_lab": ServiceUI("Memgraph Lab", "http://localhost:3000"),
    "postgres": None,
    "spark_master": ServiceUI("Spark Master", "http://localhost:8080"),
    "spark_worker": ServiceUI("Spark Worker", "http://localhost:8081"),
}


def check_memgraph(host: str = "memgraph", port: int = 7687) -> None:
    """Check Memgraph connection."""
    try:
        mg = Memgraph(host=host, port=port)
        mg.execute("RETURN 1;")
        service_status.update(
            "memgraph", ServiceState.OK, f"Connected to Memgraph at {host}:{port}"
        )
    except Exception as e:
        service_status.update(
            "memgraph", ServiceState.ERROR, f"Could not connect to Memgraph: {str(e)}"
        )


def check_postgres(
    host: str = "postgres",
    port: int = 5432,
    db: str = "magidb",
    user: str = "magiuser",
    password: str = "magipassword",
) -> None:
    """Check PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=db, user=user, password=password
        )
        conn.close()
        service_status.update(
            "postgres", ServiceState.OK, f"Connected to Postgres at {host}:{port}"
        )
    except Exception as e:
        service_status.update(
            "postgres", ServiceState.ERROR, f"Could not connect to Postgres: {str(e)}"
        )


def check_spark(master_url: str = "spark://spark-master:7077") -> None:
    """Check Spark connection and UI endpoints."""
    try:
        # Check Spark connection
        spark = (
            SparkSession.builder.master(master_url)
            .appName("MagiHealthCheck")
            .getOrCreate()
        )
        spark.stop()
        service_status.update(
            "spark_master", ServiceState.OK, f"Connected to Spark at {master_url}"
        )

        # Check worker UI
        worker_resp = requests.get("http://spark-worker:8081", timeout=3)
        if worker_resp.status_code == 200:
            service_status.update(
                "spark_worker",
                ServiceState.OK,
                "Spark Worker UI is responding",
            )
        else:
            service_status.update(
                "spark_worker",
                ServiceState.ERROR,
                f"Spark Worker UI not OK. HTTP status: {worker_resp.status_code}",
            )
    except Exception as e:
        service_status.update(
            "spark_master", ServiceState.ERROR, f"Could not connect to Spark: {str(e)}"
        )


def check_memgraph_lab(host: str = "memgraph-lab", port: int = 3000) -> None:
    """Check Memgraph Lab UI."""
    try:
        url = f"http://{host}:{port}"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            service_status.update(
                "memgraph_lab", ServiceState.OK, f"Memgraph Lab is responding at {url}"
            )
        else:
            service_status.update(
                "memgraph_lab",
                ServiceState.ERROR,
                f"Memgraph Lab not OK. HTTP status: {resp.status_code}",
            )
    except Exception as e:
        service_status.update(
            "memgraph_lab",
            ServiceState.ERROR,
            f"Could not connect to Memgraph Lab: {str(e)}",
        )


def run_health_checks() -> None:
    """Run all health checks in parallel."""
    threads = [
        threading.Thread(target=check_memgraph),
        threading.Thread(target=check_postgres),
        threading.Thread(target=check_spark),
        threading.Thread(target=check_memgraph_lab),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def all_services_ok() -> bool:
    """Check if all services are in OK state."""
    return all(
        info.state == ServiceState.OK for info in service_status.get_all().values()
    )


def format_status_markdown() -> str:
    """Format the current status as markdown with clickable links."""
    status = service_status.get_all()

    # Create markdown content
    md = ""  # Remove the header as it's now in the Accordion title

    for service, info in status.items():
        # Format service name and status
        service_name = service.replace("_", " ").title()
        line = f"{info.state.to_emoji()} **{service_name}**: {info.message}"

        # Add UI link if available
        if ui_info := SERVICE_UIS.get(service):
            line += f" ([Open {ui_info.name}: {ui_info.url}]({ui_info.url}))"

        md += line + "\n\n"

    return md


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Magi System Status", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Magi System Status")

        # Run initial health checks
        run_health_checks()

        # Create accordion with dynamic title
        initial_ok = all_services_ok()
        status_emoji = "✅" if initial_ok else "⚠️"
        with gr.Accordion(
            f"{status_emoji} Service Status", open=False
        ) as status_accordion:
            status_md = gr.Markdown(value=format_status_markdown())
            refresh_btn = gr.Button("Refresh Status", variant="primary", size="sm")

        # Update both the markdown and accordion label
        def update_status_and_label() -> tuple[str, str]:
            status_text = format_status_markdown()
            services_ok = all_services_ok()
            new_label = f"{'✅' if services_ok else '⚠️'} Service Status"
            return status_text, new_label

        refresh_btn.click(
            fn=update_status_and_label,
            outputs=[status_md, status_accordion],
        )

    return ui


# Create FastAPI app with CORS
app = FastAPI(title="Magi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, create_gradio_app(), path="/")

if __name__ == "__main__":
    uvicorn.run(
        "magi.main:app",
        host="0.0.0.0",
        port=1998,
        reload=True,
    )
