from typing import Dict, NamedTuple, Optional


class ServiceUI(NamedTuple):
    """UI information for a service."""

    name: str
    url: str


# Define service UI configurations
SERVICE_UIS: Dict[str, Optional[ServiceUI]] = {
    "memgraph": None,
    "memgraph_lab": ServiceUI("Memgraph Lab", "http://localhost:3000"),
    "postgres_(pgvector)": None,
    "spark_local": None,
}
