from typing import Dict, Optional, NamedTuple


class ServiceUI(NamedTuple):
    """UI information for a service."""

    name: str
    url: str


# Define service UI configurations
SERVICE_UIS: Dict[str, Optional[ServiceUI]] = {
    "memgraph": None,
    "memgraph_lab": ServiceUI("Memgraph Lab", "http://localhost:3000"),
    "postgres_(pgvector)": None,
    "spark_master": ServiceUI("Spark Master", "http://localhost:8080"),
    "spark_worker": ServiceUI("Spark Worker", "http://localhost:8081"),
}
