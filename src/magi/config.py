"""Configuration module for Magi."""

import os

# Service configurations with defaults
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "magidb"),
    "user": os.getenv("POSTGRES_USER", "magiuser"),
    "password": os.getenv("POSTGRES_PASSWORD", "magipassword"),
}

MEMGRAPH_CONFIG = {
    "host": os.getenv("MEMGRAPH_HOST", "memgraph"),
    "port": int(os.getenv("MEMGRAPH_PORT", "7687")),
}

MEMGRAPH_LAB_CONFIG = {
    "host": os.getenv("MEMGRAPH_LAB_HOST", "memgraph-lab"),
    "port": int(os.getenv("MEMGRAPH_LAB_PORT", "3000")),
}

SPARK_CONFIG = {
    "master_host": os.getenv("SPARK_MASTER_HOST", "spark-master"),
    "master_port": int(os.getenv("SPARK_MASTER_PORT", "8080")),
    "worker_host": os.getenv("SPARK_WORKER_HOST", "spark-worker"),
    "worker_port": int(os.getenv("SPARK_WORKER_PORT", "8081")),
}

MAGI_CONFIG = {
    "port": int(os.getenv("MAGI_PORT", "1998")),
}
