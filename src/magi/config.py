"""Configuration module for Magi."""

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def _load_env_file() -> None:
    """Load environment variables from .env file (if exists)."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


class Config:
    """Static configuration class."""

    # Load .env file only in driver process
    if os.environ.get("SPARK_EXECUTOR_ID") is None:
        _load_env_file()

    # Static configuration that gets serialized with Spark tasks
    CONFIG: Dict[str, Any] = {
        "postgres": {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "magidb"),
            "user": os.getenv("POSTGRES_USER", "magiuser"),
            "password": os.getenv("POSTGRES_PASSWORD", "magipassword"),
        },
        "memgraph": {
            "host": os.getenv("MEMGRAPH_HOST", "memgraph"),
            "port": int(os.getenv("MEMGRAPH_PORT", "7687")),
        },
        "memgraph_lab": {
            "host": os.getenv("MEMGRAPH_LAB_HOST", "memgraph-lab"),
            "port": int(os.getenv("MEMGRAPH_LAB_PORT", "3000")),
        },
        "spark": {
            "master_host": os.getenv("SPARK_MASTER_HOST", "spark-master"),
            "master_port": int(os.getenv("SPARK_MASTER_PORT", "8080")),
            "worker_host": os.getenv("SPARK_WORKER_HOST", "spark-worker"),
            "worker_port": int(os.getenv("SPARK_WORKER_PORT", "8081")),
        },
        "magi": {
            "port": int(os.getenv("MAGI_PORT", "1998")),
        },
        "aws": {
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "role_arn": os.getenv("AWS_ROLE_ARN"),
        },
        "file_processor": {
            "max_concurrent_downloads": int(
                os.getenv("MAGI_MAX_CONCURRENT_DOWNLOADS", "50")
            ),
            "batch_size": int(os.getenv("MAGI_BATCH_SIZE", "1000")),
            "max_retries": int(os.getenv("MAGI_MAX_RETRIES", "3")),
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "redis"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
        },
    }


# Export configuration sections as module-level constants
POSTGRES_CONFIG = Config.CONFIG["postgres"]
MEMGRAPH_CONFIG = Config.CONFIG["memgraph"]
MEMGRAPH_LAB_CONFIG = Config.CONFIG["memgraph_lab"]
SPARK_CONFIG = Config.CONFIG["spark"]
MAGI_CONFIG = Config.CONFIG["magi"]
AWS_CONFIG = Config.CONFIG["aws"]
FILE_PROCESSOR_CONFIG = Config.CONFIG["file_processor"]
GEMINI_CONFIG = Config.CONFIG["gemini"]
REDIS_CONFIG = Config.CONFIG["redis"]
