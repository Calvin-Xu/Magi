"""Configuration module for Magi."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env_file() -> None:
    """Load environment variables from .env file (if exists)."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL database configuration."""

    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "postgres"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "magidb"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "magiuser"))
    password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "magipassword")
    )


@dataclass(frozen=True)
class MemgraphConfig:
    """Memgraph database configuration."""

    host: str = field(default_factory=lambda: os.getenv("MEMGRAPH_HOST", "memgraph"))
    port: int = field(default_factory=lambda: int(os.getenv("MEMGRAPH_PORT", "7687")))


@dataclass(frozen=True)
class MemgraphLabConfig:
    """Memgraph Lab configuration."""

    host: str = field(
        default_factory=lambda: os.getenv("MEMGRAPH_LAB_HOST", "memgraph-lab")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("MEMGRAPH_LAB_PORT", "3000"))
    )


@dataclass(frozen=True)
class MagiConfig:
    """Magi application configuration."""

    port: int = field(default_factory=lambda: int(os.getenv("MAGI_PORT", "1998")))


@dataclass(frozen=True)
class AWSConfig:
    """AWS configuration."""

    access_key_id: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    secret_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    role_arn: Optional[str] = field(default_factory=lambda: os.getenv("AWS_ROLE_ARN"))


@dataclass(frozen=True)
class FileProcessorConfig:
    """File processor configuration."""

    max_concurrent_downloads: int = field(
        default_factory=lambda: int(os.getenv("MAGI_MAX_CONCURRENT_DOWNLOADS", "50"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("MAGI_BATCH_SIZE", "100"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAGI_MAX_RETRIES", "3"))
    )


@dataclass(frozen=True)
class RedisConfig:
    """Redis configuration."""

    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "redis"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini API configuration."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))


@dataclass(frozen=True)
class VoyageAIConfig:
    """Voyage AI configuration."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("VOYAGE_AI_API_KEY")
    )


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI configuration."""

    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter configuration."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )


@dataclass(frozen=True)
class PerplexityConfig:
    """Perplexity API configuration."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("PERPLEXITY_API_KEY")
    )


# Load .env file only in driver process
if os.environ.get("SPARK_EXECUTOR_ID") is None:
    _load_env_file()

# Create and export configuration instances as module-level constants
POSTGRES_CONFIG = PostgresConfig()
MEMGRAPH_CONFIG = MemgraphConfig()
MEMGRAPH_LAB_CONFIG = MemgraphLabConfig()
MAGI_CONFIG = MagiConfig()
AWS_CONFIG = AWSConfig()
FILE_PROCESSOR_CONFIG = FileProcessorConfig()
REDIS_CONFIG = RedisConfig()
GEMINI_CONFIG = GeminiConfig()
VOYAGE_AI_CONFIG = VoyageAIConfig()
OPENAI_CONFIG = OpenAIConfig()
OPENROUTER_CONFIG = OpenRouterConfig()
PERPLEXITY_CONFIG = PerplexityConfig()
