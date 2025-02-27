import asyncio
import aiohttp
import asyncpg
from gqlalchemy import Memgraph
from ..config import POSTGRES_CONFIG, MEMGRAPH_CONFIG, MEMGRAPH_LAB_CONFIG
from .status import ServiceState, service_status
from pyspark.sql import SparkSession


async def check_memgraph(
    host: str = MEMGRAPH_CONFIG["host"], port: int = MEMGRAPH_CONFIG["port"]
) -> None:
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


async def check_postgres(
    host: str = POSTGRES_CONFIG["host"],
    port: int = POSTGRES_CONFIG["port"],
    db: str = POSTGRES_CONFIG["database"],
    user: str = POSTGRES_CONFIG["user"],
    password: str = POSTGRES_CONFIG["password"],
) -> None:
    """Check PostgreSQL connection using asyncpg."""
    try:
        conn = await asyncpg.connect(
            host=host, port=port, database=db, user=user, password=password
        )
        await conn.execute("SELECT 1")
        await conn.close()
        service_status.update(
            "postgres_(pgvector)",
            ServiceState.OK,
            f"Connected to Postgres at {host}:{port}",
        )
    except Exception as e:
        service_status.update(
            "postgres_(pgvector)",
            ServiceState.ERROR,
            f"Could not connect to Postgres: {str(e)}",
        )


async def check_spark() -> None:
    """Check Spark local mode is running."""
    try:
        # Create a test SparkSession to verify local mode is working
        spark = (
            SparkSession.builder.appName("magi-health-check")
            .master("local[*]")
            .getOrCreate()
        )
        # Run a simple test computation
        test_df = spark.createDataFrame([(1,)], ["test"])
        test_df.count()

        service_status.update(
            "spark_local",
            ServiceState.OK,
            "Spark local mode is running",
        )
        # Don't stop the session as it might be shared
    except Exception as e:
        error_msg = f"Could not initialize Spark local mode: {str(e)}"
        service_status.update("spark_local", ServiceState.ERROR, error_msg)


async def check_memgraph_lab(
    host: str = MEMGRAPH_LAB_CONFIG["host"], port: int = MEMGRAPH_LAB_CONFIG["port"]
) -> None:
    """Check Memgraph Lab UI."""
    async with aiohttp.ClientSession() as session:
        try:
            url = f"http://{host}:{port}"
            async with session.get(url, timeout=3) as resp:
                if resp.status == 200:
                    service_status.update(
                        "memgraph_lab",
                        ServiceState.OK,
                        f"Memgraph Lab is responding at {url}",
                    )
                else:
                    service_status.update(
                        "memgraph_lab",
                        ServiceState.ERROR,
                        f"Memgraph Lab not OK. HTTP status: {resp.status}",
                    )
        except Exception as e:
            service_status.update(
                "memgraph_lab",
                ServiceState.ERROR,
                f"Could not connect to Memgraph Lab: {str(e)}",
            )


async def run_health_checks() -> None:
    """Run all health checks concurrently."""
    await asyncio.gather(
        check_memgraph(),
        check_postgres(),
        check_spark(),
        check_memgraph_lab(),
    )
