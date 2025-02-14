import asyncio
import aiohttp
import asyncpg
from gqlalchemy import Memgraph
from .status import ServiceState, service_status


async def check_memgraph(host: str = "memgraph", port: int = 7687) -> None:
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
    host: str = "postgres",
    port: int = 5432,
    db: str = "magidb",
    user: str = "magiuser",
    password: str = "magipassword",
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


async def check_spark(
    master_host: str = "spark-master", master_port: int = 8080
) -> None:
    """Check Spark master and worker health via their HTTP endpoints."""
    async with aiohttp.ClientSession() as session:
        try:
            # Check Spark Master UI
            async with session.get(
                f"http://{master_host}:{master_port}", timeout=3
            ) as master_resp:
                if master_resp.status == 200:
                    service_status.update(
                        "spark_master",
                        ServiceState.OK,
                        f"Spark Master UI is responding at {master_host}:{master_port}",
                    )
                else:
                    service_status.update(
                        "spark_master",
                        ServiceState.ERROR,
                        f"Spark Master UI not OK. HTTP status: {master_resp.status}",
                    )

            # Check Spark Worker UI
            async with session.get(
                "http://spark-worker:8081", timeout=3
            ) as worker_resp:
                if worker_resp.status == 200:
                    service_status.update(
                        "spark_worker",
                        ServiceState.OK,
                        "Spark Worker UI is responding",
                    )
                else:
                    service_status.update(
                        "spark_worker",
                        ServiceState.ERROR,
                        f"Spark Worker UI not OK. HTTP status: {worker_resp.status}",
                    )
        except Exception as e:
            error_msg = f"Could not connect to Spark services: {str(e)}"
            service_status.update("spark_master", ServiceState.ERROR, error_msg)
            service_status.update("spark_worker", ServiceState.ERROR, error_msg)


async def check_memgraph_lab(host: str = "memgraph-lab", port: int = 3000) -> None:
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
