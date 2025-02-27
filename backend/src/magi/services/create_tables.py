import asyncpg
from ..config import POSTGRES_CONFIG


async def create_tables():
    """Create necessary tables in PostgreSQL."""
    conn = await asyncpg.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )

    # Create entities table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            embedding FLOAT8[]
        );
    """)

    # Create relationship_types table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS relationship_types (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            embedding FLOAT8[]
        );
    """)

    # Create relationships table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id SERIAL PRIMARY KEY,
            from_entity INT REFERENCES entities(id),
            relationship_type INT REFERENCES relationship_types(id),
            to_entity INT REFERENCES entities(id),
            constraint_condition TEXT,
            reason TEXT,
            is_causal BOOLEAN,
            FOREIGN KEY (from_entity) REFERENCES entities(id),
            FOREIGN KEY (relationship_type) REFERENCES relationship_types(id),
            FOREIGN KEY (to_entity) REFERENCES entities(id)
        );
    """)

    await conn.close()
