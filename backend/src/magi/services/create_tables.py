async def create_tables(conn, force_recreate=False):
    """Create necessary tables in PostgreSQL with pgvector extension for embedding storage and similarity search.

    Args:
        force_recreate: If True, drop and recreate all tables
    """

    # Create pgvector extension if it doesn't exist
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Drop tables if force_recreate is True
    if force_recreate:
        print("Dropping all tables...")
        await conn.execute("DROP TABLE IF EXISTS relationships;")
        await conn.execute("DROP TABLE IF EXISTS entities;")
        await conn.execute("DROP TABLE IF EXISTS relationship_types;")
        print("Tables dropped.")

    # Create entities table with vector type for embeddings
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            embedding vector(1024),
            from_imported_schema BOOLEAN DEFAULT FALSE
        );
    """)

    # Create relationship_types table with vector type for embeddings
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS relationship_types (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            embedding vector(1024),
            from_imported_schema BOOLEAN DEFAULT FALSE
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
            source_uris TEXT[],
            from_imported_schema BOOLEAN DEFAULT FALSE,
            confidence FLOAT
        );
    """)

    # Create HNSW indexes for cosine similarity search
    # These provide better performance than exact search for large datasets
    try:
        # First, check if the tables have data
        entity_count = await conn.fetchval("SELECT COUNT(*) FROM entities")
        rel_type_count = await conn.fetchval("SELECT COUNT(*) FROM relationship_types")

        # Only attempt to create indexes if tables are empty or we're forcing recreation
        if entity_count == 0 or force_recreate:
            await conn.execute("""
                DROP INDEX IF EXISTS entities_embedding_idx;
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS entities_embedding_idx 
                ON entities USING hnsw (embedding vector_cosine_ops);
            """)
            print("Created entities index")
        else:
            print(f"Skipping entities index creation as table has {entity_count} rows")

        if rel_type_count == 0 or force_recreate:
            await conn.execute("""
                DROP INDEX IF EXISTS relationship_types_embedding_idx;
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS relationship_types_embedding_idx 
                ON relationship_types USING hnsw (embedding vector_cosine_ops);
            """)
            print("Created relationship_types index")
        else:
            print(
                f"Skipping relationship_types index creation as table has {rel_type_count} rows"
            )
    except Exception as e:
        print(f"Error with indexes: {e}")

    await conn.close()


async def reset_database(conn):
    """Drop and recreate all tables."""
    await create_tables(conn, force_recreate=True)
