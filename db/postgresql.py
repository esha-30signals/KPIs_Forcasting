"""
PostgreSQL Helper for database operations
"""
import os
import json
from typing import Dict, List, Any
import asyncpg
from fastapi import HTTPException

# Global connection pool
_pool = None

def _get_connection_string():
    """Get PostgreSQL connection string from environment variables"""
    database_url = os.getenv("PG_DATABASE_URL")
    if database_url:
        return database_url
    
    # Build from individual components
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

async def _init_connection(conn):
    """Initialize connection with JSONB codec for automatic decoding"""
    await conn.set_type_codec(
        'jsonb',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog',
        format='text'
    )

async def connect() -> bool:
    """
    Connect to PostgreSQL database
    
    Returns:
        bool: True if connected successfully, False otherwise
    """
    global _pool
    try:
        connection_string = _get_connection_string()
        _pool = await asyncpg.create_pool(
            connection_string,
            min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1")),
            max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "5")),
            command_timeout=int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60")),
            timeout=float(os.getenv("POSTGRES_CONNECT_TIMEOUT", "15")),
            init=_init_connection
        )
        
        # Test the connection
        async with _pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        
        print("✅ PostgreSQL connected successfully")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to connect to PostgreSQL: {type(e).__name__}: {e}") from e

async def query(sql: str, *args) -> List[Dict[str, Any]]:
    """
    Run query on PostgreSQL database
    
    Args:
        sql: SQL query string
        *args: Query parameters
        
    Returns:
        List[Dict[str, Any]]: Query results as list of dictionaries
    """
    if _pool is None:
        await connect()
        # raise HTTPException(
        #     status_code=500, 
        #     detail="Not connected to PostgreSQL. Call connect() first."
        # )
    
    try:
        async with _pool.acquire() as conn:
            query_timeout = float(os.getenv("POSTGRES_QUERY_TIMEOUT", os.getenv("POSTGRES_COMMAND_TIMEOUT", "60")))
            rows = await conn.fetch(sql, *args, timeout=query_timeout)
            # Convert asyncpg.Record objects to dictionaries
            # JSONB columns are automatically decoded to Python dicts/list by the codec
            return [dict(row) for row in rows]
    except Exception as e:
        message = f"Query error: {type(e).__name__}: {e}"
        print(message)
        raise HTTPException(status_code=500, detail=message)

async def execute(sql: str, *args) -> str:
    """Run a write/DDL statement on PostgreSQL."""
    if _pool is None:
        await connect()

    try:
        async with _pool.acquire() as conn:
            return await conn.execute(sql, *args)
    except Exception as e:
        print(f"Execute error: {e}")
        raise HTTPException(status_code=500, detail=f"Execute error: {str(e)}")

async def executemany(sql: str, args_list: List[tuple]) -> None:
    """Run a parameterized write statement for many rows."""
    if _pool is None:
        await connect()

    try:
        async with _pool.acquire() as conn:
            await conn.executemany(sql, args_list)
    except Exception as e:
        print(f"Executemany error: {e}")
        raise HTTPException(status_code=500, detail=f"Executemany error: {str(e)}")

async def disconnect():
    """Disconnect from PostgreSQL"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("PostgreSQL connection closed")
