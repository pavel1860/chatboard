from typing import Callable, Awaitable, TypeVar, Any, Optional, overload
import asyncio
import asyncpg
import logging
from datetime import datetime
import math
import os
import asyncio
import asyncpg
import traceback
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Any, AsyncContextManager, Callable, TypeVar, cast, AsyncGenerator, Union, Awaitable
from contextlib import asynccontextmanager
import inspect

# import psycopg2
# from psycopg2 import pool
if TYPE_CHECKING:
    from psycopg2.pool import SimpleConnectionPool

# Configure logging for Vercel compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s',
    force=True
)


# --- shared retry machinery -----------------------------------------------

T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)


async def _await_maybe(v: Union[R, Awaitable[R]]) -> R:
    if inspect.isawaitable(v):
        return await cast(Awaitable[R], v)
    return cast(R, v)

def log_database_error(operation: str, sql: str | None = None, values: Any = None,
                       error: Exception | None = None, attempt: int | None = None,
                       max_attempts: int | None = None) -> None:
    ts = datetime.utcnow().isoformat()
    ctx = {
        "timestamp": ts,
        "operation": operation,
        "error_type": type(error).__name__ if error else "Unknown",
        "error_message": str(error) if error else "No error message",
    }
    if attempt is not None and max_attempts is not None:
        ctx["retry_info"] = f"attempt {attempt}/{max_attempts}"
    if sql:
        ctx["sql"] = (sql[:200] + "...") if len(sql) > 200 else sql
    if values:
        ctx["parameters"] = str(values)[:100]
    logger.error(f"DATABASE_ERROR: {ctx}")

async def _sleep_with_backoff(base_delay: float, attempt_index: int, jitter: float = 0.25) -> None:
    # exponential backoff with light jitter
    delay = base_delay * (2 ** attempt_index)
    delay *= (1.0 + (jitter * (2 * (asyncio.get_running_loop().time() % 1.0) - 0.5)))
    await asyncio.sleep(min(delay, 5.0))  # cap to avoid runaway

class RetryRunner:
    """Reusable async retry helper for DB ops (shielded + pool reset callback)."""

    retriable_errors = (asyncpg.ConnectionDoesNotExistError,
                        asyncpg.InterfaceError,
                        ConnectionError)

    @classmethod
    async def run(
        cls,
        *,
        op: str,
        sql: Optional[str],
        values: Any,
        runner: Callable[[], Awaitable[T]],
        reset_pool: Callable[[], Awaitable[None]],
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
    ) -> T:
        max_retries = max_retries if max_retries is not None else int(os.getenv("POSTGRES_MAX_RETRIES", "3"))
        base_delay = base_delay if base_delay is not None else float(os.getenv("POSTGRES_RETRY_DELAY", "0.5"))

        for attempt in range(max_retries + 1):
            try:
                # Shield the actual driver work so cancellation doesn't corrupt connection state
                return await asyncio.shield(runner())
            except cls.retriable_errors as e:
                if attempt == max_retries:
                    log_database_error(op, sql, values, e, attempt + 1, max_retries + 1)
                    raise
                log_database_error(f"{op}_retry", sql, values, e, attempt + 1, max_retries + 1)
                logger.info("🔄 Resetting connection pool due to connection error")
                await reset_pool()
                await _sleep_with_backoff(base_delay, attempt)
            except Exception as e:
                # Non-retriable: just log and bubble up
                log_database_error(op, sql, values, e)
                raise


class Transaction:
    """
    A class representing a database transaction.
    
    This class provides methods for executing queries within a transaction,
    and for committing or rolling back the transaction.
    """
    def __init__(self):
        self.connection: Optional[asyncpg.Connection] = None
        self.transaction: Optional[Any] = None
    
    async def __aenter__(self):
        if PGConnectionManager._pool is None:
            await PGConnectionManager.initialize()
        assert PGConnectionManager._pool is not None, "Pool must be initialized"
        self.connection = await PGConnectionManager._pool.acquire()
        if self.connection is None:
            raise RuntimeError("Failed to acquire a database connection.")
        self.transaction = self.connection.transaction()
        await self.transaction.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.transaction:
            await self.transaction.__aexit__(exc_type, exc_val, exc_tb)
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query within the transaction."""
        if self.connection is None:
            raise RuntimeError("Connection is not initialized.")
        return await self.connection.execute(query, *args)
    
    async def executemany(self, query: str, args_list: List[tuple]) -> None:
        """Execute a query multiple times with different parameters within the transaction."""
        if self.connection is None:
            raise RuntimeError("Connection is not initialized.")
        return await self.connection.executemany(query, args_list)
    
    async def fetch(self, query: str, *args) -> List[dict]:
        """Fetch multiple rows from the database within the transaction as list of dicts."""
        if self.connection is None:
            raise RuntimeError("Connection is not initialized.")
        rows = await self.connection.fetch(query, *args)
        return [dict(row) for row in rows]
    
    async def fetch_one(self, query: str, *args) -> Optional[dict]:
        """Fetch a single row from the database within the transaction as dict."""
        if self.connection is None:
            raise RuntimeError("Connection is not initialized.")
        row = await self.connection.fetchrow(query, *args)
        return dict(row) if row else None
    
    async def commit(self) -> None:
        """Commit the transaction."""
        if self.transaction:
            await self.transaction.commit()
    
    async def rollback(self) -> None:
        """Roll back the transaction."""
        if self.transaction:
            await self.transaction.rollback()


class PGConnectionManager:
    _pool: Optional[asyncpg.Pool] = None
    _init_lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def _safe_close_pool(cls) -> None:
        pool, cls._pool = cls._pool, None
        if not pool:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                return
            await pool.close()
        except Exception:
            pass

    @classmethod
    async def initialize(cls, url: Optional[str] = None) -> None:
        if cls._pool is not None:
            return
        async with cls._init_lock:
            if cls._pool is not None:
                return
            import os
            url = url or os.environ.get("POSTGRES_URL", "postgresql://snack:Aa123456@localhost:5432/promptview_test")
            min_size = int(os.environ.get("POSTGRES_POOL_MIN_SIZE", "1"))
            max_size = int(os.environ.get("POSTGRES_POOL_MAX_SIZE", "5"))
            max_inactive_lifetime = float(os.environ.get("POSTGRES_MAX_INACTIVE_LIFETIME", "30.0"))
            command_timeout = float(os.environ.get("POSTGRES_COMMAND_TIMEOUT", "20.0"))

            logger.info(f"Initializing database connection pool for url: {url}")
            cls._pool = await asyncpg.create_pool(
                dsn=url,
                min_size=min_size,
                max_size=max_size,
                max_inactive_connection_lifetime=max_inactive_lifetime,
                command_timeout=command_timeout,
                server_settings={"application_name": "promptview", "jit": "off"},
            )
            try:
                async with cls._pool.acquire() as conn:
                    ok = await conn.fetchval("SELECT 1")
                    if ok != 1:
                        await cls._safe_close_pool()
                        raise RuntimeError("Database connection test failed")
                logger.info("✅ Database connection pool initialized and verified successfully")
            except Exception:
                await cls._safe_close_pool()
                raise

    # ---- Tiny wrappers using RetryRunner.run ----

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        async def _runner():
            if cls._pool is None:
                await cls.initialize()
            assert cls._pool is not None
            async with cls._pool.acquire() as conn:
                return await conn.execute(query, *args)

        return await RetryRunner.run(
            op="execute",
            sql=query,
            values=args,
            runner=_runner,
            reset_pool=cls._safe_close_pool,
        )

    @classmethod
    async def fetch(cls, query: str, *args) -> list[dict]:
        async def _runner():
            if cls._pool is None:
                await cls.initialize()
            assert cls._pool is not None
            async with cls._pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(r) for r in rows]

        return await RetryRunner.run(
            op="fetch",
            sql=query,
            values=args,
            runner=_runner,
            reset_pool=cls._safe_close_pool,
        )

    @classmethod
    async def fetch_one(cls, query: str, *args) -> Optional[dict]:
        async def _runner():
            if cls._pool is None:
                await cls.initialize()
            assert cls._pool is not None
            async with cls._pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None

        return await RetryRunner.run(
            op="fetch_one",
            sql=query,
            values=args,
            runner=_runner,
            reset_pool=cls._safe_close_pool,
        )

    @overload
    @classmethod
    async def run_in_transaction(cls, func: Callable[[asyncpg.Connection], Awaitable[R]]) -> R: ...
    @overload
    @classmethod
    async def run_in_transaction(cls, func: Callable[[asyncpg.Connection], R]) -> R: ...

    @classmethod
    async def run_in_transaction(cls, func: Callable[[asyncpg.Connection], Union[R, Awaitable[R]]]) -> R:
        async def _runner() -> R:
            if cls._pool is None:
                await cls.initialize()
            assert cls._pool is not None

            async with cls._pool.acquire() as conn:
                async with conn.transaction():
                    # normalize return to R
                    result = func(conn)  # type: ignore[misc]
                    return await _await_maybe(result)

        # If you use the RetryRunner abstraction:
        # from .retry_runner import RetryRunner  # or adjust import path
        return await RetryRunner.run(
            op="transaction",
            sql=None,
            values=None,
            runner=_runner,
            reset_pool=cls._safe_close_pool,
        )

    @classmethod
    async def close(cls) -> None:
        await cls._safe_close_pool()
        

class SyncPGConnectionManager:
    _pool: "SimpleConnectionPool | None" = None

    @classmethod
    def initialize(cls, url: Optional[str] = None) -> None:
        """Initialize the connection pool if not already initialized."""
        import psycopg2
        from psycopg2 import pool
        if cls._pool is None:
            url = url or os.environ.get("POSTGRES_URL", "postgresql://snack:Aa123456@localhost:5432/promptview_test")
            cls._pool = pool.SimpleConnectionPool(
                minconn=5,
                maxconn=20,
                dsn=url
            )

    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize()
        assert cls._pool is not None, "Pool must be initialized"
        return cls._pool.getconn()

    @classmethod
    def put_connection(cls, conn) -> None:
        """Return a connection to the pool."""
        if cls._pool is not None:
            cls._pool.putconn(conn)

    @classmethod
    def close_all(cls) -> None:
        """Close all connections in the pool."""
        if cls._pool is not None:
            cls._pool.closeall()

    @classmethod
    def execute(cls, query: str, *args) -> None:
        """Execute a query with proper connection management."""
        conn = None
        try:
            conn = cls.get_connection()
            with conn.cursor() as cur:
                cur.execute(query, args)
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print_error_sql(query, args, e)
            raise e
        finally:
            if conn:
                cls.put_connection(conn)

    @classmethod
    def fetch(cls, query: str, *args) -> List[dict]:
        """Fetch multiple rows from the database as list of dicts."""
        conn = None
        try:
            conn = cls.get_connection()
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, args)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print_error_sql(query, args, e)
            raise e
        finally:
            if conn:
                cls.put_connection(conn)

    @classmethod
    def fetch_one(cls, query: str, *args) -> Optional[dict]:
        """Fetch a single row from the database as dict."""
        conn = None
        try:
            conn = cls.get_connection()
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, args)
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print_error_sql(query, args, e)
            raise e
        finally:
            if conn:
                cls.put_connection(conn)
        
        
        
        