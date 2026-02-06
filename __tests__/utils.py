import pytest
import pytest_asyncio
from chatboard.model import NamespaceManager
from chatboard.utils.db_connections import PGConnectionManager
import textwrap
import random
from chatboard.block.block12.block import BlockChunk


@pytest_asyncio.fixture(scope="function")
async def test_db_pool():
    """Create an isolated connection pool for each test."""
    # Close any existing pool
    if PGConnectionManager._pool is not None:
        await PGConnectionManager.close()
    
    # Create a unique pool for this test
    await PGConnectionManager.initialize(
        url=f"postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"
    )
    
    yield
    
    # Clean up this test's pool
    await PGConnectionManager.close()

@pytest_asyncio.fixture()
async def clean_database(test_db_pool):
    # Now uses an isolated pool
    await NamespaceManager.recreate_all_namespaces()
    # NamespaceManager.drop_all_namespaces()
    yield
    # Don't recreate namespaces during teardown to avoid PostgreSQL type cache issues
    # await NamespaceManager.recreate_all_namespaces()
    
    



