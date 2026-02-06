import numpy as np
import pytest
import uuid
import pytest_asyncio
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import models
from chatboard.model.qdrant.connection import QdrantConnectionManager

COLLECTION_NAME = "test_collection"

@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_qdrant():
    client = QdrantConnectionManager.get_client()
    if not (await client.collection_exists(COLLECTION_NAME)):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.EUCLID),
        )
    yield
    await client.delete_collection(COLLECTION_NAME)


@pytest.mark.asyncio
async def test_qdrant_insert_and_query():
    client = QdrantConnectionManager.get_client()
    
    # Insert a point
    payload = {"name": "integration-test", "age": 42}
    vector = [0.1, 0.2, 0.3, 0.4]
    point_id = str(uuid.uuid4())

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        ],
    )

    # Run simple query
    results = await QdrantConnectionManager.simple_query(
        collection_name=COLLECTION_NAME,
        filters={"name": "integration-test"},
        limit=5
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["name"] == "integration-test"
    assert results[0]["age"] == 42
