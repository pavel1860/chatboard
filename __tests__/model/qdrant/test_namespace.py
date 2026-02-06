import pytest
import uuid
import numpy as np
from qdrant_client.http.models import VectorParams, Distance

from chatboard.model.qdrant.namespace import QdrantNamespace
from chatboard.model.qdrant.connection import QdrantConnectionManager
from chatboard.model.base_namespace import NSFieldInfo


class FakeModel:
    def __init__(self, id, content, embedding):
        self.id = id
        self.content = content
        self.embedding = embedding

    def model_dump(self):
        return {"id": self.id, "content": self.content}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["content"], data.get("embedding"))


@pytest.mark.asyncio
async def test_qdrant_namespace_save_get_delete():
    # Setup
    collection_name = "test_ns_collection"
    client = QdrantConnectionManager.get_client()

    # Ensure collection exists
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )

    # Create a QdrantNamespace manually
    ns = QdrantNamespace(name=collection_name)
    ns._model_cls = FakeModel

    # Define primary key and vector fields
    pk_field = NSFieldInfo(name="id", field_type=str, extra={"primary_key": True})
    vector_field = NSFieldInfo(
        name="embedding",
        field_type=list[float],
        extra={"is_vector": True, "dimension": 3}
    )
    ns._fields = {"id": pk_field, "content": NSFieldInfo("content", str)}
    ns._vector_fields = {"embedding": vector_field}
    ns._primary_key = pk_field

    # Manually stub transform_model
    async def fake_transform_model(model):
        return {"embedding": np.array(model.embedding, dtype=np.float32)}
    ns.transform_model = fake_transform_model

    # Save a record
    model = FakeModel(str(uuid.uuid4()), "test content", [0.1, 0.2, 0.3])
    saved = await ns.save(model)

    assert isinstance(saved, FakeModel)
    assert saved.content == "test content"

    # Get it back
    fetched = await ns.get(model.id)
    assert isinstance(fetched, FakeModel)
    assert fetched.id == model.id
    assert fetched.content == model.content

    # Delete it
    deleted = await ns.delete(model.id)
    assert isinstance(deleted, FakeModel)
    assert deleted.id == model.id

    # Verify it's gone
    assert await ns.get(model.id) is None
