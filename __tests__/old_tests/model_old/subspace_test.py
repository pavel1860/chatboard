import pytest
import pytest_asyncio
import datetime as dt
from chatboard.model.fields import ModelField, IndexType
from chatboard.model.model import Model
from chatboard.model.resource_manager import connection_manager
from chatboard.model.vectors.bm25_vectorizer import BM25Vectorizer
from chatboard.model.vectors.openai_vectorizer import OpenAISmallVectorizer, OpenAILargeVectorizer


class BasicModel(Model):
    created_at: dt.datetime = ModelField(auto_now_add=True)
    updated_at: dt.datetime = ModelField(auto_now=True)
    
    class VectorSpace:
        dense: OpenAISmallVectorizer


class TopicModel(BasicModel):
    topic: str = ModelField(index=IndexType.Text, vec="dense")
    content: str = ModelField(vec=["dense"])
    
    

class MessageModel(BasicModel):    
    content: str = ModelField(vec=["dense"])
    order: int = ModelField(index=IndexType.Integer)





@pytest_asyncio.fixture()
async def seeded_database():
    messages = [
        MessageModel(
            content="I like turtles",
            order=1
        ),
        MessageModel(
            content="hello world",
            order=2
        ),
        MessageModel(
            content="what is up",
            order=3
        ),
        MessageModel(
            content="dolphins are cool",
            order=4
        ),
    ]
    
    topics = [
        TopicModel(
            topic="animals",
            content="I like turtles"
        ),
        TopicModel(
            topic="animals",
            content="dolphins are cool"
        ),
        TopicModel(
            topic="physics",
            content="quantum mechanics is weird"
        ),        
    ]
    
    await MessageModel.batch_upsert(messages)
    await MessageModel.batch_upsert(topics)
    
    conn = connection_manager._qdrant_connection
    
    yield conn
    
    await conn.delete_collection("BasicModel")
    connection_manager._namespaces = {}
    connection_manager._active_namespaces = {}
    
    
    
    
@pytest.mark.asyncio
async def test_collisions(seeded_database):
    recs = await MessageModel.all()
    assert len(recs) == 4
    for rec in recs:
        assert rec.__class__.__name__ == "MessageModel"
        
    recs = await TopicModel.all()
    assert len(recs) == 3
    for rec in recs:
        assert rec.__class__.__name__ == "TopicModel"