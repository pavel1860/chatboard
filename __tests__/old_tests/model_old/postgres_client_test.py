import os
# os.environ["POSTGRES_URL"] = "postgresql://snack:Aa123456@localhost:5432/chatboard_test"
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"
import pytest
import pytest_asyncio
import datetime as dt
from chatboard.model.fields import ModelField, IndexType
from chatboard.model.model import Model
from chatboard.model.resource_manager import connection_manager
from chatboard.model.vectors.openai_vectorizer import OpenAISmallVectorizer
from chatboard.model.postgres_client import PostgresClient

class BasicPostgresModel(Model):
    created_at: dt.datetime = ModelField(auto_now_add=True)
    updated_at: dt.datetime = ModelField(auto_now=True)
    topic: str = ModelField(index=IndexType.Text, vec="dense")
    content: str = ModelField(vec=["dense"])
    uuid: str = ModelField(index=IndexType.Uuid)
    order: int = ModelField(index=IndexType.Integer)
    
    class VectorSpace:
        dense: OpenAISmallVectorizer
        
    class Config:
        database_type = "postgres"

@pytest_asyncio.fixture()
async def seeded_database():
    # Create and seed test data
    
    await connection_manager.init_all_namespaces()
    
    points = [
        BasicPostgresModel(
            created_at=dt.datetime(2018, 1, 1),
            topic="animals",
            content="turtles are slow",
            uuid="1234",
            order=1
        ),
        BasicPostgresModel(
            created_at=dt.datetime(2019, 1, 1),
            topic="animals",
            content="dolphins are cool",
            uuid="1432",
            order=2,
        ),
        BasicPostgresModel(
            created_at=dt.datetime(2021, 1, 1),
            topic="physics",
            content="quantum mechanics is weird",
            uuid="1154",
            order=3,
        ),
        BasicPostgresModel(
            created_at=dt.datetime(2022, 1, 1),
            topic="physics",
            content="energy is mass and mass is energy",
            uuid="6467",
            order=4,
        ),
        BasicPostgresModel(
            created_at=dt.datetime(2023, 1, 1),
            topic="movies",
            content="the matrix movie came out in 2000",
            uuid="1934",
            order=5,
        ),
        BasicPostgresModel(
            created_at=dt.datetime(2024, 1, 1),
            topic="movies",
            content="friends is the worst show",
            uuid="1934",
            order=6,
        )
    ]
    await BasicPostgresModel.batch_upsert(points)
    
    yield connection_manager
    
    # await BasicPostgresModel.delete_namespace()
    await connection_manager.drop_all_namespaces()

@pytest.mark.asyncio
async def test_basic_search(seeded_database):
    # Test vector search with cosine similarity
    recs = await BasicPostgresModel.similar("I like cats").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "animals"
    
    recs = await BasicPostgresModel.similar("I want to study quantum energy physics").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "physics"
        
    recs = await BasicPostgresModel.similar("what movie should I watch").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "movies"

@pytest.mark.asyncio
async def test_filtering(seeded_database):
    # Test filtering with different conditions
    recs = await BasicPostgresModel.filter(lambda x: x.topic == "animals")
    for rec in recs:
        assert rec.topic == "animals"
        
    recs = await BasicPostgresModel.filter(lambda x: x.topic == "movies")
    for rec in recs:
        assert rec.topic == "movies"
        
    recs = await BasicPostgresModel.filter(lambda x: (x.topic == "animals") | (x.topic == "physics"))
    for rec in recs:
        assert rec.topic in ["animals", "physics"]

@pytest.mark.asyncio
async def test_temporal_query(seeded_database):
    recs = await BasicPostgresModel.filter(lambda x: x.created_at > dt.datetime(2021, 1, 1))
    assert len(recs) == 3
    for rec in recs:
        assert rec.created_at > dt.datetime(2021, 1, 1)
        
    recs = await BasicPostgresModel.filter(lambda x: x.created_at >= dt.datetime(2021, 1, 1))
    assert len(recs) == 4
    for rec in recs:
        assert rec.created_at >= dt.datetime(2021, 1, 1)
          
    recs = await BasicPostgresModel.filter(lambda x: (x.created_at >= dt.datetime(2021, 1, 1)) & (x.created_at < dt.datetime(2024, 1, 1)))
    assert len(recs) == 3
    for rec in recs:
        assert rec.created_at >= dt.datetime(2021, 1, 1) and rec.created_at < dt.datetime(2024, 1, 1)        
        assert rec.topic in ["physics", "movies"]

@pytest.mark.asyncio
async def test_ordering(seeded_database):
    recs = await BasicPostgresModel.limit(20).order_by("order", ascending=True)
    assert [r.order for r in recs] == [1, 2, 3, 4, 5, 6]
    
    recs = await BasicPostgresModel.limit(20).order_by("order", ascending=False)
    assert [r.order for r in recs] == [6, 5, 4, 3, 2, 1]
    
    recs = await BasicPostgresModel.filter(lambda x: x.topic == "movies").order_by("order", ascending=True)
    assert [r.order for r in recs] == [5, 6]
    
    recs = await BasicPostgresModel.filter(lambda x: x.topic == "movies").order_by("order", ascending=False)
    assert [r.order for r in recs] == [6, 5]

@pytest.mark.asyncio
async def test_first_last(seeded_database):
    first_msg = await BasicPostgresModel.first()
    assert type(first_msg) != list
    assert first_msg.created_at == dt.datetime(2018, 1, 1)
    
    last_msg = await BasicPostgresModel.last()
    assert type(last_msg) != list
    assert last_msg.created_at == dt.datetime(2024, 1, 1) 