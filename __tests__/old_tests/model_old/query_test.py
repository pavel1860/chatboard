import os
os.environ["QDRANT_URL"] = "http://localhost:6333"
import pytest
import pytest_asyncio
import datetime as dt
from chatboard.model.fields import ModelField, IndexType
from chatboard.model.model import Model
from chatboard.model.resource_manager import connection_manager
from chatboard.model.vectors.bm25_vectorizer import BM25Vectorizer
from chatboard.model.vectors.openai_vectorizer import OpenAISmallVectorizer, OpenAILargeVectorizer

class BasicQueryModel(Model):
    created_at: dt.datetime = ModelField(auto_now_add=True)
    updated_at: dt.datetime = ModelField(auto_now=True)
    topic: str = ModelField(index=IndexType.Text, vec="dense")
    content: str = ModelField(vec=["dense", "sparse"])
    uuid: str = ModelField(index=IndexType.Uuid)
    order: int = ModelField(index=IndexType.Integer)
    
    class VectorSpace:
        dense: OpenAISmallVectorizer
        sparse: BM25Vectorizer




# @pytest_asyncio.fixture(scope="session", autouse=True)
@pytest_asyncio.fixture()
async def seeded_database():
    conn = connection_manager._qdrant_connection
    try:
        _ = await conn.get_collection("BasicQueryModel")
        await conn.delete_collection("BasicQueryModel")
    except Exception as e:
        pass
    points = [
        BasicQueryModel(
            created_at=dt.datetime(2018, 1, 1),
            topic="animals",
            content="turtles are slow",
            uuid="1234",
            order=1
        ),
        BasicQueryModel(
            created_at=dt.datetime(2019, 1, 1),
            topic="animals",
            content="dolphins are cool",
            uuid="1432",
            order=2,
        ),
        BasicQueryModel(
            created_at=dt.datetime(2021, 1, 1),
            topic="physics",
            content="quantum mechanics is weird",
            uuid="1154",
            order=3,
        ),
        BasicQueryModel(
            created_at=dt.datetime(2022, 1, 1),
            topic="physics",
            content="energy is mass and mass is energy",
            uuid="6467",
            order=4,
        ),
        BasicQueryModel(
            created_at=dt.datetime(2023, 1, 1),
            topic="movies",
            content="the matrix movie came out in 2000",
            uuid="1934",
            order=5,
        ),
        BasicQueryModel(
            created_at=dt.datetime(2024, 1, 1),
            topic="movies",
            content="friends is the worst show",
            uuid="1934",
            order=6,
        )
    ]
    await BasicQueryModel.batch_upsert(points)
    

    yield connection_manager
    
    await conn.delete_collection("BasicModel")
    connection_manager._namespaces = {}
    connection_manager._active_namespaces = {}


@pytest.mark.asyncio
async def test_query(seeded_database):
    # seeded_database.reset_connection()
    recs = await BasicQueryModel.similar("I like cats").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "animals"
    
    recs = await BasicQueryModel.similar("I want to study quantom energy physics").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "physics"
        
    recs = await BasicQueryModel.similar("what movie should I watch").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "movies"
        
        
@pytest.mark.asyncio
async def test_named_query(seeded_database):
    recs = await BasicQueryModel.similar("I like cats", vec="dense").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "animals"
    
    recs = await BasicQueryModel.similar("I want to study physics", vec="dense").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "physics"
        
    recs = await BasicQueryModel.similar("what movie should I watch", vec="dense").limit(2)
    assert len(recs) == 2
    for rec in recs:
        assert rec.topic == "movies"
        
        
        
        
@pytest.mark.asyncio        
async def test_filtering(seeded_database):
    recs = await BasicQueryModel.similar("the matrix is a great movie").filter(lambda x: x.topic == "animals")
    for rec in recs:
        assert rec.topic == "animals"
        
    recs = await BasicQueryModel.similar("the matrix is a great movie").filter(lambda x: x.topic == "movies")
    for rec in recs:
        assert rec.topic == "movies"
        
    recs = await BasicQueryModel.similar("the matrix is a great movie").filter(lambda x: (x.topic == "animals") | (x.topic == "physics"))
    for rec in recs:
        assert rec.topic in ["animals", "physics"]
        
        
        
        
        
@pytest.mark.asyncio
async def test_first_last(seeded_database):
    first_msg = await BasicQueryModel.first()
    assert type(first_msg) != list
    assert first_msg.created_at == dt.datetime(2018, 1, 1)
    last_msg = await BasicQueryModel.last()
    assert type(last_msg) != list
    assert last_msg.created_at == dt.datetime(2024, 1, 1)
    
    

@pytest.mark.asyncio
async def test_and_field_query(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: (x.topic == "animals") & (x.content == "turtles are slow")).limit(10)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.topic == "animals"
    assert rec.content == "turtles are slow"



@pytest.mark.asyncio
async def test_or_field_query(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: (x.topic == "animals") | (x.content == "quantum mechanics is weird")).limit(10)
    assert len(recs) == 3
    for rec in recs:
        assert rec.topic == "animals" or rec.content == "quantum mechanics is weird"
        


@pytest.mark.asyncio
async def test_or_and_filtering(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: ((x.topic == "animals") | (x.topic == "physics")) & ((x.order >= 2) & (x.order <= 3)))    
    assert len(recs) == 2


@pytest.mark.asyncio
async def test_ordering(seeded_database):
    recs = await BasicQueryModel.limit(20).order_by("order", ascending=True)
    assert [r.order for r in recs] == [1, 2, 3, 4, 5, 6]
    
    recs = await BasicQueryModel.limit(20).order_by("order", ascending=False)
    assert [r.order for r in recs] == [6, 5, 4, 3, 2, 1]
    
    recs = await BasicQueryModel.filter(lambda x: x.topic == "movies").order_by("order", ascending=True)
    assert [r.order for r in recs] == [5, 6]
    
    recs = await BasicQueryModel.filter(lambda x: x.topic == "movies").order_by("order", ascending=False)
    assert [r.order for r in recs] == [6, 5]
    
    recs = await BasicQueryModel.filter(lambda x: x.topic == "physics").order_by("order", ascending=True)
    assert [r.order for r in recs] == [3, 4]
    
    recs = await BasicQueryModel.filter(lambda x: x.topic == "physics").order_by("order", ascending=False)
    assert [r.order for r in recs] == [4, 3]
    


@pytest.mark.asyncio
async def test_temporal_query(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: (x.created_at > dt.datetime(2021, 1, 1)))
    assert len(recs) == 3
    for rec in recs:
        assert rec.created_at > dt.datetime(2021, 1, 1)
        
    recs = await BasicQueryModel.filter(lambda x: (x.created_at >= dt.datetime(2021, 1, 1)))
    assert len(recs) == 4
    for rec in recs:
        assert rec.created_at >= dt.datetime(2021, 1, 1)
          
    recs = await BasicQueryModel.filter(lambda x: (x.created_at >= dt.datetime(2021, 1, 1)) & (x.created_at < dt.datetime(2024, 1, 1)))
    assert len(recs) == 3
    for rec in recs:
        assert rec.created_at >= dt.datetime(2021, 1, 1) and rec.created_at < dt.datetime(2024, 1, 1)        
        assert rec.topic in ["physics", "movies"]
    
    recs = await BasicQueryModel.filter(lambda x: (x.created_at >= dt.datetime(2021, 1, 1)) & (x.created_at < dt.datetime(2024, 1, 1))).order_by("created_at", ascending=True)
    assert [r.topic for r in recs] == ['physics', 'physics', 'movies']
    
    recs = await BasicQueryModel.filter(lambda x: (x.created_at >= dt.datetime(2021, 1, 1)) & (x.created_at < dt.datetime(2024, 1, 1))).order_by("created_at", ascending=False)    
    assert [r.topic for r in recs] == ['movies', 'physics', 'physics']


@pytest.mark.asyncio
async def test_query_building():
    
    query = BasicQueryModel.filter(lambda x: x.topic == "animals")
    query = query.filter(lambda x: x.content == "turtles are slow")
    recs = await query.limit(1)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.topic == "animals"
    assert rec.content == "turtles are slow"
    
    
    
    
@pytest.mark.asyncio
async def test_query_date_range(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: dt.datetime(2023, 1, 1) >= x.created_at >= dt.datetime(2021, 1, 1))
    assert len(recs) == 3
    recs = await BasicQueryModel.filter(lambda x: dt.datetime(2023, 1, 1) > x.created_at >= dt.datetime(2021, 1, 1))
    assert len(recs) == 2
    recs = await BasicQueryModel.filter(lambda x: dt.datetime(2023, 1, 1) > x.created_at > dt.datetime(2021, 1, 1))
    assert len(recs) == 1
    
    
    
@pytest.mark.asyncio
async def test_query_date_range_ordering(seeded_database):
    recs = await BasicQueryModel.filter(lambda x: dt.datetime(2023, 1, 1) >= x.created_at >= dt.datetime(2021, 1, 1)).order_by("created_at", ascending=True)
    assert len(recs) == 3
    for prv, nxt in zip(recs[:-1], recs[1:]):
        assert prv.created_at <= nxt.created_at
        
    recs = await BasicQueryModel.filter(lambda x: dt.datetime(2023, 1, 1) >= x.created_at >= dt.datetime(2021, 1, 1)).order_by("created_at", ascending=False)
    for prv, nxt in zip(recs[:-1], recs[1:]):
        assert prv.created_at >= nxt.created_at
    