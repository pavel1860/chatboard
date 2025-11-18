from typing import List, Literal
import pytest
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.model3 import Model
from promptview.model.fields import KeyField, ModelField, RelationField
from promptview.model.relation_model import RelationModel
from promptview.model.postgres2.pg_query_set import select
from promptview.utils.db_connections import PGConnectionManager
import pytest_asyncio
# @pytest.fixture(scope="module", autouse=True)
# async def setup_db():
#     """Ensure we start with a clean DB schema for the tests."""
#     # Drop all namespaces (tables)
#     NamespaceManager.drop_all_tables()
#     yield
#     # Drop again after tests
#     NamespaceManager.drop_all_tables()

@pytest_asyncio.fixture()
async def setup_db():
    """Ensure we start with a clean DB schema for the tests."""
    # Drop all namespaces (tables)
    NamespaceManager.drop_all_tables()
    yield
    # Drop again after tests
    NamespaceManager.drop_all_tables()



@pytest.mark.asyncio
async def test_one_to_one(setup_db):
    class Profile(Model):
        id: int = KeyField(primary_key=True)
        bio: str = ModelField()
        user_id: int = ModelField(foreign_key=True)

    class User(Model):
        id: int = KeyField(primary_key=True)
        name: str = ModelField()
        profile: Profile | None = RelationField(foreign_key="user_id")

    await NamespaceManager.initialize_all()

    u = await User(name="Alice").save()
    p = await Profile(bio="Hello", user_id=u.id).save()

    result = await select(User).include(Profile).execute()
    assert len(result) == 1
    assert result[0].profile.bio == "Hello"

@pytest.mark.asyncio
async def test_one_to_many(setup_db):
    class Post(Model):
        id: int = KeyField(primary_key=True)
        title: str = ModelField()
        user_id: int = ModelField(foreign_key=True)

    class User(Model):
        id: int = KeyField(primary_key=True)
        name: str = ModelField()
        posts: list[Post] = RelationField(foreign_key="user_id")

    await NamespaceManager.initialize_all()

    u1 = await User(name="Bob").save()
    u2 = await User(name="Charlie").save()

    await Post(title="Post1", user_id=u1.id).save()
    await Post(title="Post2", user_id=u1.id).save()
    await Post(title="Post3", user_id=u2.id).save()

    results = await select(User).include(Post).execute()
    user_map = {u.name: u for u in results}
    assert len(user_map["Bob"].posts) == 2
    assert len(user_map["Charlie"].posts) == 1

@pytest.mark.asyncio
async def test_many_to_many(setup_db):
    class Participant(RelationModel):
        id: int = KeyField(primary_key=True)
        user_id: int = ModelField(foreign_key=True)
        conv_id: int = ModelField(foreign_key=True)

    class Conversation(Model):
        id: int = KeyField(primary_key=True)
        title: str = ModelField()

    class User(Model):
        id: int = KeyField(primary_key=True)
        name: str = ModelField()
        conversations: list[Conversation] = RelationField(
            primary_key="id",
            foreign_key="id",
            junction_keys=["user_id", "conv_id"],
            junction_model=Participant
        )

    await NamespaceManager.initialize_all()

    u1 = await User(name="Jon").save()
    u2 = await User(name="Snow").save()

    c1 = await Conversation(title="C1").save()
    c2 = await Conversation(title="C2").save()
    c3 = await Conversation(title="C3").save()

    await Participant(user_id=u1.id, conv_id=c1.id).save()
    await Participant(user_id=u1.id, conv_id=c2.id).save()
    await Participant(user_id=u2.id, conv_id=c2.id).save()
    await Participant(user_id=u2.id, conv_id=c3.id).save()

    results = await select(User).include(Conversation).execute()
    user_map = {u.name: u for u in results}
    assert {c.title for c in user_map["Jon"].conversations} == {"C1", "C2"}
    assert {c.title for c in user_map["Snow"].conversations} == {"C2", "C3"}
