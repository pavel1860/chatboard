import os
from typing import Literal
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/promptview_test"
from promptview.model.versioning import ArtifactLog, TurnStatus
import pytest
import pytest_asyncio
from enum import StrEnum
from uuid import UUID
from promptview.auth.user_manager import AuthModel
from promptview.model import Model, ArtifactModel, RepoModel, ModelField, Relation, RelationField, ManyRelation, ArtifactModel
import datetime as dt
from promptview.model.fields import KeyField, RelationField, ModelField
from promptview.model.namespace_manager import NamespaceManager
from promptview.prompt.context import Context
from promptview.model.postgres.builder import SQLBuilder





class Message(ArtifactModel):
    content: str = ModelField()
    role: Literal["user", "assistant"] = ModelField()
    user_id: int = ModelField(foreign_key=True)
    

class User(AuthModel):    
    age: int = ModelField()
    messages: Relation[Message] = RelationField(foreign_key="user_id")
    

    
    
    
    
    
@pytest_asyncio.fixture()
async def seeded_user_database():
    await NamespaceManager.create_all_namespaces("users")
    user1 = await User(name="John Doe", email="john@doe.com", age=30, emailVerified=dt.datetime.now()).save()    
    partition = await user1.create_partition("test")
    yield user1, partition
    
    await SQLBuilder.drop_all_tables()
    
    

@pytest.mark.asyncio
async def test_context_save(seeded_user_database):
    user1, partition = seeded_user_database
    async with Context(user1, partition, "revert") as ctx:
        messages = await user1.messages.tail(10)
        assert len(messages) == 0
    turns = await ArtifactLog.list_turns()
    assert len(turns) == 0
    
    
    async with Context(user1, partition, "revert") as ctx:
        message1 = await user1.messages.add(Message(content="Hello, world!", role="user"))
        
        
    turns = await ArtifactLog.list_turns()
    assert len(turns) == 1
    assert turns[0].status == TurnStatus.REVERTED

    
    async with Context(user1, partition, "commit") as ctx:
        message2 = await user1.messages.add(Message(content="Hello, world!", role="user"))
        
    turns = await ArtifactLog.list_turns()
    assert len(turns) == 2
    assert turns[0].status == TurnStatus.COMMITTED
    assert turns[1].status == TurnStatus.REVERTED

