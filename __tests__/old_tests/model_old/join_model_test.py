from enum import StrEnum
import os
# os.environ["POSTGRES_URL"] = "postgresql://snack:Aa123456@localhost:5432/chatboard_test"
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"
import pytest
import pytest_asyncio
import datetime as dt
# from chatboard.model.fields import ModelField, IndexType, ModelRelation
# from chatboard.model.model import Model, Relation
# from chatboard.model.resource_manager import connection_manager

from chatboard.model import Model, Relation, ModelField, RelationField, connection_manager




from enum import StrEnum
from typing import TYPE_CHECKING, List, Literal, Type
from uuid import UUID

from pydantic import BaseModel, Field
from chatboard.auth.user_manager import AuthModel
from chatboard.model import Model, ArtifactModel, RepoModel, ModelField, Relation, RelationField, ManyRelation, ArtifactModel
import datetime as dt
from chatboard.model.fields import KeyField, RelationField, ModelField
from chatboard.model.model import ContextModel
from chatboard.model.context import Context as BaseContext
from chatboard.model.namespace_manager import NamespaceManager
from chatboard.prompt import Block, ToolCall




class LikeType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"

# model_cls = Model
model_cls = ArtifactModel


class Like(ArtifactModel):
    type: LikeType = ModelField()
    user_id: int = ModelField(foreign_key=True)
    post_id: UUID = ModelField(foreign_key=True)
    

        


class Post(ArtifactModel):
    title: str = ModelField()
    content: str = ModelField()
    owner_id: int = ModelField(foreign_key=True)
    likes: Relation[Like] = RelationField(foreign_key="post_id")



class Message(ArtifactModel):
    content: str = ModelField(default="")
    name: str | None = ModelField(default=None)
    sender_id: int | None = ModelField(default=None)
    task_id: UUID | None = ModelField(default=None, foreign_key=True)
    role: Literal["user", "assistant", "system", "tool"] = ModelField(default="user")
    platform_id: str | None = ModelField(default=None)
    tool_calls: List[ToolCall] = ModelField(default=[])
    model: str | None = ModelField(default=None)
    run_id: str | None = ModelField(default=None)        
    
        
    def _payload_dump(self, *args, **kwargs):
        res = super()._payload_dump(*args, **kwargs)
        if self.tool_calls:
            res["tool_calls"] = [tool.model_dump() for tool in self.tool_calls]
        return res
        
    def block(self) -> Block:
        tags = []
        if self.role == "user":
            tags = ["user_input"]
        elif self.role == "assistant":
            tags = ["generation"]
        elif self.role == "tool":
            tags = ["tool"]
        else:
            raise ValueError("Invalid role")
        return Block(
            content=self.content,
            role=self.role,
            tags=tags,
            id=self.platform_id,
            tool_calls=self.tool_calls,
            model=self.model,
            run_id=self.run_id,
        )


class User(AuthModel):
    # name: str = ModelField()
    age: int = ModelField()
    posts: Relation[Post] = RelationField(foreign_key="owner_id")
    likes: Relation[Like] = RelationField(foreign_key="user_id")
    # comments: ManyRelation[Comment, UserCommentRel] = RelationField(foreign_key="user_rel_id", junction_keys=["user_id", "comment_id"])
    
    async def send_message(self, content: str):
        return await Message(content=content, name=self.name, role="user", sender_id=self.id).save()
    
    async def list_messages(self, limit: int = 10):
        return await Message.query().tail(limit)
    



@pytest_asyncio.fixture()
async def seeded_database():
    # Create and seed test data
    
    await connection_manager.init_all_namespaces()

    user1 = await User(name="John Doe 1", age=30).save()
    user2 = await User(name="Jane Doe 2", age=25).save()
    
    partition1 =await user1.create_partition("conversation 1", [user2])

    partition2 =await user1.create_partition("conversation 2", [user2])
    
    async with BaseContext(user1, partition1).start_turn() as ctx:
        post1_p1 =await user1.posts.add(Post(title="Post 1 Partition 1", content="Content 1 Partition 1"))
        

    async with BaseContext(user1, partition2).start_turn() as ctx:
        post1_p2 = await user1.posts.add(Post(title="Post 1 Partition 2", content="Content 1 Partition 2"))
    yield user1, user2, partition1, partition2, post1_p1, post1_p2
    
    await connection_manager.drop_all_namespaces()


@pytest.mark.asyncio
async def test_partition_passing_join(seeded_database):
    user1, user2, partition1, partition2, post1_p1, post1_p2 = seeded_database
    posts = await user1.join(Post, partition1).tail(10)
    assert len(posts) == 1
    assert posts[0].id == post1_p1.id

    posts = await user1.join(Post, partition2).tail(10)
    assert len(posts) == 1
    assert posts[0].id == post1_p2.id
    
    
    
    
@pytest.mark.asyncio
async def test_context_partition_join(seeded_database):
    user1, user2, partition1, partition2, post1_p1, post1_p2 = seeded_database   
    async with BaseContext(user1, partition1) as ctx:
        posts = await user1.join(Post).tail(10)
        assert len(posts) == 1
        assert posts[0].id == post1_p1.id


    async with BaseContext(user1, partition2) as ctx:
        posts = await user1.join(Post).tail(10)
        assert len(posts) == 1
        assert posts[0].id == post1_p2.id
