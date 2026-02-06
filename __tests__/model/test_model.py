import os
import uuid
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"

import pytest
import pytest_asyncio

from enum import StrEnum
from typing import TYPE_CHECKING, Any, List, Literal, Type
from uuid import UUID

from pydantic import BaseModel, Field
from chatboard.auth.user_manager import AuthModel
from chatboard.model import Model, ArtifactModel, ModelField, RelationField, ArtifactModel, Relation
import datetime as dt
from chatboard.model.fields import KeyField, RelationField, ModelField
from chatboard.model.namespace_manager import NamespaceManager
from chatboard.block import BlockChunk, ToolCall

from __tests__.utils import clean_database, test_db_pool
    


class LikeType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"

# model_cls = Model
model_cls = ArtifactModel


class Like(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    type: LikeType = ModelField()
    user_id: int = ModelField(foreign_key=True)
    comment_id: int = ModelField(foreign_key=True)

    

class Comment(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    content: str = ModelField()
    reliability: float = ModelField(default=0.5)
    user_id: int = ModelField(foreign_key=True)
    post_id: int = ModelField(foreign_key=True)
    likes: Relation[Like] = RelationField([], foreign_key="comment_id")    



class Post(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    title: str = ModelField()
    content: str = ModelField()
    owner_id: int = ModelField(foreign_key=True)
    comments: Relation[Comment] = RelationField([], foreign_key="post_id")

    



class Message(Model):
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
        
    def block(self) -> BlockChunk:
        tags = []
        if self.role == "user":
            tags = ["user_input"]
        elif self.role == "assistant":
            tags = ["generation"]
        elif self.role == "tool":
            tags = ["tool"]
        else:
            raise ValueError("Invalid role")
        return BlockChunk(
            self.content,
            role=self.role,
            tags=tags,
            id=self.platform_id,
            tool_calls=self.tool_calls,
            model=self.model,
            run_id=self.run_id,
        )



class User(AuthModel):
    age: int = ModelField()
    posts: Relation[Post] = RelationField([], foreign_key="owner_id")
    likes: Relation[Like] = RelationField([], foreign_key="user_id")
    comments: Relation[Comment] = RelationField([], foreign_key="user_id")
    
    







@pytest_asyncio.fixture()
async def seeded_database(clean_database):
    
    # try:
        # Namespaces are already created by clean_database fixture
        await NamespaceManager.create_all_namespaces()

        data = {}
        
        user = await User(name="John", age=30, anonymous_token=uuid.uuid4()).save()

        post1 = await Post(title="Post 1 Title", content="Post 1 Content", owner_id=user.id).save()
        p1_comment1 = await post1.add(Comment(content="post 1 comment 1", user_id=user.id, reliability=0.6))
        p1c1_like1 = await p1_comment1.add(Like(type=LikeType.LIKE, user_id=user.id))
        p1_comment2 = await post1.add(Comment(content="post 1 comment 2", user_id=user.id, reliability=0.7))
        p1c2_like1 = await p1_comment2.add(Like(type=LikeType.LIKE, user_id=user.id))
        p1c2_like2 = await p1_comment2.add(Like(type=LikeType.DISLIKE, user_id=user.id))
        p1_comment3 = await post1.add(Comment(content="post 1 comment 3", user_id=user.id, reliability=0.8))
        p1c3_like1 = await p1_comment3.add(Like(type=LikeType.LIKE, user_id=user.id))

        post2 = await Post(title="Post 2 Title", content="Post 2 Content", owner_id=user.id).save()
        p2_comment1 = await post2.add(Comment(content="post 2 comment 1", user_id=user.id, reliability=0.3))
        p2c1_like1 = await p2_comment1.add(Like(type=LikeType.LIKE, user_id=user.id))
        p2c1_like2 = await p2_comment1.add(Like(type=LikeType.DISLIKE, user_id=user.id))
        p2c1_like3 = await p2_comment1.add(Like(type=LikeType.LIKE, user_id=user.id))
        p2c1_like4 = await p2_comment1.add(Like(type=LikeType.DISLIKE, user_id=user.id))
        p2_comment2 = await post2.add(Comment(content="post 2 comment 2", user_id=user.id, reliability=0.4))
        p2c2_like1 = await p2_comment2.add(Like(type=LikeType.LIKE, user_id=user.id))

        yield {
            "user": user,
            "post1": post1,
            "p1_comment1": p1_comment1,
            "p1c1_like1": p1c1_like1,
            "p1_comment2": p1_comment2,
            "p1c2_like1": p1c2_like1,
            "p1c2_like2": p1c2_like2,
            "p1_comment3": p1_comment3,
            "p1c3_like1": p1c3_like1,
            "post2": post2,
            "p2_comment1": p2_comment1,
            "p2c1_like1": p2c1_like1,
            "p2c1_like2": p2c1_like2,
            "p2c1_like3": p2c1_like3,
            "p2c1_like4": p2c1_like4,
            "p2_comment2": p2_comment2,
            "p2c2_like1": p2c2_like1,
        }
    # finally:
        # NamespaceManager.drop_all_namespaces()




  


@pytest.mark.asyncio
async def test_post_query_with_comments_and_likes(seeded_database):
    # Test querying posts with comments
    posts = await Post.query().limit(10).order_by("created_at").join(Comment)
    assert len(posts) == 2
    assert len(posts[0].comments) == 3
    assert posts[0].comments[0].id == seeded_database['p1_comment1'].id
    assert posts[0].comments[1].id == seeded_database['p1_comment2'].id
    assert posts[0].comments[2].id == seeded_database['p1_comment3'].id
    assert len(posts[1].comments) == 2
    assert posts[1].comments[0].id == seeded_database['p2_comment1'].id
    assert posts[1].comments[1].id == seeded_database['p2_comment2'].id

    # Test querying posts with comments and likes
    posts = await Post.query().limit(10).order_by("created_at").join(Comment.query().join(Like))
    assert len(posts) == 2
    assert len(posts[0].comments) == 3
    assert len(posts[0].comments[0].likes) == 1
    assert len(posts[0].comments[1].likes) == 2
    assert len(posts[0].comments[2].likes) == 1
    assert len(posts[1].comments) == 2
    assert len(posts[1].comments[0].likes) == 4
    assert len(posts[1].comments[1].likes) == 1




@pytest.mark.asyncio
async def test_artifact_model_basic(seeded_database):
    
    comments = await Comment.query().limit(10).include(Like.query().where(type=LikeType.LIKE))
    likes = [l.type for c in comments for l in c.likes] 
    assert len(likes) == 6
    assert likes == [LikeType.LIKE] * 6
    comments = await Comment.query().limit(10).include(Like.query().where(type=LikeType.DISLIKE))
    likes = [l.type for c in comments for l in c.likes] 
    assert len(likes) == 3
    assert likes == [LikeType.DISLIKE] * 3
    
    
    posts = await Post.query().limit(10).include(Comment.query().where(lambda c: c.reliability > 0.5))
    reliabilities = [c.reliability for p in posts for c in p.comments]
    assert len(reliabilities) == 3
    assert reliabilities == [0.6, 0.7, 0.8]
    
    posts = await Post.query().limit(10).include(Comment.query().where(lambda c: c.reliability < 0.5))
    reliabilities = [c.reliability for p in posts for c in p.comments]
    assert len(reliabilities) == 2
    assert reliabilities == [0.3, 0.4]
    
    
    



