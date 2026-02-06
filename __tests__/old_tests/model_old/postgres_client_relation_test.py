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





class LikeType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"



class Like(Model):
    type: LikeType = ModelField()
    user_id: int = ModelField(is_foreign_key=True)
    post_id: int = ModelField(is_foreign_key=True)
    
    class Config:
        database_type = "postgres"


class Post(Model):
    title: str = ModelField()
    content: str = ModelField()
    owner_id: int = ModelField(is_foreign_key=True)
    likes: Relation[Like] = RelationField(key="post_id")
    
    class Config:
        database_type = "postgres"



class User(Model):
    name: str = ModelField()
    age: int = ModelField()
    posts: Relation[Post] = RelationField(key="owner_id")
    likes: Relation[Like] = RelationField(key="user_id")
    
    class Config:
        database_type = "postgres"



    
    
    
    

@pytest_asyncio.fixture()
async def seeded_database():
    # Create and seed test data
    
    await connection_manager.init_all_namespaces()

    user1 = await User(name="John Doe 1", age=30).save()
    user2 = await User(name="Jane Doe 2", age=25).save()
    
    # post1 = Post(title="Post 1", content="Content 1", user_id=user1.id)
    # post2 = Post(title="Post 2", content="Content 2", user_id=user1.id)
    # post3 = Post(title="Post 3", content="Content 3", user_id=user2.id)
    
    await user1.posts.add(Post(title="Post 1", content="Content 1"))
    await user1.posts.add(Post(title="Post 2", content="Content 2"))
    await user2.posts.add(Post(title="Post 3", content="Content 3"))
    
    # await Post.batch_upsert([post1, post2, post3])
    
    yield connection_manager
    
    await connection_manager.drop_all_namespaces()
    
    
    
@pytest.mark.asyncio
async def test_seperate_query(seeded_database):
    user1 = await User.get(1)
    assert user1 is not None
    assert user1.name == "John Doe 1"
    
    posts = await user1.posts.limit(10)
    assert len(posts) == 2
    assert posts[0].title == "Post 1"
    assert posts[1].title == "Post 2"
    user2 = await User.get(2)
    assert user2 is not None
    assert user2.name == "Jane Doe 2"
    posts = await user2.posts.limit(10)
    assert len(posts) == 1
    assert posts[0].title == "Post 3"
    
    
@pytest.mark.asyncio
async def test_many_to_many_relation(seeded_database):
    user1 = await User.get(1)
    assert user1 is not None
    assert user1.name == "John Doe 1"
    
    post1 = await Post.get(1)
    assert post1 is not None
    assert post1.title == "Post 1"
    user2 = await User.get(2)
    assert user2 is not None
    assert user2.name == "Jane Doe 2"
    like = await user2.likes.add(Like(type=LikeType.LIKE, post_id=post1.id))
    assert like is not None
    assert like.type == LikeType.LIKE
    assert like.user_id == user2.id
    assert like.post_id == post1.id
    
    likes = await post1.likes.limit(10)
    assert len(likes) == 1
    assert likes[0].type == LikeType.LIKE
    assert likes[0].user_id == user2.id
    assert likes[0].post_id == post1.id
    
    
