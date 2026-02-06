import os
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"
import pytest
import pytest_asyncio
from enum import StrEnum
from uuid import UUID
from chatboard.auth.user_manager import AuthModel
from chatboard.model import Model, ArtifactModel, RepoModel, ModelField, Relation, RelationField, ManyRelation, ArtifactModel
import datetime as dt
from chatboard.model.fields import KeyField, RelationField, ModelField
from chatboard.model.namespace_manager import NamespaceManager
from chatboard.prompt import Context
from chatboard.model.postgres.builder import SQLBuilder




class LikeType(StrEnum):
    LIKE = "like"
    DISLIKE = "dislike"

# model_cls = Model
model_cls = ArtifactModel


class Like(ArtifactModel):
    type: LikeType = ModelField()
    user_id: int = ModelField(foreign_key=True)
    post_id: UUID = ModelField(foreign_key=True)
    

class Comment(ArtifactModel):
    content: str = ModelField()
    post_id: UUID = ModelField(foreign_key=True)
    # user_rel_id: int = ModelField(foreign_key=True)
    

        
        
class UserCommentRel(ArtifactModel):
    user_id: int = ModelField(foreign_key=True)
    comment_id: UUID = ModelField(foreign_key=True)



class Post(ArtifactModel):
    title: str = ModelField()
    content: str = ModelField()
    owner_id: int = ModelField(foreign_key=True)
    likes: Relation[Like] = RelationField(foreign_key="post_id")
    comments: ManyRelation[Comment, UserCommentRel] = RelationField(foreign_key="post_id", junction_keys=["post_id", "comment_id"])



class User(AuthModel):
    # name: str = ModelField()
    age: int = ModelField()
    posts: Relation[Post] = RelationField(foreign_key="owner_id")
    likes: Relation[Like] = RelationField(foreign_key="user_id")
    # comments: ManyRelation[Comment, UserCommentRel] = RelationField(foreign_key="user_rel_id", junction_keys=["user_id", "comment_id"])
    comments: ManyRelation[Comment, UserCommentRel] = RelationField(primary_key="id", junction_keys=["user_id", "comment_id"], foreign_key="id")
    
    
    
    
    
    
    
    
@pytest_asyncio.fixture()
async def seeded_user_database():
    await NamespaceManager.create_all_namespaces("users")
    user1 = await User(name="John Doe", email="john@doe.com", age=30, emailVerified=dt.datetime.now()).save()
    user2 = await User(name="Jane Doe", email="jane@doe.com", age=25, emailVerified=dt.datetime.now()).save()
    
    yield user1, user2
    
    await SQLBuilder.drop_all_tables()
    

@pytest_asyncio.fixture()
async def seeded_posts_database(seeded_user_database):
    user1, user2 = seeded_user_database
    async with Context(user1).start_turn() as ctx:
        post1_u1_v1 = await user1.posts.add(Post(title="Post 1 User 1", content="Content 1 User 1 Version 1"))
    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 1 User 1 Version 2"
        post1_u1_v2 = await post.save()
        
    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 1 User 1 Version 3"
        post1_u1_v3 = await post.save()
        
    async with Context(user1).start_turn() as ctx:
        post2_u1_v1 = await user1.posts.add(Post(title="Post 2 User 1", content="Content 2 User 1 Version 1"))
        
    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 2 User 1 Version 2"
        post2_u1_v2 = await post.save()
        
    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 2 User 1 Version 3"
        post2_u1_v3 = await post.save()        
    
    
    async with Context(user2).start_turn() as ctx:
        post1_u2_v1 = await user2.posts.add(Post(title="Post 1 User 2", content="Content 1 User 2 Version 1"))    
        
    async with Context(user2).start_turn() as ctx:
        post = await user2.posts.last()
        post.content = "Content 1 User 2 Version 2"
        post1_u2_v2 = await post.save()
        
    posts_u1 = [post1_u1_v1, post1_u1_v2, post1_u1_v3, post2_u1_v1, post2_u1_v2, post2_u1_v3]
    posts_u2 = [post1_u2_v1, post1_u2_v2]
    yield user1, user2, posts_u1, posts_u2
    
    await SQLBuilder.drop_all_tables()
        


        
    
@pytest.mark.asyncio
async def test_basic_versioning(seeded_user_database):
    user1, user2 = seeded_user_database
        

    async with Context(user1).start_turn() as ctx:
        post1_u1_v1 = await user1.posts.add(Post(title="Post 1 User 1", content="Content 1 User 1 Version 1"))
        assert post1_u1_v1.id is not None
        assert post1_u1_v1.artifact_id is not None
        assert post1_u1_v1.version == 1
        assert post1_u1_v1.title == "Post 1 User 1"


    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        assert post.id == post1_u1_v1.id
        assert post.artifact_id == post1_u1_v1.artifact_id
        assert post.version == 1
        assert post.title == "Post 1 User 1"
        post.content = "Content 1 User 1 Version 2"
        post1_u1_v2 = await post.save()
        assert post1_u1_v2.id != post1_u1_v1.id
        assert post1_u1_v2.artifact_id == post1_u1_v1.artifact_id
        assert post1_u1_v2.version == 2
        assert post1_u1_v2.content == "Content 1 User 1 Version 2"
        
    

    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        assert post.id == post1_u1_v2.id
        assert post.id != post1_u1_v1.id
        assert post.version == 2
        assert post.artifact_id == post1_u1_v1.artifact_id
        assert post.artifact_id == post1_u1_v2.artifact_id
        post.content = "Content 1 User 1 Version 3"
        post1_u1_v3 = await post.save()
        assert post1_u1_v3.id != post1_u1_v2.id
        assert post1_u1_v3.version == 3
        assert post1_u1_v3.artifact_id == post1_u1_v1.artifact_id
        
        


    async with Context(user1).start_turn() as ctx:
        post2_u1_v1 = await user1.posts.add(Post(title="Post 2 User 1", content="Content 2 User 1 Version 1"))
        assert post2_u1_v1.id is not None
        assert post2_u1_v1.artifact_id is not None
        assert post2_u1_v1.version == 1
        assert post2_u1_v1.title == "Post 2 User 1"
        assert post2_u1_v1.artifact_id != post1_u1_v1.artifact_id
        assert post2_u1_v1.artifact_id != post1_u1_v2.artifact_id
        assert post2_u1_v1.artifact_id != post1_u1_v3.artifact_id
        


    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        assert post.id == post2_u1_v1.id
        assert post.artifact_id == post2_u1_v1.artifact_id
        assert post.version == 1
        assert post.title == "Post 2 User 1"
        post.content = "Content 2 User 1 Version 2"
        post2_u1_v2 = await post.save()
        assert post2_u1_v2.id != post2_u1_v1.id
        assert post2_u1_v2.version == 2
        assert post2_u1_v2.artifact_id == post2_u1_v1.artifact_id
        assert post2_u1_v2.content == "Content 2 User 1 Version 2"
        

    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 2 User 1 Version 3"
        post2_u1_v3 = await post.save()
        assert post2_u1_v3.id != post2_u1_v2.id
        assert post2_u1_v3.version == 3
        assert post2_u1_v3.artifact_id == post2_u1_v1.artifact_id
        assert post2_u1_v3.content == "Content 2 User 1 Version 3"
    
    
    # testing user2
    
    async with Context(user2).start_turn() as ctx:
        post1_u2_v1 = await user2.posts.add(Post(title="Post 1 User 2", content="Content 1 User 2 Version 1"))
        assert post1_u2_v1.id is not None
        assert post1_u2_v1.artifact_id is not None
        assert post1_u2_v1.version == 1
        assert post1_u2_v1.title == "Post 1 User 2"
        assert post1_u2_v1.artifact_id != post1_u1_v1.artifact_id
        assert post1_u2_v1.artifact_id != post1_u1_v2.artifact_id
        assert post1_u2_v1.artifact_id != post1_u1_v3.artifact_id
        
    async with Context(user2).start_turn() as ctx:
        post = await user2.posts.last()
        post.content = "Content 1 User 2 Version 2"
        post1_u2_v2 = await post.save()
        assert post1_u2_v2.id != post1_u2_v1.id
        assert post1_u2_v2.version == 2
        assert post1_u2_v2.artifact_id == post1_u2_v1.artifact_id
        assert post1_u2_v2.content == "Content 1 User 2 Version 2"
        



@pytest.mark.asyncio
async def test_query_versioned_model(seeded_posts_database):
    user1, user2, posts_u1, posts_u2 = seeded_posts_database
    post2_u1_v3 = posts_u1[5]
    post1_u1_v3 = posts_u1[2]
    post1_u2_v2 = posts_u2[1]
    
    posts = await Post.query().tail(10)
    assert len(posts) == 3
    posts_u1 = await user1.posts.tail(10)
    assert len(posts_u1) == 2
    assert posts_u1[0].id == post2_u1_v3.id
    assert posts_u1[1].id == post1_u1_v3.id
    posts_u2 = await user2.posts.tail(10)
    assert len(posts_u2) == 1
    assert posts_u2[0].id == post1_u2_v2.id
    
    # check that we are not fetching old versions
    posts_u2_v1 = await Post.query().filter(lambda p: p.content == "Content 1 User 2 Version 1")
    assert len(posts_u2_v1) == 0
    posts_u2_v2 = await Post.query().filter(lambda p: p.content == "Content 1 User 2 Version 2")
    assert len(posts_u2_v2) == 1
    assert posts_u2_v2[0].id == post1_u2_v2.id


@pytest.mark.asyncio
async def test_get_artifact(seeded_posts_database):
    user1, user2, posts_u1, posts_u2 = seeded_posts_database
    post2_u1_v3 = posts_u1[5]
    post1_u1_v3 = posts_u1[2]
    post1_u2_v2 = posts_u2[1]
    
    post_art = await Post.get_artifact(post2_u1_v3.artifact_id)
    assert post_art.id == post2_u1_v3.id
    assert post_art.artifact_id == post2_u1_v3.artifact_id
    assert post_art.version == 3
    assert post_art.content == "Content 2 User 1 Version 3"
    
    post_art = await Post.get_artifact(post2_u1_v3.artifact_id, 2)
    assert post_art.id == post2_u1_v3.id
    assert post_art.artifact_id == post2_u1_v3.artifact_id
    assert post_art.version == 2
    assert post_art.content == "Content 2 User 1 Version 2"


@pytest.mark.asyncio
async def test_versioned_relations(seeded_user_database):
    user1, user2 = seeded_user_database
    
    async with Context(user1).start_turn() as ctx:
        post1_u1_v1 = await user1.posts.add(Post(title="Post 1 User 1", content="Content 1 User 1 Version 1"))
        
    async with Context(user2).start_turn() as ctx:
        like_p1_u2_v1 = await user2.likes.add(Like(type=LikeType.LIKE, post_id=post1_u1_v1.artifact_id))
        
        
    likes_p1 = await post1_u1_v1.likes.tail(10)
    assert len(likes_p1) == 1
    assert likes_p1[0].id == like_p1_u2_v1.id
    likes = await Like.query().tail(10)
    assert len(likes) == 1
    assert likes[0].id == like_p1_u2_v1.id

    async with Context(user1).start_turn() as ctx:
        post = await user1.posts.last()
        post.content = "Content 1 User 1 Version 2"
        post1_u1_v2 = await post.save()
        assert post1_u1_v2.version == 2


    likes_p1_v2 = await post1_u1_v2.likes.tail(10)
    assert len(likes_p1_v2) == 1
    assert likes_p1_v2[0].id == like_p1_u2_v1.id





