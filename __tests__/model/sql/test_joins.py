"""
Tests for SQL2 JOIN operations
"""
from typing import List
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel, RelationField
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler






class Comment(VersionedModel):
    """Test model for comments"""
    id: int = KeyField(primary_key=True)
    text: str = ModelField()
    post_id: int = ModelField(foreign_key=True, default=0)
    
    
class Post(VersionedModel):
    """Test model for posts"""
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    text: str = ModelField()
    user_id: int = ModelField(foreign_key=True, default=0)
    comments: List[Comment] = RelationField(foreign_key="post_id")


class User(VersionedModel):
    """Test model for users"""
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    email: str = ModelField()
    posts: List[Post] = RelationField(foreign_key="user_id")


@pytest_asyncio.fixture()
async def setup_db():
    """Initialize database with test models"""
    await NamespaceManager.initialize_clean()
    yield


class TestInnerJoin:
    """Test INNER JOIN operations"""

    def test_simple_inner_join(self, setup_db):
        """Test simple INNER JOIN between two tables"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"))
        query.select("posts.id", "posts.title", "comments.text")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM posts" in sql
        assert "INNER JOIN comments ON posts.id = comments.post_id" in sql
        assert "posts.id" in sql
        assert "comments.text" in sql

    def test_multiple_joins(self, setup_db):
        """Test multiple INNER JOINs"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())
        users_rel = NsRelation(User.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"))
        query.join(users_rel, on=("user_id", "id"))
        query.select("posts.title", "comments.text", "users.name")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM posts" in sql
        assert "INNER JOIN comments ON posts.id = comments.post_id" in sql
        assert "INNER JOIN users ON posts.user_id = users.id" in sql

    def test_join_with_alias(self, setup_db):
        """Test JOIN with table alias"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"), alias="c")
        query.select("posts.id", "c.text")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "INNER JOIN comments AS c ON posts.id = c.post_id" in sql
        assert "c.text" in sql


class TestLeftJoin:
    """Test LEFT JOIN operations"""

    def test_left_join(self, setup_db):
        """Test LEFT JOIN"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"), join_type="LEFT")
        query.select("posts.id", "posts.title", "comments.text")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM posts" in sql
        assert "LEFT JOIN comments ON posts.id = comments.post_id" in sql


class TestJoinWithWhere:
    """Test JOINs with WHERE clauses"""

    def test_join_with_where(self, setup_db):
        """Test JOIN with WHERE clause"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"))
        query.select("posts.id", "posts.title", "comments.text")
        query.where(posts_rel.get("id") == 1)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "INNER JOIN comments ON posts.id = comments.post_id" in sql
        assert "WHERE posts.id = $1" in sql
        assert params == [1]

    def test_join_with_complex_where(self, setup_db):
        """Test JOIN with complex WHERE conditions"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"))
        query.select("posts.*", "comments.*")
        query.where(
            (posts_rel.get("id") > 1) & (comments_rel.get("text").is_not_null())
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "INNER JOIN" in sql
        assert "WHERE" in sql
        assert "AND" in sql
        assert params == [1]


class TestJoinWithOrderBy:
    """Test JOINs with ORDER BY"""

    def test_join_with_order_by(self, setup_db):
        """Test JOIN with ORDER BY"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"))
        query.select("posts.title", "comments.text")
        query.order_by("-posts.id")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "INNER JOIN comments" in sql
        assert "ORDER BY posts.id DESC" in sql
