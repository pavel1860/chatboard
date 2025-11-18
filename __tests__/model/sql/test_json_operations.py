"""
Tests for SQL2 JSON operations: JsonBuildObject, JsonAgg, JsonbAgg
"""
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel, RelationField
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler
from promptview.model.sql2.expressions import JsonBuildObject, JsonAgg, JsonbAgg, Value
from typing import List





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
    likes: int = ModelField(default=0)
    comments: List[Comment] = RelationField(foreign_key="post_id")

@pytest_asyncio.fixture()
async def setup_db():
    """Initialize database with test models"""
    await NamespaceManager.initialize_clean()
    yield


class TestJsonBuildObject:
    """Test jsonb_build_object function"""

    def test_simple_json_build_object(self, setup_db):
        """Test simple JsonBuildObject with two fields"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonBuildObject({
                "id": posts_rel.get("id"),
                "title": posts_rel.get("title")
            }),
            alias="post_json"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_build_object(" in sql
        assert "'id', posts.id" in sql
        assert "'title', posts.title" in sql

    def test_json_build_object_with_literal(self, setup_db):
        """Test JsonBuildObject with literal values"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonBuildObject({
                "id": posts_rel.get("id"),
                "type": Value("post")
            }),
            alias="post_data"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_build_object(" in sql
        assert "'id', posts.id" in sql
        assert "'type', $1" in sql
        assert params == ["post"]

    def test_json_build_object_multiline(self, setup_db):
        """Test JsonBuildObject with multiple fields (multiline format)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonBuildObject({
                "id": posts_rel.get("id"),
                "title": posts_rel.get("title"),
                "text": posts_rel.get("text"),
                "likes": posts_rel.get("likes")
            }),
            alias="post_full"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_build_object(" in sql
        # Should be multiline format since more than 2 keys
        assert "\n" in sql


class TestJsonAgg:
    """Test json_agg function"""

    def test_simple_json_agg(self, setup_db):
        """Test simple JsonAgg"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonAgg(posts_rel.get("id")),
            alias="post_ids"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "json_agg(posts.id) AS post_ids" in sql

    def test_json_agg_with_object(self, setup_db):
        """Test JsonAgg with JsonBuildObject"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonAgg(
                JsonBuildObject({
                    "id": posts_rel.get("id"),
                    "title": posts_rel.get("title")
                })
            ),
            alias="posts"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "json_agg(" in sql
        assert "jsonb_build_object(" in sql
        assert "'id', posts.id" in sql
        assert "'title', posts.title" in sql


class TestJsonbAgg:
    """Test jsonb_agg function"""

    def test_simple_jsonb_agg(self, setup_db):
        """Test simple JsonbAgg"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonbAgg(posts_rel.get("title")),
            alias="titles"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_agg(posts.title) AS titles" in sql

    def test_jsonb_agg_with_object(self, setup_db):
        """Test JsonbAgg with JsonBuildObject"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonbAgg(
                JsonBuildObject({
                    "id": posts_rel.get("id"),
                    "title": posts_rel.get("title"),
                    "likes": posts_rel.get("likes")
                })
            ),
            alias="posts_data"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_agg(" in sql
        assert "jsonb_build_object(" in sql


class TestNestedJson:
    """Test nested JSON structures"""

    def test_nested_json_objects(self, setup_db):
        """Test nested JsonBuildObject"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            JsonBuildObject({
                "id": posts_rel.get("id"),
                "stats": JsonBuildObject({
                    "likes": posts_rel.get("likes"),
                    "title": posts_rel.get("title")
                })
            }),
            alias="post_with_stats"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "jsonb_build_object(" in sql
        # Should have nested structure
        assert sql.count("jsonb_build_object(") >= 2


class TestJsonWithJoins:
    """Test JSON operations with JOINs"""

    def test_json_agg_with_join(self, setup_db):
        """Test JsonAgg with JOIN and GROUP BY"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        query = SelectQuerySet(posts_rel)
        query.join(comments_rel, on=("id", "post_id"), join_type="LEFT")
        query.select("posts.id", "posts.title")
        query.select_expr(
            JsonAgg(
                JsonBuildObject({
                    "comment_id": comments_rel.get("id"),
                    "text": comments_rel.get("text")
                })
            ),
            alias="comments"
        )
        query.group_by("posts.id", "posts.title")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "LEFT JOIN comments" in sql
        assert "json_agg(" in sql
        assert "jsonb_build_object(" in sql
        assert "GROUP BY posts.id, posts.title" in sql
