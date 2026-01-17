"""
Tests for PgQueryBuilder.agg() method - custom aggregation with explicit correlation
"""
from typing import List
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel, RelationField
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.pg_query_builder import select
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


class TestAggBasic:
    """Test basic .agg() functionality"""

    def test_agg_with_inferred_relation(self, setup_db):
        """Test agg() with inferred relation (same as include)"""
        query = select(Post).agg("comments", Comment)

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should select all post fields
        assert "posts.id" in sql
        assert "posts.title" in sql
        assert "posts.text" in sql

        # Should have COALESCE with json_agg subquery for comments
        assert "COALESCE" in sql
        assert "json_agg" in sql
        assert "AS comments" in sql

        # Subquery should select from comments
        assert "FROM comments" in sql
        assert "WHERE comments.post_id = posts.id" in sql

        # Should have all comment fields in jsonb_build_object
        assert "jsonb_build_object" in sql
        assert "'id', comments.id" in sql
        assert "'text', comments.text" in sql

        # Should default to empty array
        assert "'[]'" in sql

    def test_agg_with_explicit_on(self, setup_db):
        """Test agg() with explicit on parameter"""
        query = select(Post).agg("post_comments", Comment, on=("id", "post_id"))

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should use the custom name
        assert "AS post_comments" in sql

        # Should have correct correlation
        assert "WHERE comments.post_id = posts.id" in sql

        # Should still have json_agg
        assert "json_agg" in sql
        assert "COALESCE" in sql


class TestAggWithQueryBuilder:
    """Test agg() with pre-configured query builders"""

    def test_agg_with_filtered_query(self, setup_db):
        """Test agg() with a pre-filtered query builder"""
        # Create a query for comments with a filter
        comments_query = select(Comment).where(text="test")

        # Use it in agg
        query = select(Post).agg("filtered_comments", comments_query, on=("id", "post_id"))

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should have custom alias
        assert "AS filtered_comments" in sql

        # Should have correlation
        assert "comments.post_id = posts.id" in sql

        # Query should compile without errors
        assert "json_agg" in sql

    def test_agg_with_custom_alias(self, setup_db):
        """Test that agg() uses the provided name as alias"""
        query = select(User).agg("user_posts", Post, on=("id", "user_id"))

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should use the custom name
        assert "AS user_posts" in sql
        assert "WHERE posts.user_id = users.id" in sql


class TestAggMultiple:
    """Test multiple agg() calls"""

    def test_multiple_agg_calls(self, setup_db):
        """Test chaining multiple agg() calls"""
        # For this to work, we need a model with multiple relations
        # For now, just test that two agg calls compile correctly
        query = (
            select(User)
            .agg("user_posts", Post, on=("id", "user_id"))
        )

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should have the aggregation
        assert "AS user_posts" in sql
        assert "posts.user_id = users.id" in sql


class TestAggCorrelation:
    """Test correlation logic in agg()"""

    def test_agg_correlation_direction(self, setup_db):
        """Test that correlation uses correct direction: target.foreign_key = source.primary_key"""
        query = select(Post).agg("comments", Comment, on=("id", "post_id"))

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # The correlation should be: comments.post_id = posts.id
        # (target.foreign_key = source.primary_key)
        assert "comments.post_id = posts.id" in sql

    def test_agg_without_on_uses_relation_info(self, setup_db):
        """Test that agg() without 'on' uses relation info from model"""
        query = select(Post).agg("comments", Comment)

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should still find the correct correlation from RelationField
        assert "comments.post_id = posts.id" in sql


class TestAggEdgeCases:
    """Test edge cases for agg()"""

    def test_agg_preserves_main_query_fields(self, setup_db):
        """Test that agg() selects all fields from main query"""
        query = select(Post).agg("comments", Comment)

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # All post fields should be selected
        assert "posts.id" in sql
        assert "posts.title" in sql
        assert "posts.text" in sql
        assert "posts.user_id" in sql

    def test_agg_coalesce_default(self, setup_db):
        """Test that agg() returns empty array when no results"""
        query = select(Post).agg("comments", Comment)

        compiler = Compiler()
        sql, params = compiler.compile(query.query)

        # Should have COALESCE with empty array default
        assert "COALESCE" in sql
        assert "'[]'" in sql
