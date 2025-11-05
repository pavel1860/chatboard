"""
Tests for SQL2 aggregation operations: COUNT, SUM, AVG, MIN, MAX, GROUP BY
"""
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel, RelationField
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler
from promptview.model.sql2.expressions import Count, Sum, Avg, Min, Max, Coalesce, Value
from typing import List






class Post(VersionedModel):
    """Test model for posts"""
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    text: str = ModelField()
    likes: int = ModelField(default=0)
    views: int = ModelField(default=0)
    user_id: int = ModelField(foreign_key=True, default=0)
    
    
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

class TestCountAggregation:
    """Test COUNT aggregation"""

    def test_count_star(self, setup_db):
        """Test COUNT(*)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Count(), alias="total_count")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COUNT(*) AS total_count" in sql

    def test_count_field(self, setup_db):
        """Test COUNT(field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Count(posts_rel.get("id")), alias="post_count")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COUNT(posts.id) AS post_count" in sql

    def test_count_distinct(self, setup_db):
        """Test COUNT(DISTINCT field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Count(posts_rel.get("user_id"), distinct=True), alias="unique_users")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COUNT(DISTINCT posts.user_id) AS unique_users" in sql


class TestSumAggregation:
    """Test SUM aggregation"""

    def test_sum(self, setup_db):
        """Test SUM(field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Sum(posts_rel.get("likes")), alias="total_likes")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "SUM(posts.likes) AS total_likes" in sql


class TestAvgAggregation:
    """Test AVG aggregation"""

    def test_avg(self, setup_db):
        """Test AVG(field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Avg(posts_rel.get("views")), alias="avg_views")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "AVG(posts.views) AS avg_views" in sql


class TestMinMaxAggregation:
    """Test MIN and MAX aggregations"""

    def test_min(self, setup_db):
        """Test MIN(field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Min(posts_rel.get("likes")), alias="min_likes")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "MIN(posts.likes) AS min_likes" in sql

    def test_max(self, setup_db):
        """Test MAX(field)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(Max(posts_rel.get("views")), alias="max_views")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "MAX(posts.views) AS max_views" in sql


class TestGroupBy:
    """Test GROUP BY operations"""

    def test_group_by_single_field(self, setup_db):
        """Test GROUP BY single field"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.user_id")
        query.select_expr(Count(), alias="post_count")
        query.group_by("posts.user_id")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "posts.user_id" in sql
        assert "COUNT(*) AS post_count" in sql
        assert "GROUP BY posts.user_id" in sql

    def test_group_by_multiple_fields(self, setup_db):
        """Test GROUP BY multiple fields"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.user_id", "posts.title")
        query.select_expr(Sum(posts_rel.get("likes")), alias="total_likes")
        query.group_by("posts.user_id", "posts.title")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "GROUP BY posts.user_id, posts.title" in sql

    def test_group_by_with_multiple_aggregates(self, setup_db):
        """Test GROUP BY with multiple aggregates"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.user_id")
        query.select_expr(Count(), alias="post_count")
        query.select_expr(Sum(posts_rel.get("likes")), alias="total_likes")
        query.select_expr(Avg(posts_rel.get("views")), alias="avg_views")
        query.group_by("posts.user_id")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COUNT(*) AS post_count" in sql
        assert "SUM(posts.likes) AS total_likes" in sql
        assert "AVG(posts.views) AS avg_views" in sql
        assert "GROUP BY posts.user_id" in sql


class TestCoalesce:
    """Test COALESCE function"""

    def test_coalesce_with_values(self, setup_db):
        """Test COALESCE with multiple values"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            Coalesce(posts_rel.get("text"), Value("No content")),
            alias="content"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COALESCE(posts.text, $1) AS content" in sql
        assert params == ["No content"]

    def test_coalesce_with_multiple_fields(self, setup_db):
        """Test COALESCE with multiple fields"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select_expr(
            Coalesce(posts_rel.get("likes"), posts_rel.get("views"), Value(0)),
            alias="engagement"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "COALESCE(posts.likes, posts.views, $1) AS engagement" in sql
        assert params == [0]


class TestComplexAggregations:
    """Test complex aggregation scenarios"""

    def test_aggregation_with_where(self, setup_db):
        """Test aggregation with WHERE clause"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.user_id")
        query.select_expr(Count(), alias="published_count")
        query.where(posts_rel.get("title").is_not_null())
        query.group_by("posts.user_id")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.title IS NOT NULL" in sql
        assert "COUNT(*) AS published_count" in sql
        assert "GROUP BY posts.user_id" in sql

    def test_aggregation_with_order_by(self):
        """Test aggregation with ORDER BY on aggregate"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.user_id")
        query.select_expr(Count(), alias="post_count")
        query.group_by("posts.user_id")
        query.order_by("-post_count")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "GROUP BY posts.user_id" in sql
        assert "ORDER BY post_count DESC" in sql
