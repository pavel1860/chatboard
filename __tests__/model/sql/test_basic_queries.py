"""
Tests for basic SQL2 query operations: SELECT, WHERE, ORDER BY, LIMIT, OFFSET, DISTINCT
"""
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler
from promptview.model.sql2.expressions import Value


class Post(VersionedModel):
    """Test model for posts"""
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    text: str = ModelField()
    likes: int = ModelField(default=0)
    views: int = ModelField(default=0)
    status: str = ModelField(default="draft")


# Initialize database once for all tests
# @pytest.fixture(scope="module", autouse=True)
@pytest_asyncio.fixture()
async def setup_db():
    """Initialize database with test models"""
    await NamespaceManager.initialize_clean()
    yield
    # Cleanup if needed


class TestBasicSelect:
    """Test basic SELECT queries"""

    def test_select_all_fields(self, setup_db):
        """Test selecting all fields from a table"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "SELECT" in sql
        assert "posts.id" in sql
        assert "posts.title" in sql
        assert "posts.text" in sql
        assert "FROM posts" in sql
        assert params == []

    def test_select_specific_fields(self, setup_db):
        """Test selecting specific fields"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.id", "posts.title")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "posts.id" in sql
        assert "posts.title" in sql
        # Should not include other fields when explicitly selected
        assert sql.count("posts.") == 2

    def test_select_wildcard(self, setup_db):
        """Test selecting all fields with wildcard"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.*")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "posts.id" in sql
        assert "posts.title" in sql
        assert "posts.text" in sql


class TestWhereClause:
    """Test WHERE clause operations"""

    def test_where_equals(self, setup_db):
        """Test WHERE with equality"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("id") == 1)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.id = $1" in sql
        assert params == [1]

    def test_where_greater_than(self, setup_db):
        """Test WHERE with greater than"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("likes") > 10)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.likes > $1" in sql
        assert params == [10]

    def test_where_and(self, setup_db):
        """Test WHERE with AND condition"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(
            (posts_rel.get("likes") > 10) & (posts_rel.get("status") == "published")
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE" in sql
        assert "AND" in sql
        assert params == [10, "published"]

    def test_where_or(self, setup_db):
        """Test WHERE with OR condition"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(
            (posts_rel.get("likes") > 100) | (posts_rel.get("views") > 1000)
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE" in sql
        assert "OR" in sql
        assert params == [100, 1000]

    def test_where_is_null(self, setup_db):
        """Test WHERE with IS NULL"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("text").is_null())

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.text IS NULL" in sql

    def test_where_is_not_null(self, setup_db):
        """Test WHERE with IS NOT NULL"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("text").is_not_null())

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.text IS NOT NULL" in sql

    def test_where_in(self, setup_db):
        """Test WHERE with IN operator"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("status").isin(["published", "draft", "archived"]))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.status IN" in sql
        assert params == ["published", "draft", "archived"]

    def test_where_like(self, setup_db):
        """Test WHERE with LIKE operator"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.where(posts_rel.get("title").like("%python%"))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.title LIKE $1" in sql
        assert params == ["%python%"]


class TestOrderBy:
    """Test ORDER BY operations"""

    def test_order_by_asc(self, setup_db):
        """Test ORDER BY ascending"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.order_by("posts.id")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "ORDER BY posts.id ASC" in sql

    def test_order_by_desc(self, setup_db):
        """Test ORDER BY descending"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.order_by("-posts.likes")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "ORDER BY posts.likes DESC" in sql

    def test_order_by_multiple(self, setup_db):
        """Test ORDER BY multiple fields"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.order_by("posts.status", "-posts.likes")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "ORDER BY posts.status ASC, posts.likes DESC" in sql


class TestLimitOffset:
    """Test LIMIT and OFFSET operations"""

    def test_limit(self, setup_db):
        """Test LIMIT clause"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.limit(10)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "LIMIT 10" in sql

    def test_offset(self, setup_db):
        """Test OFFSET clause"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.offset(20)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "OFFSET 20" in sql

    def test_limit_and_offset(self, setup_db):
        """Test LIMIT and OFFSET together"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.limit(10)
        query.offset(20)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "LIMIT 10" in sql
        assert "OFFSET 20" in sql


class TestDistinct:
    """Test DISTINCT operations"""

    def test_distinct(self, setup_db):
        """Test simple DISTINCT"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.status")
        query.distinct()

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "SELECT DISTINCT" in sql

    def test_distinct_on(self, setup_db):
        """Test DISTINCT ON (Postgres specific)"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.*")
        query.distinct("posts.status")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "SELECT DISTINCT ON (posts.status)" in sql


class TestComplexQueries:
    """Test complex query combinations"""

    def test_complete_query(self, setup_db):
        """Test query with all clauses"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.id", "posts.title", "posts.likes")
        query.where(posts_rel.get("status") == "published")
        query.order_by("-posts.likes")
        query.limit(5)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "SELECT" in sql
        assert "posts.id" in sql
        assert "WHERE posts.status = $1" in sql
        assert "ORDER BY posts.likes DESC" in sql
        assert "LIMIT 5" in sql
        assert params == ["published"]
