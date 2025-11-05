"""
Tests for SQL2 raw SQL escape hatches: Raw expressions and RawRelation
"""
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel, RelationField
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation, RawRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler
from promptview.model.sql2.expressions import Raw, Value
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
    views: int = ModelField(default=0)
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


class TestRawExpressions:
    """Test Raw SQL expressions"""

    def test_raw_in_select(self, setup_db):
        """Test Raw expression in SELECT clause"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.id", "posts.title")
        query.select_expr(
            Raw("array_agg(DISTINCT posts.title ORDER BY posts.title)"),
            alias="titles"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "array_agg(DISTINCT posts.title ORDER BY posts.title) AS titles" in sql

    def test_raw_in_where(self, setup_db):
        """Test Raw expression in WHERE clause"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.*")
        query.where(Raw("posts.likes > 10 AND posts.views > 100"))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE posts.likes > 10 AND posts.views > 100" in sql

    def test_raw_with_parameters(self, setup_db):
        """Test Raw expression with parameterized values"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.*")
        query.where(
            Raw("posts.title ILIKE $1", Value("%python%"))
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "posts.title ILIKE $1" in sql
        assert params == ["%python%"]

    def test_raw_complex_expression(self, setup_db):
        """Test Raw expression with complex PostgreSQL features"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.id", "posts.title")
        query.where(
            Raw("to_tsvector('english', posts.title) @@ to_tsquery($1)",
                Value("python & programming"))
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "to_tsvector('english', posts.title) @@ to_tsquery($1)" in sql
        assert params == ["python & programming"]

    def test_raw_with_combined_conditions(self, setup_db):
        """Test Raw expression combined with regular conditions"""
        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.select("posts.*")
        query.where(
            (posts_rel.get("id") > 0) &
            Raw("posts.created_at > NOW() - INTERVAL '7 days'")
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "posts.id > $1" in sql
        assert "posts.created_at > NOW() - INTERVAL '7 days'" in sql
        assert "AND" in sql
        assert params == [0]


class TestRawRelation:
    """Test RawRelation for raw SQL subqueries"""

    def test_raw_relation_as_source(self, setup_db):
        """Test RawRelation as query source"""
        raw_posts = RawRelation(
            sql="SELECT * FROM posts WHERE likes > 10",
            name="popular_posts",
            namespace=Post.get_namespace()
        )

        query = SelectQuerySet(raw_posts)
        query.select("popular_posts.*")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM (\n    SELECT * FROM posts WHERE likes > 10\n) AS popular_posts" in sql

    def test_raw_relation_without_namespace(self, setup_db):
        """Test RawRelation with explicit field list"""
        user_stats = RawRelation(
            sql="""
                SELECT user_id, COUNT(*) as post_count, AVG(views) as avg_views
                FROM posts
                GROUP BY user_id
                HAVING COUNT(*) > 5
            """,
            name="user_stats",
            fields=["user_id", "post_count", "avg_views"]
        )

        posts_rel = NsRelation(Post.get_namespace())
        query = SelectQuerySet(posts_rel)
        query.join(user_stats, on=("user_id", "user_id"), alias="stats")
        query.select("posts.id", "posts.title", "stats.post_count")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM posts" in sql
        assert "INNER JOIN" in sql
        assert "SELECT user_id, COUNT(*) as post_count" in sql
        assert "AS stats" in sql

    def test_raw_relation_as_cte(self, setup_db):
        """Test RawRelation as a CTE"""
        complex_agg = RawRelation(
            sql="""
                SELECT user_id, COUNT(*) as post_count
                FROM posts
                WHERE likes > 100
                GROUP BY user_id
            """,
            name="top_users",
            fields=["user_id", "post_count"]
        )

        posts_rel = NsRelation(Post.get_namespace())
        main = SelectQuerySet(posts_rel)
        main.with_cte(complex_agg, alias="stats")
        main.join(complex_agg, on=("user_id", "user_id"), alias="user_stats")
        main.select("posts.*", "user_stats.post_count")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH stats AS" in sql
        assert "SELECT user_id, COUNT(*) as post_count" in sql
        assert "FROM posts" in sql
        assert "INNER JOIN stats AS user_stats" in sql
        # Should not duplicate the SQL
        assert sql.count("SELECT user_id, COUNT(*) as post_count") == 1

    def test_raw_relation_union(self, setup_db):
        """Test RawRelation with UNION"""
        union_query = RawRelation(
            sql="""
                SELECT id, title, 'published' as status
                FROM posts
                WHERE likes > 100

                UNION ALL

                SELECT id, title, 'draft' as status
                FROM posts
                WHERE likes <= 100
            """,
            name="all_posts",
            fields=["id", "title", "status"]
        )

        query = SelectQuerySet(union_query)
        query.select("all_posts.*")
        query.where(union_query.get("status") == "published")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "UNION ALL" in sql
        assert "WHERE all_posts.status = $1" in sql
        assert params == ["published"]


class TestRawRelationRecursiveCTE:
    """Test RawRelation for recursive CTEs"""

    def test_recursive_cte(self, setup_db):
        """Test RawRelation as recursive CTE (like branch hierarchy)"""
        branch_hierarchy = RawRelation(
            sql="""
                SELECT
                    id,
                    name,
                    parent_id,
                    0 AS depth
                FROM branches
                WHERE parent_id IS NULL

                UNION ALL

                SELECT
                    b.id,
                    b.name,
                    b.parent_id,
                    bh.depth + 1
                FROM branches b
                JOIN branch_hierarchy bh ON b.parent_id = bh.id
            """,
            name="branch_hierarchy",
            fields=["id", "name", "parent_id", "depth"]
        )

        query = SelectQuerySet(branch_hierarchy)
        query.with_cte(branch_hierarchy, alias="branch_hierarchy")
        query.select("branch_hierarchy.*")
        query.where(branch_hierarchy.get("depth") < 5)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WITH branch_hierarchy AS" in sql
        assert "UNION ALL" in sql
        assert "FROM branch_hierarchy" in sql
        assert "WHERE branch_hierarchy.depth < $1" in sql
        assert params == [5]


class TestCombinedRawAndNormal:
    """Test combining Raw expressions and RawRelation with normal query operations"""

    def test_raw_expression_and_relation(self, setup_db):
        """Test using both Raw expressions and RawRelation in one query"""
        filtered_posts = RawRelation(
            sql="SELECT * FROM posts WHERE jsonb_array_length(tags) > 3",
            name="tagged_posts",
            namespace=Post.get_namespace()
        )

        query = SelectQuerySet(filtered_posts)
        query.select("tagged_posts.id", "tagged_posts.title")
        query.select_expr(
            Raw("CASE WHEN tagged_posts.likes > 100 THEN 'popular' ELSE 'normal' END"),
            alias="popularity"
        )
        query.where(Raw("tagged_posts.views > 1000"))
        query.order_by("-tagged_posts.likes")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "FROM (\n    SELECT * FROM posts WHERE jsonb_array_length(tags) > 3\n)" in sql
        assert "CASE WHEN tagged_posts.likes > 100" in sql
        assert "WHERE tagged_posts.views > 1000" in sql
        assert "ORDER BY tagged_posts.likes DESC" in sql
