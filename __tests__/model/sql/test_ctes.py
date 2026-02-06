"""
Tests for SQL2 CTE (Common Table Expression) operations
"""
import pytest
import pytest_asyncio
from chatboard.model import KeyField, ModelField, VersionedModel, RelationField
from chatboard.model.namespace_manager2 import NamespaceManager
from chatboard.model.sql2.relations import NsRelation
from chatboard.model.sql2.relational_queries import SelectQuerySet
from chatboard.model.sql2.compiler import Compiler
from chatboard.model.sql2.expressions import JsonBuildObject, JsonAgg
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


class TestBasicCTE:
    """Test basic CTE operations"""

    def test_simple_cte(self, setup_db):
        """Test simple CTE with WHERE clause"""
        posts_rel = NsRelation(Post.get_namespace())

        # Create CTE
        filtered_posts = SelectQuerySet(posts_rel)
        filtered_posts.select("posts.id", "posts.title")
        filtered_posts.where(posts_rel.get("likes") > 10)

        # Main query using CTE
        main = SelectQuerySet(filtered_posts)
        main.with_cte(filtered_posts, alias="popular_posts")
        main.select("popular_posts.*")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH popular_posts AS" in sql
        assert "WHERE posts.likes > $1" in sql
        assert "FROM popular_posts" in sql
        assert params == [10]

    def test_cte_with_custom_alias(self, setup_db):
        """Test CTE with custom alias"""
        posts_rel = NsRelation(Post.get_namespace())

        filtered_posts = SelectQuerySet(posts_rel)
        filtered_posts.select("posts.*")
        filtered_posts.where(posts_rel.get("id") > 0)

        main = SelectQuerySet(filtered_posts)
        main.with_cte(filtered_posts, alias="my_custom_cte")
        main.select("my_custom_cte.*")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH my_custom_cte AS" in sql
        assert "FROM my_custom_cte" in sql


class TestCTEWithJoins:
    """Test CTEs with JOIN operations"""

    def test_cte_with_join(self, setup_db):
        """Test joining a CTE with another table"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        # Create CTE
        filtered_posts = SelectQuerySet(posts_rel)
        filtered_posts.select("posts.id", "posts.title")
        filtered_posts.where(posts_rel.get("likes") > 10)

        # Main query joins with CTE
        main = SelectQuerySet(comments_rel)
        main.with_cte(filtered_posts, alias="popular_posts")
        main.join(filtered_posts, on=("post_id", "id"), alias="post")
        main.select("comments.*", "post.title")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH popular_posts AS" in sql
        assert "FROM comments" in sql
        assert "INNER JOIN popular_posts AS post ON comments.post_id = post.id" in sql


class TestMultipleCTEs:
    """Test multiple CTEs in one query"""

    def test_multiple_ctes(self, setup_db):
        """Test query with multiple CTEs"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        # CTE 1: Popular posts
        popular_posts = SelectQuerySet(posts_rel)
        popular_posts.select("posts.id", "posts.title")
        popular_posts.where(posts_rel.get("likes") > 10)

        # CTE 2: Recent comments
        recent_comments = SelectQuerySet(comments_rel)
        recent_comments.select("comments.*")
        recent_comments.where(comments_rel.get("id") > 0)

        # Main query uses both CTEs
        main = SelectQuerySet(popular_posts)
        main.with_cte(popular_posts, alias="popular")
        main.with_cte(recent_comments, alias="recent")
        main.join(recent_comments, on=("id", "post_id"), alias="comment")
        main.select("popular.*", "comment.text")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH popular AS" in sql
        assert "recent AS" in sql
        assert sql.index("WITH popular") < sql.index("recent AS")


class TestNestedCTEs:
    """Test nested CTEs (should be flattened)"""

    def test_nested_ctes_flattened(self, setup_db):
        """Test that nested CTEs are flattened to top level"""
        posts_rel = NsRelation(Post.get_namespace())

        # Inner CTE
        inner_cte = SelectQuerySet(posts_rel)
        inner_cte.select("posts.*")
        inner_cte.where(posts_rel.get("likes") > 5)

        # Outer CTE that uses inner CTE
        outer_cte = SelectQuerySet(inner_cte)
        outer_cte.with_cte(inner_cte, alias="inner")
        outer_cte.select("inner.*")

        # Main query uses outer CTE
        main = SelectQuerySet(outer_cte)
        main.with_cte(outer_cte, alias="outer")
        main.select("outer.*")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        # Both CTEs should be at the top level (flattened)
        assert "WITH inner AS" in sql
        assert "outer AS" in sql
        # Inner should come before outer
        assert sql.index("WITH inner") < sql.index("outer AS")


class TestCTEWithSubquery:
    """Test CTEs with subquery field expressions"""

    def test_cte_with_subquery_field(self, setup_db):
        """Test CTE referenced in subquery field expression"""
        posts_rel = NsRelation(Post.get_namespace())
        comments_rel = NsRelation(Comment.get_namespace())

        # Create CTE for filtered comments
        filtered_comments = SelectQuerySet(comments_rel)
        filtered_comments.select("comments.id", "comments.text", "comments.post_id")
        filtered_comments.where(comments_rel.get("text").is_not_null())

        # Create subquery that selects FROM the CTE
        cte_subquery = SelectQuerySet(filtered_comments)
        cte_subquery.select_expr(
            JsonAgg(
                JsonBuildObject({
                    "id": filtered_comments.get("id"),
                    "text": filtered_comments.get("text")
                })
            ),
            alias="comments_data"
        )
        cte_subquery.where(filtered_comments.get("post_id") == posts_rel.get("id"))

        # Main query
        main = SelectQuerySet(posts_rel)
        main.with_cte(filtered_comments, alias="comments_cte")
        main.select("posts.id", "posts.title")
        main.select_subquery(cte_subquery, alias="comments")

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH comments_cte AS" in sql
        assert "FROM comments_cte" in sql
        # Should not inline the CTE SQL
        assert sql.count("comments.text IS NOT NULL") == 1


class TestCTEWithAggregations:
    """Test CTEs with aggregations"""

    def test_cte_with_count(self, setup_db):
        """Test CTE with COUNT aggregation"""
        posts_rel = NsRelation(Post.get_namespace())

        # CTE with aggregation
        post_stats = SelectQuerySet(posts_rel)
        post_stats.select("posts.user_id")
        from chatboard.model.sql2.expressions import Count
        post_stats.select_expr(Count(), alias="post_count")
        post_stats.group_by("posts.user_id")

        # Main query
        main = SelectQuerySet(post_stats)
        main.with_cte(post_stats, alias="stats")
        main.select("stats.*")
        main.where(main.get("post_count") > 5)

        compiler = Compiler()
        sql, params = compiler.compile(main)

        assert "WITH stats AS" in sql
        assert "COUNT(*) AS post_count" in sql
        assert "GROUP BY posts.user_id" in sql
        assert params == [5]
