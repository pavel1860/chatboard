"""
Tests for SQL2 LTREE operations (PostgreSQL hierarchical tree queries)
"""
import pytest
import pytest_asyncio
from promptview.model import KeyField, ModelField, VersionedModel
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model.sql2.relations import NsRelation
from promptview.model.sql2.relational_queries import SelectQuerySet
from promptview.model.sql2.compiler import Compiler
from promptview.model.sql2.expressions import LtreeNlevel, LtreeSubpath, LtreeLca, Value


class Category(VersionedModel):
    """Test model for categories with ltree path"""
    _namespace_name = "categories"
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    path: str = ModelField()  # ltree type in database


@pytest_asyncio.fixture()
async def setup_db():
    """Initialize database with test models"""
    await NamespaceManager.initialize_clean()
    yield


class TestLtreeOperators:
    """Test LTREE operators"""

    def test_ancestor_of(self, setup_db):
        """Test @> operator (ancestor of)"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.*")
        query.where(categories_rel.get("path").ancestor_of("Top.Science.Astronomy"))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "categories.path @> $1" in sql
        assert params == ["Top.Science.Astronomy"]

    def test_descendant_of(self, setup_db):
        """Test <@ operator (descendant of)"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.*")
        query.where(categories_rel.get("path").descendant_of("Top.Science"))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "categories.path <@ $1" in sql
        assert params == ["Top.Science"]

    def test_ltree_match(self, setup_db):
        """Test ~ operator (matches lquery pattern)"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.*")
        query.where(categories_rel.get("path").ltree_match("*.Science.*"))

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "categories.path ~ $1" in sql
        assert params == ["*.Science.*"]

    def test_ltree_concat(self, setup_db):
        """Test || operator (concatenate paths)"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select_expr(
            categories_rel.get("path").ltree_concat("NewLevel"),
            alias="extended_path"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "categories.path || $1 AS extended_path" in sql
        assert params == ["NewLevel"]


class TestLtreeFunctions:
    """Test LTREE functions"""

    def test_nlevel(self, setup_db):
        """Test nlevel() function - returns number of labels"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.path")
        query.select_expr(
            categories_rel.get("path").nlevel(),
            alias="depth"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "nlevel(categories.path) AS depth" in sql

    def test_nlevel_with_comparison(self, setup_db):
        """Test nlevel() with comparison in WHERE"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.*")
        query.where(categories_rel.get("path").nlevel() > 2)

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "WHERE nlevel(categories.path) > $1" in sql
        assert params == [2]

    def test_subpath_with_offset_and_length(self, setup_db):
        """Test subpath() with offset and length"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select_expr(
            categories_rel.get("path").subpath(0, 2),
            alias="parent_path"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "subpath(categories.path, 0, 2) AS parent_path" in sql

    def test_subpath_with_offset_only(self, setup_db):
        """Test subpath() with offset only"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select_expr(
            categories_rel.get("path").subpath(1),
            alias="subpath"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "subpath(categories.path, 1) AS subpath" in sql

    def test_lca(self, setup_db):
        """Test lca() function - lowest common ancestor"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)

        # Note: LCA typically takes multiple paths, this is a simplified test
        path1 = categories_rel.get("path")
        query.select_expr(
            LtreeLca(path1, path1),
            alias="common_ancestor"
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "lca(categories.path, categories.path) AS common_ancestor" in sql


class TestLtreeComplexQueries:
    """Test complex LTREE queries"""

    def test_ltree_with_multiple_conditions(self, setup_db  ):
        """Test LTREE with multiple conditions"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.*")
        query.where(
            categories_rel.get("path").descendant_of("Top") &
            (categories_rel.get("path").nlevel() == 3)
        )

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "categories.path <@ $1" in sql
        assert "nlevel(categories.path) = $2" in sql
        assert "AND" in sql
        assert params == ["Top", 3]

    def test_ltree_in_select_and_where(self, setup_db):
        """Test LTREE functions in both SELECT and WHERE"""
        categories_rel = NsRelation(Category.get_namespace())
        query = SelectQuerySet(categories_rel)
        query.select("categories.name", "categories.path")
        query.select_expr(
            categories_rel.get("path").nlevel(),
            alias="level"
        )
        query.select_expr(
            categories_rel.get("path").subpath(0, 1),
            alias="root"
        )
        query.where(categories_rel.get("path").nlevel() >= 2)
        query.order_by("level")

        compiler = Compiler()
        sql, params = compiler.compile(query)

        assert "nlevel(categories.path) AS level" in sql
        assert "subpath(categories.path, 0, 1) AS root" in sql
        assert "WHERE nlevel(categories.path) >= $1" in sql
        assert "ORDER BY level ASC" in sql
        assert params == [2]
