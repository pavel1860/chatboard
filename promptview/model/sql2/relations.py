from typing import Iterator, Protocol, TYPE_CHECKING
from ..base.base_namespace import BaseNamespace
from ..postgres2.pg_field_info import PgFieldInfo
if TYPE_CHECKING:
    from .relational_queries import QuerySet




class RelField:
    def __init__(self, source,  name: str, field_info: PgFieldInfo | None = None, alias: str | None = None, is_query: bool = False):
        self.name = name
        self.field_info = field_info
        self.alias = alias
        self.source = source
        self.is_query = is_query

    def __repr__(self):
        # Handle case where source is an Expression (from select_expr)
        if hasattr(self.source, 'name'):
            return f"RelField({self.source.name}.{self.name})"
        else:
            # Source is an expression, just show the field name
            return f"RelField({self.name}: {type(self.source).__name__})"

    def __str__(self):
        # Handle case where source is an Expression (from select_expr)
        if hasattr(self.source, 'name'):
            return f"{self.source.name}.{self.name}"
        else:
            # Source is an expression, just show the field name
            return f"{self.name}"

    # Comparison operators
    def __eq__(self, other):
        from .expressions import Eq
        return Eq(self, other)

    def __ne__(self, other):
        from .expressions import Neq
        return Neq(self, other)

    def __gt__(self, other):
        from .expressions import Gt
        return Gt(self, other)

    def __ge__(self, other):
        from .expressions import Gte
        return Gte(self, other)

    def __lt__(self, other):
        from .expressions import Lt
        return Lt(self, other)

    def __le__(self, other):
        from .expressions import Lte
        return Lte(self, other)

    # Helper methods for SQL operations
    def is_null(self):
        from .expressions import IsNull
        return IsNull(self)

    def is_not_null(self):
        from .expressions import IsNotNull
        return IsNotNull(self)

    def isin(self, values: list):
        from .expressions import In
        return In(self, values)

    def not_in(self, values: list):
        from .expressions import NotIn
        return NotIn(self, values)

    def between(self, lower, upper):
        from .expressions import Between
        return Between(self, lower, upper)

    def like(self, pattern: str):
        from .expressions import Like
        return Like(self, pattern)

    def ilike(self, pattern: str):
        from .expressions import ILike
        return ILike(self, pattern)

    # LTREE methods (PostgreSQL ltree extension)

    def ancestor_of(self, subpath):
        """
        Check if this ltree path is an ancestor of subpath.
        Uses LTREE @> operator.

        Example: categories.get("path").ancestor_of("Top.Science.Astronomy")
        SQL: path @> 'Top.Science.Astronomy'
        """
        from .expressions import LtreeAncestor
        return LtreeAncestor(self, subpath)

    def descendant_of(self, path):
        """
        Check if this ltree path is a descendant of path.
        Uses LTREE <@ operator.

        Example: categories.get("path").descendant_of("Top.Science")
        SQL: path <@ 'Top.Science'
        """
        from .expressions import LtreeDescendant
        return LtreeDescendant(self, path)

    def ltree_match(self, pattern):
        """
        Check if this ltree path matches lquery pattern.
        Uses LTREE ~ operator.

        Example: categories.get("path").ltree_match("*.Science.*")
        SQL: path ~ '*.Science.*'
        """
        from .expressions import LtreeMatch
        return LtreeMatch(self, pattern)

    def ltree_concat(self, other):
        """
        Concatenate this ltree path with another.
        Uses LTREE || operator.

        Example: categories.get("path").ltree_concat("NewLevel")
        SQL: path || 'NewLevel'
        """
        from .expressions import LtreeConcat
        return LtreeConcat(self, other)

    def nlevel(self):
        """
        Get the number of levels in this ltree path.
        Uses nlevel() function.

        Example: categories.get("path").nlevel()
        SQL: nlevel(path)
        """
        from .expressions import LtreeNlevel
        return LtreeNlevel(self)

    def subpath(self, offset: int, length: int | None = None):
        """
        Extract a subpath from this ltree path.
        Uses subpath() function.

        Example: categories.get("path").subpath(0, 2)
        SQL: subpath(path, 0, 2)
        """
        from .expressions import LtreeSubpath
        return LtreeSubpath(self, offset, length)



class RelationProtocol(Protocol):
    # alias: str | None
    # name: str
    
    @property
    def sources(self) -> "tuple[RelationProtocol, ...]":
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def alias(self) -> str | None:
        ...
        
    @property
    def final_name(self) -> str:
        ...
        
    def get(self, field_name: str) -> RelField:
        ...
        
    def iter_fields(self, include_sources: set[str] | None = None) -> "Iterator[RelField]":
        ...
        
        
    def get_source_and_field(self, field_name: str) -> tuple["RelationProtocol", RelField]:
        ...
    
        


class NsRelation:
    
    def __init__(self, namespace: BaseNamespace, alias: str | None = None):
        self.namespace = namespace
        self.alias = alias
        
    @property
    def sources(self) -> tuple[RelationProtocol, ...]:
        return (self,)
        
    
    @property
    def name(self) -> str:
        return self.namespace.name
    
    @property
    def final_name(self) -> str:
        return self.alias or self.name
    
    def get(self, field_name: str) -> RelField:
        field_info = self.namespace.get_field(field_name)
        return RelField(self, field_info.name, field_info)
        
    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        if include_sources is not None and self.namespace.name not in include_sources:
            return
        for field_info in self.namespace.iter_fields():
            yield RelField(self, field_info.name, field_info)
        
        
    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        return self, self.get(field_name)
    
    
    def print(self):
        return f"NS({self.name})"


class RawRelation:
    """
    Raw SQL relation escape hatch.

    Use this when you need to use raw SQL as a source, CTE, or subquery.
    Useful for complex SQL that the query builder doesn't support yet.

    Examples:
        # With namespace (gets field metadata automatically)
        complex_posts = RawRelation(
            sql='''
                SELECT * FROM posts
                WHERE jsonb_array_length(tags) > 5
            ''',
            name="tagged_posts",
            namespace=Post.get_namespace()
        )

        # Without namespace (explicit field list)
        user_stats = RawRelation(
            sql='''
                SELECT user_id, COUNT(*) as post_count
                FROM posts
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY user_id
            ''',
            name="user_stats",
            fields=["user_id", "post_count"]
        )
    """

    def __init__(
        self,
        sql: str,
        name: str,
        namespace: BaseNamespace | None = None,
        fields: list[str] | None = None,
        alias: str | None = None
    ):
        """
        Create a raw SQL relation.

        Args:
            sql: Raw SQL SELECT query (without wrapping parentheses)
            name: Name for this relation (used in generated SQL)
            namespace: Optional namespace to get field metadata from
            fields: List of field names this query exposes (if no namespace provided)
            alias: Optional alias for the relation
        """
        self.sql = sql
        self._name = name
        self.namespace = namespace
        self.field_names = fields or []
        self.alias = alias

    @property
    def sources(self) -> tuple[RelationProtocol, ...]:
        return (self,)

    @property
    def name(self) -> str:
        return self._name

    @property
    def final_name(self) -> str:
        return self.alias or self._name

    def get(self, field_name: str) -> RelField:
        """Get a field by name."""
        # If we have a namespace, get field info from it
        if self.namespace:
            field_info = self.namespace.get_field(field_name)
            return RelField(self, field_info.name, field_info)
        else:
            # No namespace, use explicit field list or allow any field
            if self.field_names and field_name not in self.field_names:
                raise ValueError(f"Field {field_name} not in declared fields: {self.field_names}")
            return RelField(self, field_name, field_info=None)

    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over fields."""
        if include_sources is not None and self._name not in include_sources:
            return

        # If we have a namespace, iterate its fields
        if self.namespace:
            for field_info in self.namespace.iter_fields():
                yield RelField(self, field_info.name, field_info)
        elif self.field_names:
            # Use explicit field list
            for field_name in self.field_names:
                yield RelField(self, field_name, field_info=None)
        # If neither namespace nor fields, can't iterate (must use * or explicit get)

    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        return self, self.get(field_name)

    def print(self):
        return f"RAW({self.name})"


class Source:
    """
    Wraps a relation (table/subquery) with optional join metadata.
    Acts as a transparent proxy to the underlying relation.
    """

    def __init__(
        self,
        base: RelationProtocol,
        alias: str | None = None,
        join_on: tuple[RelField, RelField] | None = None,
        join_type: str = "INNER"
    ):
        self.base = base
        self._alias = alias
        self.join_on = join_on  # (left_field, right_field)
        self.join_type = join_type

    @property
    def name(self) -> str:
        """Get the base name of the wrapped relation"""
        return self.base.name

    @property
    def alias(self) -> str | None:
        """Get the alias for this source"""
        return self._alias

    @property
    def final_name(self) -> str:
        """Get the effective name (alias if set, otherwise base name)"""
        return self._alias or self.base.final_name

    @property
    def sources(self) -> tuple[RelationProtocol, ...]:
        """Return the base's leaf sources (for compatibility with RelationProtocol)"""
        return self.base.sources if hasattr(self.base, 'sources') else (self.base,)

    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over fields, wrapping them to reference this Source"""
        # Check if base is a QuerySet (subquery/CTE)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For QuerySets (CTEs/subqueries), iterate over their projected fields
            for field in self.base.iter_projection_fields(include_sources):
                yield RelField(
                    source=self,
                    name=field.name,
                    field_info=field.field_info,
                    alias=field.alias,
                    is_query=field.is_query
                )
        else:
            # For regular relations, proxy to base fields
            for field in self.base.iter_fields(include_sources):
                yield RelField(
                    source=self,
                    name=field.name,
                    field_info=field.field_info,
                    alias=field.alias,
                    is_query=field.is_query
                )

    def get(self, field_name: str) -> RelField:
        """Get a field by name, wrapping it to reference this Source"""
        # Check if base is a QuerySet (subquery/CTE)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For QuerySets, find the field in projected fields
            for field in self.base.iter_projection_fields():
                if field.name == field_name:
                    return RelField(
                        source=self,
                        name=field.name,
                        field_info=field.field_info,
                        alias=field.alias,
                        is_query=field.is_query
                    )
            raise ValueError(f"Field {field_name} not found in QuerySet projection")
        else:
            # For regular relations, proxy to base
            field = self.base.get(field_name)
            return RelField(
                source=self,
                name=field.name,
                field_info=field.field_info,
                alias=field.alias,
                is_query=getattr(field, 'is_query', False)
            )

    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        """Resolve field and return self as the source"""
        # Check if base is a QuerySet (subquery/CTE)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For QuerySets, find the field in projected fields
            for field in self.base.iter_projection_fields():
                if field.name == field_name:
                    return self, RelField(
                        source=self,
                        name=field.name,
                        field_info=field.field_info,
                        alias=field.alias,
                        is_query=field.is_query
                    )
            raise ValueError(f"Field {field_name} not found in QuerySet projection")
        else:
            # For regular relations, proxy to base
            _, field = self.base.get_source_and_field(field_name)
            return self, RelField(
                source=self,
                name=field.name,
                field_info=field.field_info,
                alias=field.alias,
                is_query=getattr(field, 'is_query', False)
            )

    def get_on_clause(self) -> str:
        """Generate SQL ON clause for this join"""
        if not self.join_on:
            raise ValueError(f"Source {self.name} has no join information")
        left_field, right_field = self.join_on
        return f"{left_field.source.final_name}.{left_field.name} = {self.final_name}.{right_field.name}"

    def __repr__(self):
        join_info = f", join_on={self.join_on}" if self.join_on else ""
        return f"Source({self.base.name}, alias={self._alias}{join_info})"


TypeRelationInput = tuple[RelationProtocol, ...] | RelationProtocol


class Relation:
    """
    A relation composed of multiple sources with join information.
    Each source (except the first) contains join metadata.
    """

    sources: tuple[Source, ...]  # Type annotation for the attribute

    def __init__(self, sources: list[Source], alias: str | None = None):
        if not sources:
            raise ValueError("Relation must have at least one source")
        self.sources = tuple(sources)  # Store as tuple
        self.alias = alias

    @property
    def name(self) -> str:
        """Generate name from all source names"""
        if len(self.sources) == 1:
            return self.sources[0].name
        return "_".join([s.name for s in self.sources])

    @property
    def final_name(self) -> str:
        """Get the effective name (alias if set, otherwise generated name)"""
        return self.alias or self.name

    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        """Find which source contains this field"""
        # Parse "source.field" or just "field"
        parts = field_name.split(".")

        if len(parts) == 1:
            # Try first source (default)
            return self.sources[0].get_source_and_field(field_name)
        elif len(parts) == 2:
            # Find the source by name
            source_name, field = parts
            for source in self.sources:
                if source.name == source_name or source.final_name == source_name:
                    return source.get_source_and_field(field)
            raise ValueError(f"Source {source_name} not found in {self.name}")
        else:
            raise ValueError(f"Invalid field name: {field_name} on {self.name}")

    def get(self, field_name: str) -> RelField:
        """Get a field by name"""
        _, field = self.get_source_and_field(field_name)
        return field

    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over all fields from all sources"""
        for source in self.sources:
            if include_sources is not None and source.name not in include_sources:
                continue
            for field in source.iter_fields(include_sources):
                yield field

    def print(self):
        """Print a simple representation"""
        return f"REL({self.name})"

    def print_tree(self, indent: int = 0):
        """Print a tree representation of the relation structure"""
        print(" " * indent + "REL", self.name)
        indent += 1
        for source in self.sources:
            join_info = f" [{source.join_type} JOIN]" if source.join_on else ""
            print(" " * indent + f"SOURCE: {source.name} (alias={source.alias}){join_info}")

    def __repr__(self):
        sources_str = ", ".join([repr(s) for s in self.sources])
        return f"Relation([{sources_str}])"