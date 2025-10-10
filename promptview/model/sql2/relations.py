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
        return f"RelField({self.source.name}.{self.name})"

    def __str__(self):
        return f"{self.source.name}.{self.name}"

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

    def in_(self, values: list):
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
        # Check if base is a QuerySet (subquery)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For subqueries, yield a single field representing the subquery itself
            yield RelField(
                source=self.base,
                name=self.final_name,
                is_query=True
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
        # Check if base is a QuerySet (subquery)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For subqueries, the field_name should match the alias
            if field_name == self.final_name:
                return RelField(source=self.base, name=self.final_name, is_query=True)
            raise ValueError(f"Field {field_name} not found. Subquery only exposes '{self.final_name}'")
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
        # Check if base is a QuerySet (subquery)
        from .relational_queries import QuerySet
        is_subquery = isinstance(self.base, QuerySet)

        if is_subquery:
            # For subqueries, return the subquery field
            if field_name == self.final_name:
                return self.base, RelField(source=self.base, name=self.final_name, is_query=True)
            raise ValueError(f"Field {field_name} not found. Subquery only exposes '{self.final_name}'")
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


class JsonObjectRelation:
    """
    Transforms a source relation into a JSON object structure.
    Maps source fields to JSON keys using jsonb_build_object().
    """

    def __init__(self, source: RelationProtocol, field_mapping: dict[str, str], alias: str | None = None):
        """
        Args:
            source: The source relation to transform
            field_mapping: Map of JSON keys to source field names
                          e.g., {"comment_id": "id", "comment_text": "text"}
            alias: Optional alias for this relation
        """
        self.field_mapping = field_mapping
        self._alias = alias
        # Wrap source in Source if it isn't already
        if isinstance(source, Source):
            self._source = source
        else:
            self._source = Source(base=source)

    @property
    def name(self) -> str:
        return self._alias or f"{self._source.name}_json"

    @property
    def alias(self) -> str | None:
        return self._alias

    @property
    def final_name(self) -> str:
        return self._alias or self.name

    @property
    def sources(self) -> tuple[Source, ...]:
        """Return the wrapped source"""
        return (self._source,)

    def get(self, field_name: str) -> RelField:
        """Get a field - only the JSON object itself is available"""
        if field_name == self.final_name or field_name == "json_object":
            # Return a special field representing the JSON object
            return RelField(self, "json_object", is_query=False)
        raise ValueError(f"Field {field_name} not found. JsonObjectRelation only exposes 'json_object'")

    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over fields - yields the JSON object itself"""
        yield RelField(self, "json_object", is_query=False)

    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        """Resolve field reference"""
        return self, self.get(field_name)

    def __repr__(self):
        return f"JsonObjectRelation({self._source.name}, mapping={self.field_mapping})"


class Relation:
    """
    A relation composed of multiple sources with join information.
    Each source (except the first) contains join metadata.
    """

    def __init__(self, sources: list[Source], alias: str | None = None):
        if not sources:
            raise ValueError("Relation must have at least one source")
        self.sources = sources
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