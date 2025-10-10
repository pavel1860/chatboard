from .relations import RelationProtocol, Relation, RelField, TypeRelationInput, Source, NsRelation
from .expressions import Expression, WhereClause
from typing import Iterator






def join(
    left_rel: RelationProtocol,
    right_rel: RelationProtocol,
    on: tuple[str, str],
    join_type: str = "INNER",
    alias: str | None = None
) -> Relation:
    """
    Join two relations.

    Args:
        left_rel: Left side of the join (can be NsRelation, Relation, or Source)
        right_rel: Right side of the join (usually NsRelation)
        on: Tuple of (left_field, right_field) names
        join_type: Type of join (INNER, LEFT, RIGHT, FULL)
        alias: Optional alias for the right source

    Returns:
        A Relation with the new source added
    """
    # Resolve fields from both sides
    l_source, l_field = left_rel.get_source_and_field(on[0])
    r_source, r_field = right_rel.get_source_and_field(on[1])

    # Collect sources from the left side
    if isinstance(left_rel, Relation):
        # Left is already a Relation, extend its sources
        left_sources = left_rel.sources
    elif isinstance(left_rel, Source):
        # Left is a Source wrapper
        left_sources = [left_rel]
    else:
        # Left is a leaf relation (NsRelation, QuerySet, etc.)
        # Wrap it in a Source
        left_sources = [Source(base=left_rel)]

    # Create Source for the right side with join information
    # If r_source is already a Source, use its base
    base_relation = r_source.base if isinstance(r_source, Source) else r_source

    right_source = Source(
        base=base_relation,
        alias=alias,
        join_on=(l_field, r_field),
        join_type=join_type
    )

    # Combine and return new Relation
    return Relation(
        sources=left_sources + [right_source],
        alias=left_rel.alias if isinstance(left_rel, Relation) else None
    )
    







class QuerySet(Relation):
    def __init__(self, sources: TypeRelationInput, alias: str | None = None):
        # Convert TypeRelationInput to list of Sources
        if isinstance(sources, (list, tuple)):
            source_list = []
            for src in sources:
                if isinstance(src, Source):
                    source_list.append(src)
                else:
                    source_list.append(Source(base=src))
        elif isinstance(sources, Source):
            source_list = [sources]
        else:
            # Single RelationProtocol (NsRelation, etc.)
            source_list = [Source(base=sources)]

        super().__init__(source_list, alias=alias)
        self.projection_fields: dict[str, dict] | None = None
        self.where_clause = WhereClause()
        self.group_by_fields: list[str] = []
        self.ctes = list[QuerySet]()
        self.recursive_cte = False

    def join(self, target: RelationProtocol, on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
        """Add a join to this QuerySet (mutates in place)"""
        # Resolve fields
        l_source, l_field = self.get_source_and_field(on[0])
        r_source, r_field = target.get_source_and_field(on[1])

        # Create Source for the right side with join information
        base_relation = r_source.base if isinstance(r_source, Source) else r_source

        right_source = Source(
            base=base_relation,
            alias=alias,
            join_on=(l_field, r_field),
            join_type=join_type
        )

        # Add to sources
        self.sources = self.sources + [right_source]
        return self
    
    # projection
    def select(self, *fields: str):
        self.projection_fields = {}
        for f in fields:
            self.get(f)
            self.projection_fields[f] = {}
        return self

    # filtering
    def where(self, condition: Expression):
        """Add a WHERE condition to the query"""
        self.where_clause &= condition
        return self

    # grouping
    def group_by(self, *fields: str):
        """Add GROUP BY fields to the query"""
        self.group_by_fields.extend(fields)
        return self
    
    
    def include(self, target: "QuerySet", alias: str):
        """Include a subquery in SELECT clause as a field (scalar subquery)"""
        # Add the subquery to projection_fields
        if self.projection_fields is None:
            self.projection_fields = {}
        # Store the QuerySet directly - compiler will handle it
        self.projection_fields[alias] = {"subquery": target}
        return self
    
    def with_cte(self, cte: "QuerySet"):
        self.ctes.append(cte)
        return self
    
    def iter_projection_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over fields that were selected via .select() or .include()"""
        if self.projection_fields is None:
            # No projection specified, return all fields
            for field in self.iter_fields(include_sources):
                yield field
        else:
            # Iterate through projection_fields to maintain order
            for proj_name, proj_meta in self.projection_fields.items():
                # Check if this is a subquery
                if "subquery" in proj_meta:
                    # Yield a RelField representing the subquery
                    subquery = proj_meta["subquery"]
                    yield RelField(source=subquery, name=proj_name, is_query=True)
                else:
                    # Regular field - find it in the sources
                    for field in self.iter_fields(include_sources):
                        qualified_name = f"{field.source.name}.{field.name}"
                        if field.name == proj_name or qualified_name == proj_name:
                            yield field
                            break

        
            
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.sources})"
    


class SelectQuerySet(QuerySet):
    pass
        
        
    
        
        
    
    


