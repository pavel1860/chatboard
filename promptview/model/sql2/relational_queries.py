from .relations import RelationProtocol, Relation, RelField, TypeRelationInput, Source, NsRelation, RawRelation
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
        right_rel: Right side of the join (usually NsRelation or QuerySet)
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
        left_sources = (left_rel,)
    else:
        # Left is a leaf relation (NsRelation, QuerySet, etc.)
        # Wrap it in a Source
        left_sources = (Source(base=left_rel),)

    # Create Source for the right side with join information
    # If right_rel is a QuerySet (including CTEs), use it directly as the base
    if isinstance(right_rel, QuerySet):
        base_relation = right_rel
    else:
        # Otherwise, extract the base from the resolved source
        base_relation = r_source.base if isinstance(r_source, Source) else r_source

    right_source = Source(
        base=base_relation,
        alias=alias,
        join_on=(l_field, r_field),
        join_type=join_type
    )

    # Combine and return new Relation
    return Relation(
        sources=list(left_sources + (right_source,)),
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
        self.order_by_fields: list[tuple[str, str]] = []  # List of (field, direction) tuples
        self.limit_value: int | None = None
        self.offset_value: int | None = None
        self.distinct_enabled: bool = False
        self.distinct_on_fields: list[str] = []  # Fields for DISTINCT ON (Postgres-specific)
        self.ctes = list[QuerySet]()
        self.recursive_cte = False
        self.scalar_expr: Expression | None = None  # For scalar subqueries (select_scalar)

    def join(self, target: RelationProtocol, on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
        """Add a join to this QuerySet (mutates in place)"""
        # Resolve fields
        l_source, l_field = self.get_source_and_field(on[0])
        r_source, r_field = target.get_source_and_field(on[1])

        # Create Source for the right side with join information
        # If target is a QuerySet (including CTEs), use it directly as the base
        if isinstance(target, QuerySet):
            base_relation = target
        else:
            # Otherwise, extract the base from the resolved source
            base_relation = r_source.base if isinstance(r_source, Source) else r_source

        right_source = Source(
            base=base_relation,
            alias=alias,
            join_on=(l_field, r_field),
            join_type=join_type
        )

        # Add to sources
        self.sources = self.sources + (right_source,)
        return self
    
    # projection
    def select(self, *fields: str):
        """
        Select fields by name.

        Examples:
            query.select('posts.id', 'posts.title')  # Select specific fields
            query.select('posts.*')  # Select all fields from posts
            query.select('posts.*', 'comments.*')  # Select all from multiple sources
        """
        if self.projection_fields is None:
            self.projection_fields = {}

        for f in fields:
            if f.endswith('.*'):
                # Wildcard selection: select all fields from this source
                source_name = f[:-2]  # Remove '.*'

                # Find the source
                target_source = None
                for source in self.sources:
                    if source.name == source_name or source.final_name == source_name:
                        target_source = source
                        break

                if not target_source:
                    raise ValueError(f"Source {source_name} not found")

                # Add all fields from this source
                for field in target_source.iter_fields():
                    qualified_name = f"{target_source.final_name}.{field.name}"
                    self.projection_fields[qualified_name] = {}
            else:
                # Regular field selection
                self.get(f)  # Validate field exists
                self.projection_fields[f] = {}

        return self

    def select_expr(self, expr: Expression, alias: str):
        """Select an expression with an alias (for aggregates, functions, subqueries, etc.)"""
        if self.projection_fields is None:
            self.projection_fields = {}
        self.projection_fields[alias] = {"expr": expr}
        return self

    def select_subquery(self, subquery: "QuerySet", alias: str):
        """
        Select a subquery with an alias.

        Use this to include subqueries (including CTEs) as fields.

        Example:
            # Create a CTE
            filtered_comments = SelectQuerySet(comments_rel).where(...)

            # Create subquery that selects from CTE
            cte_subquery = SelectQuerySet(filtered_comments)
            cte_subquery.select_scalar(JsonAgg(...))

            # Use it in main query
            main.with_cte(filtered_comments)
            main.select_subquery(cte_subquery, alias="comments")
        """
        if self.projection_fields is None:
            self.projection_fields = {}
        self.projection_fields[alias] = {"subquery": subquery}
        return self

    def select_scalar(self, expr: Expression):
        """
        Select a single scalar expression (for subqueries used as expressions).

        This marks the query as a scalar subquery that returns a single value without an alias.
        Use this when the query will be embedded in expressions like COALESCE, IN, etc.

        Example:
            comments_subquery.select_scalar(
                JsonAgg(JsonBuildObject({...}))
            )
        """
        self.scalar_expr = expr
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

    # ordering
    def order_by(self, *fields: str):
        """
        Add ORDER BY fields to the query.

        Use '-' prefix for descending order:
            query.order_by('posts.created_at')  # ASC
            query.order_by('-posts.created_at')  # DESC
            query.order_by('posts.title', '-posts.created_at')  # Multiple
        """
        for field in fields:
            if field.startswith('-'):
                # Descending order
                self.order_by_fields.append((field[1:], 'DESC'))
            else:
                # Ascending order (default)
                self.order_by_fields.append((field, 'ASC'))
        return self

    # pagination
    def limit(self, limit: int):
        """Set LIMIT for the query"""
        self.limit_value = limit
        return self

    def offset(self, offset: int):
        """Set OFFSET for the query"""
        self.offset_value = offset
        return self

    # distinct
    def distinct(self, *fields: str):
        """
        Add DISTINCT or DISTINCT ON to the query.

        Usage:
            query.distinct()  # SELECT DISTINCT
            query.distinct('posts.author_id')  # SELECT DISTINCT ON (posts.author_id)
            query.distinct('posts.author_id', 'posts.category')  # Multiple fields
        """
        if fields:
            # DISTINCT ON (Postgres-specific)
            self.distinct_on_fields = list(fields)
        else:
            # Simple DISTINCT
            self.distinct_enabled = True
        return self
    
    
    def include(self, target: "QuerySet", alias: str):
        """Include a subquery in SELECT clause as a field (scalar subquery)"""
        # Add the subquery to projection_fields
        if self.projection_fields is None:
            self.projection_fields = {}
        # Store the QuerySet directly - compiler will handle it
        self.projection_fields[alias] = {"subquery": target}
        return self
    
    def with_cte(self, cte: "QuerySet | RawRelation", alias: str | None = None, recursive: bool = False):
        """
        Add a CTE (Common Table Expression) to this query.

        If the CTE itself has nested CTEs, they will be flattened and merged into this query's CTE list,
        since SQL doesn't support nested WITH clauses.

        Args:
            cte: The QuerySet or RawRelation to use as a CTE
            alias: Optional custom name for the CTE. If not provided, uses cte.alias or generates a default name.
            recursive: If True, marks the CTE as recursive (for self-referencing CTEs)

        Example:
            main.with_cte(filtered_posts, alias="my_posts")
            main.with_cte(hierarchy, alias="tree", recursive=True)
        """
        if alias:
            # Set the alias on the CTE
            cte.alias = alias

        # Mark as recursive if specified
        if recursive and isinstance(cte, QuerySet):
            cte.recursive_cte = True
            # If this CTE is recursive, mark the parent query as having recursive CTEs
            self.recursive_cte = True

        # If the CTE is a QuerySet and has nested CTEs, merge them (flatten nested CTEs)
        # RawRelation doesn't have CTEs, so skip this check for them
        if isinstance(cte, QuerySet) and cte.ctes:
            for nested_cte in cte.ctes:
                # Avoid duplicates - only add if not already in our CTE list
                if nested_cte not in self.ctes:
                    self.ctes.append(nested_cte)
            # If any nested CTE is recursive, mark this query as recursive
            if cte.recursive_cte:
                self.recursive_cte = True
            # Clear the CTEs from the nested query since they're now at the top level
            cte.ctes = []

        # Add the CTE itself (avoid duplicates)
        if cte not in self.ctes:
            self.ctes.append(cte)

        return self
    
    def iter_projection_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        """Iterate over fields that were selected via .select(), .select_expr(), or .include()"""
        if self.projection_fields is None:
            # No projection specified, return all fields
            for field in self.iter_fields(include_sources):
                yield field
        else:
            # Iterate through projection_fields to maintain order
            for proj_name, proj_meta in self.projection_fields.items():
                # Check if this is an expression (from select_expr)
                if "expr" in proj_meta:
                    # Yield a RelField representing the expression
                    # The compiler will recognize this and compile the expression
                    expr = proj_meta["expr"]
                    yield RelField(source=expr, name=proj_name, is_query=False)
                # Check if this is a subquery (from include)
                elif "subquery" in proj_meta:
                    # Yield a RelField representing the subquery
                    subquery = proj_meta["subquery"]
                    yield RelField(source=subquery, name=proj_name, is_query=True)
                else:
                    # Regular field - find it in the sources
                    for field in self.iter_fields(include_sources):
                        qualified_name = f"{field.source.final_name}.{field.name}"
                        if field.name == proj_name or qualified_name == proj_name:
                            yield field
                            break

        
            
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.sources})"
    


class SelectQuerySet(QuerySet):
    pass
        
        
    
        
        
    
    


