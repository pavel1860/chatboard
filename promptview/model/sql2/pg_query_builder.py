from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generator, Generic, Literal, Type, TypeVar
from .relations import RawRelation, RelationProtocol, Source, NsRelation, RelField, Relation
from .relational_queries import join, SelectQuerySet, QuerySet, Relation, NsRelation, Expression
from .expressions import Coalesce, JsonBuildObject, JsonAgg, Value
from .compiler import Compiler
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ..model3 import Model
    from ..relation_info import RelationInfo
    from ..base.base_namespace import BaseNamespace


Ts = TypeVar("Ts", bound="Model")

RetModel = TypeVar("RetModel")


def json_agg(query: "SelectQuerySet", alias: str):
    from .expressions import Expression

    # If query has projection fields (e.g., from nested includes), use those
    # Otherwise use base table fields
    if query.projection_fields is not None:
        fields = {}
        for field in query.iter_projection_fields():
            # If the field's source is an Expression (e.g., from nested include),
            # use the expression directly. Otherwise use the field.
            if isinstance(field.source, Expression):
                fields[field.name] = field.source
            else:
                fields[field.name] = field
    else:
        fields = {field.name: field for field in query.iter_fields()}

    json_obj = JsonBuildObject(fields)
    j_agg = JsonAgg(json_obj)
    query.select_scalar(j_agg)
    return query


def json_object(query: "SelectQuerySet", alias: str):
    """Return a single JSON object (for one-to-one relations)."""
    from .expressions import Expression

    # If query has projection fields (e.g., from nested includes), use those
    # Otherwise use base table fields
    if query.projection_fields is not None:
        fields = {}
        for field in query.iter_projection_fields():
            # If the field's source is an Expression (e.g., from nested include),
            # use the expression directly. Otherwise use the field.
            if isinstance(field.source, Expression):
                fields[field.name] = field.source
            else:
                fields[field.name] = field
    else:
        fields = {field.name: field for field in query.iter_fields()}

    json_obj = JsonBuildObject(fields)
    query.select_scalar(json_obj)
    return query

T_co = TypeVar("T_co", covariant=True)


class QuerySetSingleAdapter(Generic[T_co]):
    def __init__(self, queryset: "PgQueryBuilder[T_co]", parse: bool = True):
        self.queryset = queryset

    def __await__(self) -> Generator[Any, None, T_co]:
        async def await_query():
            results = await self.queryset.execute()
            if results:
                return results[0]
            return None
            # raise ValueError("No results found")
            # return None
            # raise DoesNotExist(self.queryset.model)
        return await_query().__await__()  



class PgQueryBuilder(Generic[Ts]):
    def __init__(self):
        self._query = None
        self._return_json = False  # Flag to return dicts instead of Model instances
        self._parse = None
        self._parse_rows = None
        self._parse_models: Callable[[list[Ts]], Awaitable[list[Ts]]] | None = None


    @property
    def query(self):
        if self._query is None:
            raise ValueError("Query is not set")
        return self._query
    
    
    def alias(self, alias: str):
        self.query.alias = alias
        return self
        
    def select(self, *targets: Type[Ts]):
        from ..model3 import Model
        ns_lookup = {}
        for target in targets:
            if issubclass(target, Model):
                ns = target.get_namespace()
                ns_source = NsRelation(ns)
                # sources.append(ns_source)
                ns_lookup[ns.name] = ns_source
            else:
                raise ValueError(f"Invalid target: {target}")
        for ns_name, source in ns_lookup.items():
            if self._query is None:
                self._query = SelectQuerySet(source)
            else:
                self.query.join(source, on=(source.primary_key, source.foreign_key))
            self.query.select(f"{source.final_name}.*")
            
        return self
    
    
    def raw(self, sql: str, name: str, namespace: "BaseNamespace"):
        if not self._query is None:
            raise ValueError("Query is already set")
        raw_rel = RawRelation(sql, name, namespace)
        self._query = SelectQuerySet(raw_rel)
        return self
    
    
    def parse(self, func: Callable[[Ts], Any], target: Literal["rows", "models"] = "models"):
        if target == "rows":
            self._parse_rows = func
        elif target == "models":
            self._parse_models = func
        else:
            raise ValueError(f"Invalid target: {target}")
        return self
    
    
    def where(self, condition: Expression | None = None, **kwargs):
        """
        Add WHERE conditions to the query.

        Args:
            condition: An Expression object (e.g., posts_rel.get("id") == 1)
            **kwargs: Field=value pairs for equality conditions (e.g., id=1, status="published")

        Usage:
            .where(posts_rel.get("id") > 5)  # Expression
            .where(id=1)                      # Keyword argument
            .where(id=1, status="active")     # Multiple keyword arguments
            .where(posts_rel.get("id") > 5, status="active")  # Both combined
        """
        from .expressions import Eq, And

        if kwargs:
            # Need to resolve field names to RelField objects
            if not self.query.sources:
                raise ValueError("Query has no sources to resolve fields from")

            for field_name, value in kwargs.items():
                # Get the RelField object from the query
                field = self.query.get(field_name)

                # Create equality expression with the RelField
                if condition is None:
                    condition = Eq(field, value)
                else:
                    condition = condition & Eq(field, value)

        if condition is None:
            raise ValueError("Condition is not set")

        self.query.where(condition)
        return self
    
    def limit(self, n: int):
        self.query.limit(n)
        return self
    
    def offset(self, n: int):
        self.query.offset(n)
        return self
    
    def order_by(self, *fields: str):
        self.query.order_by(*fields)
        return self

    # def json(self) -> "PgQueryBuilder[Ts]":
    #     """
    #     Configure the query to return dicts instead of Model instances.

    #     Usage:
    #         posts = await select(Post).include(Comment).json()
    #         # posts[0] is a dict
    #     """
    #     self._return_json = True
    #     return self

    def use_cte(self, cte: "PgQueryBuilder[Ts]", cte_name: str | None = None, recursive: bool = False) -> "PgQueryBuilder[Ts]":
        """
        Add a CTE (Common Table Expression) to this query.

        Args:
            cte: The query builder to use as a CTE
            cte_name: Optional custom name for the CTE
            recursive: If True, marks the CTE as recursive (for self-referencing CTEs)

        Usage:
            # Regular CTE
            popular_posts = select(Post).where(views__gt=1000)
            results = await select(Comment).use_cte(popular_posts, cte_name="popular")

            # Recursive CTE (for hierarchies, graphs, etc.)
            hierarchy = PgQueryBuilder().raw(sql="...", name="tree", namespace=Node.get_namespace())
            results = await select(Node).use_cte(hierarchy, cte_name="tree", recursive=True)
        """
        self.query.with_cte(cte.query, alias=cte_name, recursive=recursive)
        return self
    
    def join_cte(self, cte: "PgQueryBuilder[Model]", cte_name: str | None = None, on: tuple[str, str] | None = None, alias: str | None = None, recursive: bool = False):
        """
        Add a CTE and immediately join it to this query.

        Args:
            cte: The query builder to use as a CTE
            cte_name: Optional custom name for the CTE
            on: Join condition as (left_field, right_field) tuple
            alias: Optional alias for the joined CTE (different from cte_name)
            recursive: If True, marks the CTE as recursive
        """
        self.use_cte(cte, cte_name=cte_name, recursive=recursive)
        self.join(cte, on=on, alias=alias)
        return self
    
    def join(self, target: "PgQueryBuilder[Model]", on: tuple[str, str] | None = None, alias: str | None = None):
        if on is None:
            rel, source = self._infer_relation(target)
            if rel is None:
                raise ValueError(f"No relation found for {target}")
            on = (rel.primary_key, rel.foreign_key)
        self.query.join(target.query, on=on, alias=alias)
        return self
    
        
    def _infer_relation(self, target: "Type[Model] | PgQueryBuilder[Model]"):
        """
        Infer the relation to the target model from the query sources.
        Returns a tuple of (relation, source) where source is the Source object that has the relation.
        """
        if isinstance(target, PgQueryBuilder):
            rel = None
            for source in target.query.sources:
                ns = source.base.namespace
                if ns is None:
                    raise ValueError("Namespace is not set")
                for src in self.query.sources:
                    if src.base.namespace is None:
                        continue
                    rel = src.base.namespace.get_relation_for_namespace(ns)
                    if rel is not None:
                        return (rel, src)
        else:
            ns = target.get_namespace()
            for source in self.query.sources:
                rel = source.base.namespace.get_relation_for_namespace(ns)
                if rel is not None:
                    return (rel, source)
        return (None, None)
    
    def distinct_on(self, *fields: str):
        self.query.distinct(*fields)
        return self

    
    def one(self) -> "QuerySetSingleAdapter[Ts]":
        self.query.limit(1)
        return QuerySetSingleAdapter[Ts](self)
            
    def first(self) -> "QuerySetSingleAdapter[Ts]":
        """
        Get the first record based on the default_order_field.
        Orders by default_order_field in ascending order and limits to 1.
        """
        # Get the namespace from the first source
        if not self.query.sources:
            raise ValueError("Query has no sources")

        first_source = self.query.sources[0]
        namespace = first_source.base.namespace

        # Get the default order field from the namespace
        order_field = namespace.default_order_field

        # Order by default field in ascending order (to get "first")
        self.query.order_by(f"{namespace.name}.{order_field}")
        self.query.limit(1)

        return QuerySetSingleAdapter[Ts](self)

    def last(self) -> "QuerySetSingleAdapter[Ts]":
        """
        Get the last record based on the default_order_field.
        Orders by default_order_field in descending order and limits to 1.
        """
        # Get the namespace from the first source
        if not self.query.sources:
            raise ValueError("Query has no sources")

        first_source = self.query.sources[0]
        namespace = first_source.base.namespace

        # Get the default order field from the namespace
        order_field = namespace.default_order_field

        # Order by default field in descending order (to get "last")
        self.query.order_by(f"-{namespace.name}.{order_field}")
        self.query.limit(1)

        return QuerySetSingleAdapter[Ts](self)

    def head(self, n: int) -> "PgQueryBuilder[Ts]":
        """
        Get the first n records based on the default_order_field.
        Orders by default_order_field in ascending order and limits to n.
        """
        # Get the namespace from the first source
        if not self.query.sources:
            raise ValueError("Query has no sources")

        first_source = self.query.sources[0]
        namespace = first_source.base.namespace

        # Get the default order field from the namespace
        order_field = namespace.default_order_field

        # Order by default field in ascending order (to get first n)
        self.query.order_by(f"{namespace.name}.{order_field}")
        self.query.limit(n)

        return self

    def tail(self, n: int) -> "PgQueryBuilder[Ts]":
        """
        Get the last n records based on the default_order_field.
        Orders by default_order_field in descending order and limits to n.
        """
        # Get the namespace from the first source
        if not self.query.sources:
            raise ValueError("Query has no sources")

        first_source = self.query.sources[0]
        namespace = first_source.base.namespace

        # Get the default order field from the namespace
        order_field = namespace.default_order_field

        # Order by default field in descending order (to get last n)
        self.query.order_by(f"-{namespace.name}.{order_field}")
        self.query.limit(n)

        return self
    
    
    def include(self, target: "Type[Model] | PgQueryBuilder") -> "PgQueryBuilder[Ts]":
        from ..versioning.models import VersionedModel
        # Check if target is already a query builder with nested includes
        if isinstance(target, PgQueryBuilder):
            # Extract model type from the query builder's first source
            target_query_builder = target
            model_cls = target.get_cls()
            # We'll use the existing query instead of creating a fresh one
        else:
            model_cls = target            
            target_query_builder = None
            
        needs_versioning = issubclass(model_cls, VersionedModel)

        rel, relation_source = self._infer_relation(target)
        if rel is None:
            raise ValueError(f"No relation found for {target}")

        # If no fields are explicitly selected yet, select all fields from sources
        # This ensures that include() adds to the selection rather than replacing it
        if self.query.projection_fields is None:
            fields_to_select = []
            for source in self.query.sources:
                for field in source.iter_fields():
                    fields_to_select.append(f"{source.final_name}.{field.name}")
            if fields_to_select:
                self.query.select(*fields_to_select)

        target_rel = NsRelation(rel.foreign_namespace)
        if rel.is_many_to_many:
            # Many-to-many: Need to join through junction table
            # Example: Post.tags -> Post -> PostTag -> Tag

            # Get the junction table relation
            junction_rel = NsRelation(rel.relation_model.get_namespace())

            # Create subquery that joins target with junction
            # Use existing query if provided (for nested includes)
            if target_query_builder:
                target_query = target_query_builder.query
            else:
                target_query = SelectQuerySet(target_rel)

            # Join junction table to target table
            # junction_keys[0] = primary_id (e.g., post_id)
            # junction_keys[1] = foreign_id (e.g., tag_id)
            target_query.join(
                junction_rel,
                on=(rel.foreign_key, rel.junction_keys[1]),  # Tag.id = PostTag.tag_id
                join_type="INNER"
            )

            # Correlate with the source that has the relation
            # Where junction.primary_id = relation_source.primary_key
            # Example: WHERE PostTag.post_id = Post.id
            target_query.where(
                junction_rel.get(rel.junction_keys[0]) == relation_source.get(rel.primary_key)
            )

            # Aggregate into JSON
            json_query = json_agg(target_query, rel.name)
            self.query.select_expr(
                Coalesce(json_query, Value("'[]'", inline=True)),
                alias=rel.name
            )

        elif rel.is_one_to_one:
            # One-to-one: Similar to one-to-many but returns single object, not array
            # Use existing query if provided (for nested includes)
            if target_query_builder:
                target_query = target_query_builder.query
            else:
                target_query = SelectQuerySet(target_rel)

            # Correlate with the source that has the relation
            target_query.where(target_rel.get(rel.foreign_key) == relation_source.get(rel.primary_key))
            target_query.limit(1)  # Ensure single result
            json_query = json_object(target_query, rel.name)
            self.query.select_expr(
                Coalesce(json_query, Value("'null'", inline=True)),  # null, not []
                alias=rel.name
            )
        else:
            # One-to-many: Default case for regular relations
            # Use existing query if provided (for nested includes)
            if target_query_builder:
                target_query = target_query_builder.query
            else:
                target_query = SelectQuerySet(target_rel)

            # Correlate with the source that has the relation
            # Where target.foreign_key = relation_source.primary_key
            target_query.where(target_rel.get(rel.foreign_key) == relation_source.get(rel.primary_key))
            json_query = json_agg(target_query, rel.name)
            self.query.select_expr(
                Coalesce(json_query, Value("'[]'", inline=True)),
                alias=rel.name
            )
            
        if needs_versioning:
            target_query = self.add_versioning_to_query(target_query)
        return self
    
    
    def add_versioning_to_query(self, target_query) -> "PgQueryBuilder[Ts]":
        from ..versioning.models import Artifact
        version_cte = next((cte for cte in self.query.ctes if cte.name == "artifacts_turns_branch_hierarchy"), None)
        if version_cte is None:
            version_cte = Artifact.query()
            self.use_cte(version_cte, cte_name="artifact_cte")
        target_query.join(version_cte, on=("artifact_id", "id"), alias="ac")
        return target_query

            
            
    
    def checkout(self, branch_id: int) -> "PgQueryBuilder[Ts]":
        from ..versioning.models import Artifact
        art_query = Artifact.query(branch_id=branch_id)
        self.join_cte(art_query, cte_name="artifact_cte", alias="ac")
        return self
    
    
    def use_versioning(self) -> "PgQueryBuilder[Ts]":
        from ..versioning.models import Artifact
        art_query = Artifact.query()
        self.join_cte(art_query, cte_name="artifact_cte", alias="ac")
        return self
    
        
    
    def _build_json_query(self, relation: "RelationProtocol", alias: str):
        query = SelectQuerySet(relation)
        {field.name: field for field in relation.iter_fields()}
        fields = {field.name: field for field in relation.iter_fields()}        
        json_obj = JsonBuildObject(fields)
        j_agg = JsonAgg(json_obj)
        query.select_expr(j_agg, alias=alias)
        return query
        
        
    
    def print(self):
        if self.query is None:
            raise ValueError("Query is not set")
        sql, params = self.render()
        print("----- QUERY -----")
        print(sql)
        print("----- PARAMS -----")
        print(params)
        return self
        
    def render(self):
        if self.query is None:
            raise ValueError("Query is not set")
        compiler = Compiler()
        sql, params = compiler.compile(self.query)
        return sql, params
    
    
    
    def __await__(self):
        return self.execute().__await__()
    
    
    def get_cls(self) -> Type[Ts]:
        first_source = self.query.sources[0]
        namespace = first_source.base.namespace
        model_cls = namespace._model_cls
        return model_cls


    def parse_row(self, row: dict[str, Any], return_json: bool = False) -> dict[str, Any] | Ts:
        """
        Parse a database row, deserializing JSON fields from included relations.

        By default returns a Model instance. Use .json() to get dicts instead.
        """
        import json
        from .expressions import Expression

        parsed = {}

        # Iterate over projected fields to know what to parse
        for field in self.query.iter_projection_fields():
            field_name = field.name
            if field_name not in row:
                continue

            value = row[field_name]

            # Check if this field is from an expression (like include())
            if isinstance(field.source, Expression):
                # Deserialize JSON strings from included relations
                if isinstance(value, str) and value:
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as-is if not valid JSON
                parsed[field_name] = value
            else:
                # Regular field - deserialize using field info from namespace
                source_namespace = field.source.base.namespace
                if source_namespace.has_field(field_name):
                    field_info = source_namespace.get_field(field_name)
                    parsed[field_name] = field_info.deserialize(value)
                else:
                    parsed[field_name] = value

        # Return dict if json() was called, otherwise return Model instance
        if return_json:
            return parsed
        else:
            # Get the model class from the first source namespace
            first_source = self.query.sources[0]
            namespace = first_source.base.namespace
            model_cls = namespace._model_cls
            if not model_cls:
                raise ValueError("Model class not set on namespace")
            return model_cls(**parsed)
        
        
    async def json(self) -> list[dict[str, Any]]:
        sql, params = self.render()
        rows = await PGConnectionManager.fetch(sql, *params)
        return [self.parse_row(row, return_json=True) for row in rows]

    
    async def json_parse(self, func: Callable[[list[dict[str, Any]]],  RetModel]) -> RetModel:
        rows = await self.json()
        return func(rows)

    async def execute(self) -> list[Ts]:
        """
        Execute the query and return results.

        Args:
            parse: If True, parse JSON fields from included relations (default: True)
        """
        sql, params = self.render()
        rows = await PGConnectionManager.fetch(sql, *params)

        model_cls = self.get_cls()
        rows = [self.parse_row(row) for row in rows]
        if self._parse_models is not None:
            rows = await self._parse_models(rows)
        return rows
        





def select(*targets: Type[Ts], fields: list[str] | str | None = "*")->PgQueryBuilder[Ts]:
    # return PgQueryBuilder().select(*targets)
    # return PgQueryBuilder().select(*targets).use_versioning()
    query = targets[0].query()
    for target in targets[1:]:
        query.join(target.query())
    return query