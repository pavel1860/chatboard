from typing import TYPE_CHECKING, Any, Generator, Generic, Type, TypeVar
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


def json_agg(query: "SelectQuerySet", alias: str):
    json_obj = JsonBuildObject({field.name: field for field in query.iter_fields()})
    j_agg = JsonAgg(json_obj)
    query.select_scalar(j_agg)
    return query

T_co = TypeVar("T_co", covariant=True)


class QuerySetSingleAdapter(Generic[T_co]):
    def __init__(self, queryset: "PgQueryBuilder[T_co]", parse: bool = True):
        self.queryset = queryset
        self.parse = parse

    def __await__(self) -> Generator[Any, None, T_co]:
        async def await_query():
            results = await self.queryset.execute(parse=self.parse)
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


    @property
    def query(self):
        if self._query is None:
            raise ValueError("Query is not set")
        return self._query
        
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
            
        return self
    
    
    def raw(self, sql: str, name: str, namespace: "BaseNamespace"):
        if not self._query is None:
            raise ValueError("Query is already set")
        raw_rel = RawRelation(sql, name, namespace)
        self._query = SelectQuerySet(raw_rel)
        return self
    
    
    def where(self, condition: Expression):
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

    def json(self) -> "PgQueryBuilder[Ts]":
        """
        Configure the query to return dicts instead of Model instances.

        Usage:
            posts = await select(Post).include(Comment).json()
            # posts[0] is a dict
        """
        self._return_json = True
        return self

    def use_cte(self, cte: "PgQueryBuilder[Ts]", alias: str | None = None) -> "PgQueryBuilder[Ts]":
        """
        Add a CTE (Common Table Expression) to this query.

        Args:
            cte: The query builder to use as a CTE
            alias: Optional custom name for the CTE

        Usage:
            # Create a CTE for popular posts
            popular_posts = SelectQuerySet(posts_rel)
            popular_posts.select("posts.id", "posts.title")
            popular_posts.where(posts_rel.get("views") > 1000)

            # Use it in main query with custom name
            results = await select(Post).use_cte(popular_posts, alias="popular")
        """
        self.query.with_cte(cte.query, alias=alias)
        return self
    
    def join(self, target: "PgQueryBuilder[Ts]", on: tuple[str, str] | None = None):
        if on is None:
            rel = self._infer_relation(target)            
            if rel is None:
                raise ValueError(f"No relation found for {target}")
            self.query.join(rel, on=(rel.primary_key, rel.foreign_key))
            


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
    
    
    def include(self, target: "Type[Model]") -> "PgQueryBuilder[Ts]":
        rel = self._infer_relation(target)
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
            raise NotImplementedError("Many-to-many relations are not supported yet")
            # select(rel.relation_model)
            # target_query.where(target_rel.get(rel.primary_key) == target_rel.get(rel.foreign_key))
            # json_query = json_agg(target_query, rel.name)
            self.query.select_expr(
                Coalesce(json_query, Value("'[]'", inline=True)),
                alias=rel.name
            )

        elif rel.is_one_to_one:
            raise NotImplementedError("One-to-one relations are not supported yet")
        else:
            target_query = SelectQuerySet(target_rel)
            target_query.where(target_rel.get(rel.primary_key) == target_rel.get(rel.foreign_key))
            json_query = json_agg(target_query, rel.name)
            self.query.select_expr(
                Coalesce(json_query, Value("'[]'", inline=True)),
                alias=rel.name
            )
            # self.query.join(json_query.sources[0], on=(rel.primary_key, rel.foreign_key), join_type="LEFT")
        return self
    
        
    def _infer_relation(self, target: "Type[Model] | PgQueryBuilder[Ts]"):
        if isinstance(target, PgQueryBuilder):
            ns = target.query.namespace
            if ns is None:
                raise ValueError("Namespace is not set")
            for source in self.query.sources:
                if isinstance(source.base, RawRelation):
                    if source.base.namespace is None:
                        continue
                    return source.base.namespace.get_relation_for_namespace(ns)
        else:
            ns = target.get_namespace()
            for source in self.query.sources: 
                rel = source.base.namespace.get_relation_for_namespace(ns)
                if rel is not None:
                    return rel
        return None
    
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


    def parse_row(self, row: dict[str, Any]) -> dict[str, Any] | Ts:
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
        if self._return_json:
            return parsed
        else:
            # Get the model class from the first source namespace
            first_source = self.query.sources[0]
            namespace = first_source.base.namespace
            model_cls = namespace._model_cls
            if not model_cls:
                raise ValueError("Model class not set on namespace")
            return model_cls(**parsed)


    async def execute(self, parse: bool = True):
        """
        Execute the query and return results.

        Args:
            parse: If True, parse JSON fields from included relations (default: True)
        """
        sql, params = self.render()
        rows = await PGConnectionManager.fetch(sql, *params)

        if parse:
            return [self.parse_row(row) for row in rows]
        return rows
        





def select(*targets: Type[Ts])->PgQueryBuilder[Ts]:
    return PgQueryBuilder().select(*targets)