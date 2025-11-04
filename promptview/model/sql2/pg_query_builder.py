from typing import TYPE_CHECKING, Any, Generator, Generic, Type, TypeVar
from .relations import RelationProtocol, Source, NsRelation, RelField, Relation
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
    def __init__(self, queryset: "PgQueryBuilder[T_co]"):
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
        
    def _infer_relation(self, target: "Type[Model]"):
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
    
    
    # def parse_row(self, row: dict[str, Any]) -> MODEL:
    #     # Convert scalar columns first
    #     data = dict(row)        
    #     data = self.namespace.deserialize(data)

    #     obj = self.model_class(**data)
    #     if self.parser:
    #         obj = self.parser(obj)
    #     return obj
            
        
    async def execute(self):
        sql, params = self.render()
        rows = await PGConnectionManager.fetch(sql, *params)
        return rows
        # return [self.parse_row(row) for row in rows]
        # return [self.query.parse_row(row) for row in rows]
        





def select(*targets: Type[Ts])->PgQueryBuilder[Ts]:
    return PgQueryBuilder().select(*targets)