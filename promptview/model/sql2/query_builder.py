from typing import TYPE_CHECKING, Generic, Type, TypeVar
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



class QueryBuilder(Generic[Ts]):
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
    
    
    def include(self, target: "Type[Model]"):
        rel = self._infer_relation(target)
        if rel is None:
            raise ValueError(f"No relation found for {target}")
        
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
            
        
    async def execute(self):
        sql, params = self.render()
        rows = await PGConnectionManager.fetch(sql, *params)
        return rows
        # return [self.query.parse_row(row) for row in rows]
        





def select(*targets: Type[Ts])->QueryBuilder[Ts]:
    return QueryBuilder().select(*targets)