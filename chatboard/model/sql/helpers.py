
from typing import TYPE_CHECKING

from ..relation_info import RelationInfo
from .expressions import Coalesce, json_build_object, Function, Not, IsNull, Value, Eq
from .queries import Column, SelectQuery, Table
if TYPE_CHECKING:
    from ..postgres2.pg_query_set import PgSelectQuerySet











class NestedQuery:
    
    
    def __init__(
        self, 
        query: "SelectQuerySet", 
        alias: str, 
        pk_col: Column, 
        parent_query: "SelectQuerySet",
        parent_table: Table,
        rel: RelationInfo
    ):
        self.query_set = query
        self.alias = alias
        self.pk_col = pk_col
        self.parent_query = parent_query
        self.parent_table = parent_table
        self.depth = 1
        self.rel = rel
        
    @property
    def name(self):
        return self.alias
        
    @property
    def query(self):
        return self.query_set.query
    
    def update_depth(self):
        for c in self.query.columns:
            if isinstance(c, NestedQuery):
                c.depth = self.depth + 1
                c.update_depth()
    
    # @property
    # def depth(self):
    #     depth = 1
    #     parent = self.parent_query
    #     for i in range(10):
            
    #     return self.parent_query.query_depth + 1
    def build_query(self):
        if not self.query.columns:
            raise ValueError(f"Nested Query set for model {self.query_set.model_class.__name__} has no columns")
        if self.depth == 1:
            return self.wrap_query_in_json_agg()
        else:
            return self.embed_query_as_subquery()
    
    def wrap_query_in_json_agg(self) -> Coalesce:
        obj = json_build_object(**{col.name: col for col in self.query.columns})
        agg = Function(
            "json_agg",
            obj,
            distinct=True,
            filter_where=Not(IsNull(self.pk_col))
        )
        return Coalesce(agg, Value("[]", inline=True), alias=self.alias)

    def embed_query_as_subquery(self):
        obj = json_build_object(**{
            c.name: c for c in self.query.columns
        })
        subq = SelectQuery()
        subq.columns = [Function("json_agg", obj)]
        subq.from_table = self.query.from_table
        subq.where &= Eq(Column(self.rel.foreign_key, self.query.from_table), Column(self.rel.primary_key, self.parent_table))
        coalesced = Coalesce(subq, Value("[]", inline=True))
        return coalesced
        

    def get_query(self):
        if self.query_set.query_depth == 1:
            return self.wrap_query_in_json_agg()
        else:
            return self.query
