from .relations import RelationProtocol, JoinedRelation, Relation, RelField, TypeRelationInput, SubQueryRelation
from typing import Iterator






def join(left_rel: RelationProtocol, right_rel: RelationProtocol, on: tuple[str, str], join_type: str = "INNER", alias: str | None = None) -> RelationProtocol:
    l_source, l_field = left_rel.get_source_and_field(on[0])
    r_source, r_field = right_rel.get_source_and_field(on[1])    
    right_join_rel = JoinedRelation(l_source, r_source, (l_field, r_field), join_type, alias)
    return Relation(
        left_rel.sources + (right_join_rel,),
        alias=left_rel.alias
    )
    







class QuerySet(Relation):
    def __init__(self, sources: TypeRelationInput, alias: str | None = None):
        super().__init__(sources, alias=alias)        
        self.projection_fields: dict[str, dict] | None = None
        self.ctes = list[QuerySet]()
        self.recursive_cte = False       
        
        
    # def select(self, *fields: str):
    #     self.target = project(self.target, fields)
    #     return self
    
    def join(self, target: RelationProtocol, on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
        l_source, l_field = self.get_source_and_field(on[0])
        r_source, r_field = target.get_source_and_field(on[1])    
        right_join_rel = JoinedRelation(l_source, r_source, (l_field, r_field), join_type, alias)
        self.sources = self.sources + (right_join_rel,)        
        return self
    
    # projection
    def select(self, *fields: str):
        self.projection_fields = {}
        for f in fields:
            self.get(f)
            self.projection_fields[f] = {}        
        return self
    
    
    def include(self, target: "QuerySet", alias: str):
        sub_query = SubQueryRelation(target, alias=alias)
        self.sources = self.sources + (sub_query,)
        return self
    
    def with_cte(self, cte: "QuerySet"):
        self.ctes.append(cte)
        return self
    
    def iter_projection_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        for field in self.iter_fields(include_sources):
            if self.projection_fields is None or field.name in self.projection_fields:
                yield field

        
            
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.sources})"
    


class SelectQuerySet(QuerySet):
    pass
        
        
    
        
        
    
    


