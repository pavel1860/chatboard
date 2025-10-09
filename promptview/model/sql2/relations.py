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
    
    
class SubQueryRelation:
    
    def __init__(self, source: "QuerySet", alias: str):
        self.sources = (source,)
        self._alias = alias
    
    @property
    def name(self) -> str:
        return self._alias
    
    @property
    def alias(self) -> str:
        return self._alias
    
    @property
    def final_name(self) -> str:
        return self.alias or self.name
    
    def get(self, field_name: str) -> RelField:
        if field_name == self.alias:
            return RelField(self.sources[0], self.alias, is_query=True)
        raise ValueError(f"Field {field_name} not found on {self.name}")
    
    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        yield RelField(self.sources[0], self.alias, is_query=True)
        
    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        return self.sources[0], self.get(field_name)
    
    
TypeRelationInput = tuple[RelationProtocol, ...] | RelationProtocol


class Relation:
    
    
    def __init__(self, sources: TypeRelationInput, alias: str | None = None):
        if not isinstance(sources, tuple):
            sources = (sources,)
        self.sources = sources
        self.name = self._gen_name(sources)
        self.alias = alias
        
        
    @property
    def final_name(self) -> str:
        return self.alias or self.name
        
    
    def _gen_name(self, sources: tuple[RelationProtocol, ...]):
        if not isinstance(sources, tuple):
            sources = (sources,)
        if len(sources) == 1:
            if isinstance(sources[0], BaseNamespace):
                return sources[0].name
            else:
                return sources[0].name
        else:
            return "_".join([source.name for source in sources])
        
    # def join(self, target: "Relation", on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
    #     # self.sources = self.sources + (target,)
    #     return Relation(self.sources + (target,), alias=self.alias)
    def get_source_and_field(self, field_name: str) -> tuple[RelationProtocol, RelField]:
        _f = field_name.split(".")
        if len(_f) == 1:
            # if len(self.sources) == 1:
            #     source = self.sources[0]
            #     field = self._get_field(source, _f[0])
            #     return source, field
            # else:
            #     raise ValueError(f"for multiple sources, field name must be in the format of 'source.field'")
            source = self.sources[0]
            field = self._get_field(source, _f[0])
            return source, field
        elif len(_f) == 2:
            source = self._get_source(_f[0])
            field = self._get_field(source, _f[1])
            return source, field
        else:
            raise ValueError(f"Invalid field name: {field_name} on {self.name}")
        
    def _get_source(self, source_name: str) -> RelationProtocol:        
        for source in self.sources:
            if source.name == source_name:
                return source
        raise ValueError(f"Source {source_name} not found on {self.name}")
    
    def _get_field(self, source: RelationProtocol, field_name: str) -> RelField:
        for field in self.iter_fields(include_sources={source.name}):
            if field.name == field_name:
                return field
        raise ValueError(f"Field {field_name} not found on {source.name}")
        
    def get(self, field_name: str) -> RelField:
        _, field = self.get_source_and_field(field_name)
        return field
        
    
    def iter_fields(self, include_sources: set[str] | None = None) -> Iterator[RelField]:
        for source in self.sources:
            if include_sources is not None and source.name not in include_sources:
                continue
            for field in source.iter_fields():
                    yield field
                    
    def print(self):
        return f"REL({self.name})"
            
            
    def print_tree(self, indent: int = 0):
        print(" " * indent + "REL", self.name)
        indent += 1
        for source in self.sources:
            if isinstance(source, NsRelation):
                print(" " * indent + "NS", source.name)
            elif isinstance(source, Relation):
                source.print_tree(indent)
            else:
                raise ValueError(f"Invalid source: {source}")
            
        
        
    def __repr__(self):
        fields = ", ".join([f"{field}" for field in self.iter_fields()])
        # table = self.source.name if isinstance(self.source, BaseNamespace) else "None"
        return f"{self.__class__.__name__}({self.name} -> {{{fields}}})"


# class RelationProjection(Relation):
    
#     def __init__(self, sources: "tuple[RelationProtocol, ...]", fields: tuple[str, ...], alias: str | None = None):
#         super().__init__(sources, alias=alias)
#         self.fields = fields
        
        
#     def iter_fields(self):
#         for field in super().iter_fields():
#             if field.name in self.fields:
#                 yield field
        

class JoinedRelation(Relation):
    def __init__(self, left_rel: RelationProtocol, right_rel: RelationProtocol, on: tuple[RelField, RelField], join_type: str = "INNER", alias: str | None = None):
        super().__init__((right_rel,), alias=alias)
        self.left_rel = left_rel
        self.right_rel = right_rel
        self.on = on
        self.join_type = join_type
        
        
    def get_on_clause(self) -> str:
        return f"{self.left_rel.final_name}.{self.on[0].name} = {self.right_rel.final_name}.{self.on[1].name}"
