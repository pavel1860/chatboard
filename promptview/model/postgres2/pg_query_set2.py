import uuid

from ..base.base_namespace import BaseNamespace
from ..postgres2.pg_field_info import PgFieldInfo

class RelField:
    def __init__(self, source,  name: str, field_info: PgFieldInfo, alias: str | None = None):
        self.name = name
        self.field_info = field_info
        self.alias = alias
        self.source = source

    def __repr__(self):
        return f"RelField({self.source.name}.{self.name})"
    
    def __str__(self):
        return f"{self.source.name}.{self.name}"

class Relation:
    
    
    def __init__(self, sources: "tuple[Relation | BaseNamespace, ...]", alias: str | None = None):
        self.sources = sources
        self.name = self._gen_name(sources)
        self.alias = alias
        
    
    def _gen_name(self, sources: "tuple[Relation | BaseNamespace, ...]"):
        if len(sources) == 1:
            if isinstance(sources[0], BaseNamespace):
                return sources[0].name
            else:
                return sources[0].name
        else:
            return "_".join([source.name for source in sources])
    
    def iter_fields(self):
        for source in self.sources:
            if isinstance(source, BaseNamespace):
                for field_info in source.iter_fields():
                    yield RelField(source, field_info.name, field_info)
            elif isinstance(source, Relation):
                for field in source.iter_fields():
                    yield field
            else:
                raise ValueError(f"Invalid source: {source}")
            
        
        
    def __repr__(self):
        fields = ", ".join([f"{field}" for field in self.iter_fields()])
        # table = self.source.name if isinstance(self.source, BaseNamespace) else "None"
        return f"{self.__class__.__name__}({self.name} -> {{{fields}}})"


class RelationProjection(Relation):
    
    def __init__(self, sources: "tuple[Relation | BaseNamespace, ...]", fields: tuple[str, ...], alias: str | None = None):
        super().__init__(sources, alias=alias)
        self.fields = fields
        
        
    def iter_fields(self):
        for field in super().iter_fields():
            if field.name in self.fields:
                yield field
        

class JoinedRelation(Relation):
    def __init__(self, left_rel: "Relation", right_rel: "Relation", on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
        super().__init__((left_rel, right_rel), alias=alias)
        self.on = on
        self.join_type = join_type

def project(relation: "Relation", fields: tuple[str, ...]):     
    return RelationProjection(relation.sources, fields, alias=relation.alias)
    

def join(left_rel: "Relation", right_rel: "Relation", on: tuple[str, str], join_type: str = "INNER", alias: str | None = None):
    return JoinedRelation(
        left_rel,
        right_rel,
        alias=alias,
        on=on,
        join_type=join_type
    )
