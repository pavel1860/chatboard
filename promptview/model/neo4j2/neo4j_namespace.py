# neo4j_namespace.py


from ..base.base_namespace import BaseNamespace
from .neo4j_field_info import Neo4jFieldInfo
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from ..model3 import Model

class Neo4jNamespace(BaseNamespace["Model", Neo4jFieldInfo]):
    def __init__(self, name: str, *fields: Neo4jFieldInfo):
        super().__init__(name, db_type="neo4j")
        self._primary_key: Optional[Neo4jFieldInfo] = None

        for field in fields:
            self._register_field(field)

    def _register_field(self, field: Neo4jFieldInfo):
        if field.is_primary_key:
            if self._primary_key:
                raise ValueError(f"Primary key already defined: {self._primary_key.name}")
            self._primary_key = field
        self.add_field(field)

    @property
    def primary_key(self) -> Neo4jFieldInfo:
        if not self._primary_key:
            raise ValueError(f"No primary key defined for Neo4j namespace '{self.name}'")
        return self._primary_key

    def __repr__(self):
        return f"<Neo4jNamespace {self.name} fields={[f.name for f in self.iter_fields()]}>"
