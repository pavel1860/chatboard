# neo4j_field_info.py

from typing import Any, Optional, Type
from ..base.base_field_info import BaseFieldInfo

class Neo4jFieldInfo(BaseFieldInfo):
    def __init__(
        self,
        name: str,
        field_type: Type[Any],
        default: Any = None,
        is_primary_key: bool = False,
    ):
        super().__init__(name=name, field_type=field_type, default=default, is_primary_key=is_primary_key)

    def serialize(self, value: Any) -> Any:
        if value is None:
            return self.default
        # Neo4j supports: int, float, str, bool, list of primitives
        if isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, list) and all(isinstance(v, (int, float, str, bool)) for v in value):
            return value
        raise TypeError(f"Unsupported Neo4j value type: {type(value)} for field '{self.name}'")

    def deserialize(self, value: Any) -> Any:
        return value

    def __repr__(self):
        base = f"Neo4jFieldInfo(name={self.name}, type={self.data_type.__name__}"
        if self.is_primary_key:
            base += ", primary_key=True"
        return base + ")"
