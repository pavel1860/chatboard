from ..base.base_field_info import BaseFieldInfo
from enum import Enum
from typing import Any, Optional

class Distance(str, Enum):
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"

class QdrantFieldInfo(BaseFieldInfo):
    def __init__(
        self,
        name: str,
        field_type: type,
        *,
        is_vector: bool = False,
        dimension: Optional[int] = None,
        distance: Optional[Distance] = None,
        **kwargs
    ):
        super().__init__(name, field_type, is_vector=is_vector, dimension=dimension, **kwargs)
        self.distance = distance

        if self.is_vector and (self.dimension is None or self.distance is None):
            raise ValueError(f"Vector field '{name}' requires both 'dimension' and 'distance'")

    def serialize(self, value: Any) -> Any:
        if self.is_vector:
            return [float(x) for x in value]
        return super().serialize(value)

    def deserialize(self, value: Any) -> Any:
        if self.is_vector:
            return value  # Already float[]
        return super().deserialize(value)
