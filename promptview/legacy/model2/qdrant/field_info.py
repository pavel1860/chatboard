from promptview.model.base_namespace import NSFieldInfo
from typing import Any, Type
import datetime as dt
import uuid
import json

class QdrantFieldInfo(NSFieldInfo):
    """Qdrant-compatible field information for payload serialization/validation."""

    def serialize(self, value: Any) -> Any:
        """Serialize the value for Qdrant payload."""
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dt.datetime):
            return value.isoformat()

        if isinstance(value, uuid.UUID):
            return str(value)

        if isinstance(value, list):
            return [self.serialize(v) for v in value]

        if isinstance(value, dict):
            return {k: self.serialize(v) for k, v in value.items()}

        raise ValueError(f"Unsupported Qdrant payload type: {type(value)}")

    def deserialize(self, value: Any) -> Any:
        """Deserialize payload value from Qdrant."""
        if self.data_type == dt.datetime and isinstance(value, str):
            return dt.datetime.fromisoformat(value)
        if self.data_type == uuid.UUID and isinstance(value, str):
            return uuid.UUID(value)
        return value
