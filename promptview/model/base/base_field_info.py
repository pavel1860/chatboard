# base_field_info.py

from typing import Any, List, Type, Optional, get_args, get_origin
import uuid
import datetime as dt
class BaseFieldInfo:
    def __init__(
        self,
        name: str,
        field_type: Type,
        *,
        default: Any = None,
        is_optional: bool = False,
        is_primary_key: bool = False,
        is_foreign_key: bool = False,
        is_vector: bool = False,
        dimension: Optional[int] = None,
        distance: str = "cosine",
        is_key: bool = False,
        index: bool = False,
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        enum_values: Optional[List[str]] = None,
        order_by: bool = False,
        foreign_cls: Optional[Type] = None,
        self_ref: bool = False,
        rel_name: str | None = None,
        enforce_foreign_key: bool = True,
    ):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.is_optional = is_optional
        self.is_primary_key = is_primary_key
        self.is_foreign_key = is_foreign_key
        self.is_vector = is_vector
        self.dimension = dimension
        self.is_key = is_key or is_primary_key
        self.index = index
        self.on_delete = on_delete
        self.on_update = on_update
        self.enum_values = enum_values
        self.is_enum = enum_values is not None
        self.order_by = order_by
        self.foreign_cls = foreign_cls
        self.self_ref = self_ref
        self.rel_name: str | None = rel_name
        self.enforce_foreign_key = enforce_foreign_key
        # For key fields (uuid or int)
        if self.is_key:
            self.key_type = "uuid" if field_type is uuid.UUID else "int"

        origin = get_origin(field_type)
        # Detect list fields
        self.is_list = origin in (list, List)
        if origin == dict:
            self.data_type = dict
        else:
            self.data_type = get_args(field_type)[0] if self.is_list else field_type
        self.is_temporal = self.data_type is type and issubclass(self.data_type, (dt.datetime, dt.date, dt.time))

    def serialize(self, value: Any) -> Any:
        return value

    def deserialize(self, value: Any) -> Any:
        return value
