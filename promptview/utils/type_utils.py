from typing import Any, Union, get_origin, get_args
from types import UnionType, NoneType
from pydantic import BaseModel
from datetime import datetime, date, time
from enum import Enum
from promptview.model.base.base_namespace import Serializable



SerializableType = int | str | float | bool | list | dict | datetime | date | time | BaseModel


DeserializableType = dict | list | str | int | float | bool



TYPE_REGISTRY = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "datetime": datetime,
    "date": date,
    "time": time,
    "BaseModel": BaseModel,
}

REVERSE_TYPE_REGISTRY = {v: k for k, v in TYPE_REGISTRY.items()}

class UnknownType(Exception):
    pass

def type_to_str(value_type: type) -> str:
    # Handle Union types (including X | None syntax)
    origin = get_origin(value_type)
    if origin is Union or isinstance(value_type, UnionType):
        args = get_args(value_type)
        # Filter out NoneType to get the actual types
        non_none_args = [arg for arg in args if arg is not NoneType]
        is_optional = len(non_none_args) < len(args)

        # Convert each non-None type to string
        type_strs = [type_to_str(arg) for arg in non_none_args]
        result = " | ".join(type_strs)

        if is_optional:
            result = f"{result} | None"
        return result

    # Check registry first
    val = REVERSE_TYPE_REGISTRY.get(value_type, None)
    if val is not None:
        return val

    # For class types not in registry, use the class name
    if isinstance(value_type, type):
        return value_type.__name__

    raise UnknownType(f"Unknown type: {value_type}")

def type_to_str_or_none(value_type: type) -> str | None:
    try:
        return type_to_str(value_type)
    except UnknownType:
        return None




def str_to_type(value_type: str) -> type:
    val = TYPE_REGISTRY.get(value_type, None)
    if val is None:
        raise UnknownType(f"Unknown type: {value_type}")
    return val



def serialize_value(value: SerializableType) -> DeserializableType:
    if isinstance(value, BaseModel):
        return value.model_dump()
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, time):
        return value.isoformat()
    return value

def deserialize_value(value: DeserializableType, hint: str) -> SerializableType:
    _type = str_to_type(hint)
    if _type == BaseModel:
        return BaseModel.model_validate(value)
    elif _type == datetime:
        return datetime.fromisoformat(value)
    elif _type == date:
        return date.fromisoformat(value)
    elif _type == time:
        return time.fromisoformat(value)
    return _type(value)





class UnsetType(Enum):
    UNSET = "UNSET"
    
    def __repr__(self):
        return "UNSET"
    
    def __bool__(self):
        return False
    
    @staticmethod
    def get_value(value: Any, default: Any = None) -> Any:
        """
        Resolve UNSET to default value.

        - get_value(UNSET, x) → x (use default, even if None)
        - get_value(None, x) → None (explicit clear)
        - get_value(val, x) → val (use provided value)
        """
        if value is UNSET:
            return default
        return value

UNSET = UnsetType.UNSET