from pydantic import BaseModel
from datetime import datetime, date, time

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
    val = REVERSE_TYPE_REGISTRY.get(value_type, None)
    if val is None:
        raise UnknownType(f"Unknown type: {value_type}")
    return val

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