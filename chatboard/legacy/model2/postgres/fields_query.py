import textwrap
import inspect
from enum import Enum
import json
from typing import TYPE_CHECKING, List, Literal, Set, Any, Optional, Iterator
import uuid
import datetime as dt
import numpy as np
from pydantic import BaseModel
from chatboard.model.vectors import Vector
from chatboard.utils.model_utils import get_list_type, is_list_type, make_json_serializable, make_json_string_deserializable
from chatboard.model.base_namespace import NSFieldInfo, Serializable
if TYPE_CHECKING:
    from chatboard.model.base_namespace import Model, Namespace
    
    
PgIndexType = Literal["btree", "hash", "gin", "gist", "spgist", "brin"]   
    
    
    
class PgFieldInfo(NSFieldInfo):
    """PostgreSQL field information"""
    index: PgIndexType | None = None
    sql_type: str
    
    SERIAL_TYPE = "SERIAL"
    
    def __init__(
        self,
        name: str,
        field_type: type[Any],
        default: Any | None = None,
        is_optional: bool = False,
        foreign_key: bool = False,
        is_key: bool = False,
        is_vector: bool = False,
        dimension: int | None = None,
        is_primary_key: bool = False,
        namespace: "Namespace | None" = None,
    ):
        super().__init__(
            name, 
            field_type, 
            default=default,
            is_optional=is_optional,
            foreign_key=foreign_key,
            is_key=is_key,
            is_vector=is_vector,
            dimension=dimension,
            namespace=namespace,
            is_primary_key=is_primary_key,
            )
        if is_primary_key and name == "id" and field_type is int:
            self.sql_type = PgFieldInfo.SERIAL_TYPE  # Use the constant from SQLBuilder
        else:            
            self.sql_type = self.build_sql_type()
        # if extra and "index" in extra:
        #     self.index = extra["index"]
            
    def serialize(self, value: Any) -> Any:
        """Serialize the value for the database"""
        if value is None:
            if self.default is not None:
                return self.default
            return None
        if self.is_key and self.key_type == "uuid" and value is None:
            value = str(uuid.uuid4())
        elif self.is_vector:
            value = f"[{', '.join(map(str, value))}]"
        elif self.sql_type == "JSONB":
            if self.is_list:
                value = [make_json_serializable(item) if isinstance(item, dict) else item for item in value]
                value = json.dumps(value)
            # elif issubclass(self.data_type, BaseModel):
                # if type(value) is dict:
                    # value = 
                # value = value.model_dump()
            elif self.data_type is dict or issubclass(self.data_type, BaseModel):
                if isinstance(value, Serializable):
                    value = value.serialize()
                else:
                    value = make_json_serializable(value)
                return json.dumps(value)

                # parsed_values = {}
                # for k, v in value.items():
                #     if isinstance(v, uuid.UUID):
                #         parsed_values[k] = str(v)
                #     elif isinstance(v, dt.datetime):
                #         parsed_values[k] = v.isoformat()
                #     else:
                #         parsed_values[k] = v                                        
                # try:
                #     return json.dumps(parsed_values)
                # except Exception as e:
                #     raise ValueError(f"Failed to serialize dict: {value}") from e
                
                
        return value
    
    def get_placeholder(self, index: int) -> str:
        """Get the placeholder for the value"""
        if self.sql_type == "JSONB":
            if self.is_list:
                return f'${index}::JSONB'
            else:
                return f'${index}::JSONB'
        elif self.is_temporal:
            if self.db_type:
                return f'${index}::{self.db_type}'
            return f'${index}::TIMESTAMP'
        # elif self.is_vector:
            # return f'${index}::vector'
        else:
            return f'${index}'
    
    def deserialize(self, value: Any) -> Any:
        """Deserialize the value from the database"""
        if self.is_key and self.key_type == "uuid":
            if type(value) is None:
                raise ValueError("UUID field can not be None")
            return uuid.UUID(str(value))            
        if value is None:
            return value
        if self.is_list and type(value) is str:
            return json.loads(value)
        elif self.data_type is dict:
            if type(value) is str:
                return make_json_string_deserializable(value)
            else:
                return value
        elif self.is_enum and not self.is_literal:
            return self.data_type(value)
        elif self.is_vector:
            return np.fromstring(value.strip("[]"), sep=",")
        elif issubclass(self.data_type, BaseModel):
            if type(value) is str:
                return self.data_type.model_validate_json(value)
            else:
                if self.is_list:
                    return [self.data_type.model_validate(item) for item in value]
                return self.data_type.model_validate(value)
        return value
            
    
    def build_sql_type(self) -> str:
        sql_type = None
        
        # if inspect.isclass(self.data_type):
        #     if issubclass(self.data_type, BaseModel):
        #         sql_type = "JSONB"
        #     elif issubclass(self.data_type, Enum):
        #         sql_type = "TEXT" 
        if self.db_type:
            sql_type = self.db_type               
        elif self.is_temporal:
            sql_type = "TIMESTAMP"
        elif self.is_enum:
            sql_type = self.enum_name
        elif self.is_enum:
            sql_type = "TEXT"
        elif self.data_type is int:
            sql_type = "INTEGER[]" if self.is_list else "INTEGER"
        elif self.data_type is float:
            sql_type = "FLOAT[]" if self.is_list else "FLOAT"
        elif self.data_type is str:
            sql_type = "TEXT[]" if self.is_list else "TEXT"
        elif self.data_type is bool:
            sql_type = "BOOLEAN[]" if self.is_list else "BOOLEAN"
        elif self.data_type is uuid.UUID:
            sql_type = "UUID[]" if self.is_list else "UUID"
        elif self.data_type is dict:
            sql_type = "JSONB"
        elif issubclass(self.data_type, BaseModel):
            sql_type = "JSONB"
        elif issubclass(self.data_type, Enum):
            sql_type = "TEXT"
        elif self.data_type is Vector:
            sql_type = f"VECTOR({self.dimension})"        
        if sql_type is None:
            raise ValueError(f"Unsupported field type: {self.data_type}")
        return sql_type
    
    
    @classmethod
    def to_sql_type(cls, field_type: type[Any], extra: dict[str, Any] | None = None) -> str:
        """Map a Python type to a SQL type"""
        db_field_type = None
        if is_list_type(field_type):
            list_type = get_list_type(field_type)
            if list_type is int:
                db_field_type = "INTEGER[]"
            elif list_type is float:
                db_field_type = "FLOAT[]"
            elif list_type is str:
                db_field_type = "TEXT[]"
            elif inspect.isclass(list_type):
                if issubclass(list_type, BaseModel):
                    db_field_type = "JSONB"                        
        else:
            if extra and extra.get("db_type"):
                custom_type = extra.get("db_type")
                if type(custom_type) != str:
                    raise ValueError(f"Custom type is not a string: {custom_type}")
                db_field_type = custom_type
            elif field_type == bool:
                db_field_type = "BOOLEAN"
            elif field_type == int:
                db_field_type = "INTEGER"
            elif field_type == float:
                db_field_type = "FLOAT"
            elif field_type == str:
                db_field_type = "TEXT"
            elif field_type == dt.datetime:
                # TODO: sql_type = "TIMESTAMP WITH TIME ZONE"
                db_field_type = "TIMESTAMP"
            elif isinstance(field_type, dict):
                db_field_type = "JSONB"
            elif isinstance(field_type, type):
                if issubclass(field_type, BaseModel):
                    db_field_type = "JSONB"
                if issubclass(field_type, Enum):
                    db_field_type = "TEXT"                                        
        if db_field_type is None:
            raise ValueError(f"Unsupported field type: {field_type}")
        return db_field_type
    
    def __repr__(self) -> str:
        params = self._param_repr_list()
        return f"PgFieldInfo(name={self.name}, data_type={self.data_type.__name__}, sql_type={self.sql_type}, {', '.join(params)})"





