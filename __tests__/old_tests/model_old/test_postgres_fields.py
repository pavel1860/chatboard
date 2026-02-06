from enum import StrEnum
from typing import Literal, Optional, Type
from chatboard.model.model import Model
from chatboard.model.fields import ModelField
from chatboard.model.postgres.namespace import PgFieldInfo, PostgresNamespace
from pydantic import BaseModel


class TestField(BaseModel):
    name: str
    age: int
    is_admin: bool
    
    
class TestEnum(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TestModel(BaseModel):
    str_field: str = ModelField(default="")
    optional_int_field: int | None = ModelField()
    float_field: Optional[float] = ModelField(default=0)
    bool_field: bool = ModelField(default=False)
    list_type: list[str] = ModelField(default=[])
    optional_list: list[str] | None = ModelField(default=[])
    model_list: list[TestField] = ModelField(default=[])
    dict_field: dict[str, int] = ModelField(default={})
    optional_dict: dict[str, int] | None = ModelField(default=None)
    model_dict: dict[str, TestField] = ModelField(default={})
    literal_field: Literal["user", "assistant"] = ModelField(default="user")
    enum_field: TestEnum = ModelField(default=TestEnum.USER)
    
ns = PostgresNamespace("test_model", is_versioned=False)

# for field, info in TestModel.model_fields.items():
def get_field_info(field: str, model: Type[BaseModel]):
    info = model.model_fields[field]
    name = field
    field_type = info.annotation
    if not field_type:
        raise ValueError(f"Field {field} has no type")
    extra = info.json_schema_extra
    if extra is None:
        extra = {}
    return  name, field_type, extra

def build_field_info(field: str):
    name, field_type, extra = get_field_info(field, TestModel)    
    field_info = ns.add_field(name, field_type, extra)
    return field_info



def test_str_field():
    field_info = build_field_info("str_field")    
    assert field_info.name == "str_field"
    assert field_info.sql_type == "TEXT"
    assert field_info.is_optional is False
    assert field_info.field_type is str


def test_optional_int_field():
    field_info = build_field_info("optional_int_field")     
    assert field_info.name == "optional_int_field"
    assert field_info.sql_type == "INTEGER"
    assert field_info.data_type is int
    assert field_info.is_optional

def test_float_field():
    field_info = build_field_info("float_field")
    assert field_info.name == "float_field"
    assert field_info.sql_type == "FLOAT"
    assert field_info.data_type is float
    assert field_info.is_optional


def test_bool_field():
    field_info = build_field_info("bool_field")
    assert field_info.name == "bool_field"
    assert field_info.sql_type == "BOOLEAN"
    assert field_info.data_type is bool
    assert field_info.is_optional is False

def test_list_type():
    field_info = build_field_info("list_type")
    assert field_info.name == "list_type"
    assert field_info.sql_type == "TEXT[]"
    assert field_info.data_type is str
    assert field_info.is_list
    assert field_info.list_origin_type is str

def test_optional_list():
    field_info = build_field_info("optional_list")
    assert field_info.name == "optional_list"
    assert field_info.is_optional
    assert field_info.sql_type == "TEXT[]"
    assert field_info.data_type is str
    assert field_info.is_list
    assert field_info.list_origin_type is str


def test_model_list():
    field_info = build_field_info("model_list")
    assert field_info.name == "model_list"
    assert field_info.sql_type == "JSONB"
    assert field_info.data_type is TestField
    assert field_info.is_list
    assert field_info.list_origin_type is TestField


def test_dict_field():
    field_info = build_field_info("dict_field")
    assert field_info.name == "dict_field"
    assert field_info.sql_type == "JSONB"
    assert field_info.data_type is dict
    assert field_info.is_list is False
    assert field_info.list_origin_type is None


def test_optional_dict():
    field_info = build_field_info("optional_dict")
    assert field_info.name == "optional_dict"
    assert field_info.is_optional
    assert field_info.sql_type == "JSONB"
    assert field_info.data_type is dict
    assert field_info.is_list is False
    assert field_info.list_origin_type is None


def test_model_dict():
    field_info = build_field_info("model_dict")
    assert field_info.name == "model_dict"
    assert field_info.sql_type == "JSONB"
    assert field_info.data_type is dict
    assert field_info.is_list is False
    assert field_info.list_origin_type is None



def test_literal_field():
    field_info = build_field_info("literal_field")
    assert field_info.sql_type == "literal_field"
    assert field_info.data_type is Literal
    assert field_info.is_literal
    assert field_info.enum_name == "literal_field"
    assert field_info.get_enum_values_safe() == ["user", "assistant"]

def test_enum_field():
    field_info = build_field_info("enum_field")
    assert field_info.sql_type == "TestEnum"
    assert field_info.data_type is TestEnum
    assert field_info.is_enum
    assert field_info.is_literal is False
    assert field_info.enum_name == "TestEnum"
    assert field_info.get_enum_values_safe() == ["user", "assistant"]
