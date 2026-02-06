import inspect
import json
import uuid
import jsonref
from types import UnionType
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, TypedDict, Union, get_args, get_origin
from pydantic import BaseModel, Field, create_model
from enum import Enum, StrEnum
import datetime as dt


def is_list_type(pydantic_model):
    return get_origin(pydantic_model) is list


def unpack_list_model(pydantic_model):
    return get_args(pydantic_model)[0]


def get_list_type(model):
    return get_args(model)[0]



def is_optional_type(model):
    return get_origin(model) is Union and type(None) in get_args(model)


class Config:
    arbitrary_types_allowed = True



def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)
                
                
                
                
# class FunctionParameter(TypedDict, total=False):
#     type: str
#     description: Optional[str]
#     enum: Optional[List[str]]
#     properties: Optional[Dict[str, Any]]
#     required: Optional[List[str]]

# class FunctionSchema(TypedDict):
#     name: str
#     description: Optional[str]
#     parameters: Dict[str, Any]  # JSON Schema object

# class ToolFunction(TypedDict):
#     type: Literal["function"]
#     function: FunctionSchema

def schema_to_function(schema: Type[BaseModel]) -> dict[str, Any]:
    """convert a pydantic model to a function schema"""
    schema_dict = schema.model_json_schema()
    remove_a_key(schema_dict, "title")

    schema_dict = jsonref.replace_refs(schema_dict, proxies=False)
    remove_a_key(schema_dict, "$defs")

    return {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": schema.__doc__,
            "parameters": schema_dict,
        }
    }

def _schema_to_ts(value_type, indent: int = 2, depth: int = 0) -> str:
    """Convert Pydantic model to TypeScript type notation string."""

    # Handle basic types directly
    if value_type == str:
        return 'string'
    if value_type in (int, float):
        return 'number'
    if value_type == bool:
        return 'boolean'
    if value_type == type(None):
        return 'null'
    if value_type == None:
        return 'any'
    
    origin = get_origin(value_type)
    # Handle Literal types
    if origin == Literal:
        literal_args = get_args(value_type)
        # Convert each literal value to a string representation
        literal_values = []
        for arg in literal_args:
            if isinstance(arg, str):
                literal_values.append(f"'{arg}'")
            elif isinstance(arg, (int, float, bool)):
                literal_values.append(str(arg))
            else:
                literal_values.append(f"'{str(arg)}'")
        return ' | '.join(literal_values)
    
    # Handle Enum types
    if isinstance(value_type, type) and issubclass(value_type, Enum):
        # Convert enum values to TypeScript union type
        enum_values = []
        for v in value_type:
            if isinstance(v.value, str):
                enum_values.append(f"'{v.value}'")
            elif isinstance(v.value, (int, float, bool)):
                enum_values.append(str(v.value))
            else:
                enum_values.append(f"'{str(v.value)}'")
        return ' | '.join(enum_values)
    
    # Handle list types
    if origin == list:
        list_type_args = get_args(value_type)
        if list_type_args:
            item_ts_type = _schema_to_ts(list_type_args[0], indent, depth)
            return f'{item_ts_type}[]'
        return 'any[]'
    
    # Handle dict types
    if origin == dict:
        dict_type_args = get_args(value_type)
        if len(dict_type_args) == 2:
            key_type = _schema_to_ts(dict_type_args[0], indent, depth)
            value_type = _schema_to_ts(dict_type_args[1], indent, depth)
            # In TypeScript, only string, number, and symbol can be used as index types
            if key_type not in ['string', 'number']:
                key_type = 'string'
            return f'{{ [key: {key_type}]: {value_type} }}'
        return '{ [key: string]: any }'
    
    # Handle Union types
    if origin == UnionType or origin == Union:
        union_args = get_args(value_type)
        ts_types = [_schema_to_ts(arg, indent, depth) for arg in union_args]
        return ' | '.join(ts_types)
    
    # If not a Pydantic model, return any
    if not hasattr(value_type, 'model_fields'):
        return 'any'
    
    # Handle Pydantic models
    indent_str = ' ' * indent * depth
    fields_indent_str = indent_str + ' ' * indent
    fields = []
    for field_name, field in value_type.model_fields.items():
        field_type = field.annotation
        ts_type = _schema_to_ts(field_type, indent, depth + 1)
            
        # Add question mark for optional fields (those with default values or None default)
        is_optional = field.default is not None or field.default_factory is not None
        optional_marker = '?' if is_optional else ''
            
        # Add field description if available
        description = field.description or ''
        if description:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type}, // {description}')
        else:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type},')
            
    return '{\n' + '\n'.join(fields) + f'\n{indent_str}}}'

def schema_to_ts(value_type, indent: int = 2) -> str:
    return _schema_to_ts(value_type, indent, 0)

def make_optional(model: BaseModel) -> BaseModel:
    optional_fields = {k: (Optional[v], None) for k, v in model.__annotations__.items()}
    return create_model(
        model.__name__, 
        **optional_fields, 
        __config__=Config
    )




def make_optional_model(
    model: Type[BaseModel],
    exclude: Optional[Iterable[str]] = None,
) -> Type[BaseModel]:
    """
    Create a new Pydantic model where all fields are optional,
    preserving field metadata and constraints. You can exclude fields by name.

    Args:
        model (Type[BaseModel]): The original Pydantic model class
        exclude (Optional[Iterable[str]]): Field names to exclude from the new model

    Returns:
        Type[BaseModel]: A new model class with all (non-excluded) fields optional
    """
    fields: Dict[str, tuple[Any, Any]] = {}
    exclude_set = set(exclude or [])

    for field_name, model_field in model.model_fields.items():
        if field_name in exclude_set:
            continue

        # Make the type optional
        optional_type = Optional[model_field.annotation]
        
        # Create a copy of the field info to preserve metadata
        field_kwargs = {}
        
        # Copy field constraints and metadata
        if model_field.description:
            field_kwargs['description'] = model_field.description
        if model_field.title:
            field_kwargs['title'] = model_field.title
            
        # Get validation rules if they exist
        if hasattr(model_field, 'metadata') and model_field.metadata:
            validation_rules = model_field.metadata.get('validation_rules', {})
            
            # Numeric constraints
            if 'ge' in validation_rules:
                field_kwargs['ge'] = validation_rules['ge']
            if 'gt' in validation_rules:
                field_kwargs['gt'] = validation_rules['gt']
            if 'le' in validation_rules:
                field_kwargs['le'] = validation_rules['le']
            if 'lt' in validation_rules:
                field_kwargs['lt'] = validation_rules['lt']
            if 'multiple_of' in validation_rules:
                field_kwargs['multiple_of'] = validation_rules['multiple_of']
                
            # String constraints
            if 'min_length' in validation_rules:
                field_kwargs['min_length'] = validation_rules['min_length']
            if 'max_length' in validation_rules:
                field_kwargs['max_length'] = validation_rules['max_length']
            if 'pattern' in validation_rules:
                field_kwargs['pattern'] = validation_rules['pattern']
            
        # Additional metadata from json_schema_extra
        if model_field.json_schema_extra:
            field_kwargs.update(model_field.json_schema_extra)
            
        # Make all fields optional by setting default=None
        field = Field(default=None, **field_kwargs)
        
        fields[field_name] = (optional_type, field)

    # Create new model with optional fields
    optional_model = create_model(
        f"{model.__name__}Optional",
        **fields,
    )

    return optional_model



def get_model_fields(model_instance, model_class):
    fields = {field: getattr(model_instance, field) for field in model_class.__fields__.keys()}
    return model_class(**fields)



# def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):
#     for field, info in cls_.__fields__.items():
#         if sub_cls_filter is not None and (not inspect.isclass(info.annotation) or not issubclass(info.annotation, sub_cls_filter)):
#             continue
#         yield field, info


# def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):
#     for field, info in cls_.__fields__.items():
#         if sub_cls_filter is not None:
#             if not exclude and (inspect.isclass(info.annotation) and issubclass(info.annotation, sub_cls_filter)):
#                 yield field, info
#             if exclude and (not inspect.isclass(info.annotation) or not issubclass(info.annotation, sub_cls_filter)):
#                 yield field, info                
#             continue
#         yield field, info



def iterate_class_fields(cls_, sub_cls_filter=None, exclude=False):

    def _issubclass(cls_, sub_cls_filter):
        if type(sub_cls_filter) == list:
            return any([issubclass(cls_, sub_cls) for sub_cls in sub_cls_filter])
        return issubclass(cls_, sub_cls_filter)

    for field, info in cls_.__fields__.items():
        if sub_cls_filter is not None:
            if not exclude and (inspect.isclass(info.annotation) and _issubclass(info.annotation, sub_cls_filter)):
                yield field, info
            if exclude and (not inspect.isclass(info.annotation) or not _issubclass(info.annotation, sub_cls_filter)):
                yield field, info                
            continue
        yield field, info




def serialize_class(cls_: Any):
    output_type = "object"
    if is_list_type(cls_):
        output_type = "array"
        output_class = unpack_list_model(cls_)
    else:
        output_class = cls_
    if hasattr(output_class, 'model_json_schema'):
        schema = output_class.model_json_schema()
        version = 'v2'
    else:
        schema = schema_to_function(output_class)
        version = 'v1'
    return {
        "type": output_type,
        "schema": schema,
        "pydantic_version": version
    }






def describe_enum(enum_cls: Enum | StrEnum, delimiter: str = ", ") -> str:
    return delimiter.join([v.value for v in enum_cls])

def describe_literal(literal, delimiter="|"):
    args = get_args(literal)
    return delimiter.join(args)

def is_union_type(obj):
    orig = get_origin(obj)
    if orig and orig.__name__ == "UnionType":
        return True
    return hasattr(obj, "__origin__") and obj.__origin__ == Union




def get_type(arg, delimiter="|"):
    if isinstance(arg, type):
        if issubclass(arg, Enum):
            return describe_enum(arg, delimiter)
    elif get_origin(arg) is Literal:
        return describe_literal(arg, delimiter)
    return arg.__name__


def get_union_args(field_info, delimiter="|"):    
    union_args = get_args(field_info.annotation)
    type_args = []
    for arg in union_args:
        if arg == type(None):
            is_optional = True
        else:
            type_args.append(get_type(arg, delimiter))
    if is_optional:
        type_args.append("None")
    return delimiter.join(type_args)


def get_list_args(field_info, delimiter="|"):
    args = get_args(field_info.annotation)[0]
    return f"List[{stringify_field_info(args, delimiter)}]"

def get_field_info_origin(field_info, delimiter="|"):
    field_origin = get_origin(field_info.annotation)
    if field_origin == list:
        return get_list_args(field_info, delimiter)    
    else:
        return get_union_args(field_info.annotation)


def stringify_field_info(field_info, delimiter="|"):
    field_type = field_info.annotation if hasattr(field_info, "annotation") else field_info
    field_origin = get_origin(field_type)
    if field_origin == list:
        return get_list_args(field_info, delimiter)
    elif is_union_type(field_type):
        return get_union_args(field_info, delimiter)
    else:
        return get_type(field_type, delimiter)





def get_field_type(field_info):
    field_type = field_info.annotation if hasattr(field_info, "annotation") else field_info
    field_origin = get_origin(field_type)
    if field_origin == list:
        field_type = get_args(field_info.annotation)[0]
    elif field_origin == UnionType:
        field_type = get_args(field_info.annotation)
    return field_type, field_origin
    

def get_complex_fields(model_class):
    complex_fields = {}

    for field, field_info in model_class.model_fields.items():
        field_type, field_origin = get_field_type(field_info)
        if inspect.isclass(field_type):
            if issubclass(field_type, BaseModel):
                complex_fields[field] = field_type            
    return complex_fields





def make_json_serializable(data: dict[str, Any]):
    try:    
        for k, v in data.items():
            if isinstance(v, uuid.UUID):
                data[k] = str(v)
            elif isinstance(v, dt.datetime):
                data[k] = v.isoformat()
            elif isinstance(v, dict):
                data[k] = make_json_serializable(v)
            elif isinstance(v, list):
                data[k] = [make_json_serializable(item) if isinstance(item, dict) else item for item in v]
    except Exception as e:
        raise ValueError(f"Failed to serialize dict: {data}") from e
    return data
    
    
    
def make_json_string_deserializable(data_str: str):
    data = json.loads(data_str)
    ret_data = {}
    try:
        for k, v in data.items():
            if k.isdigit():
                ret_data[int(k)] = v
            else:
                ret_data[k] = v
    except Exception as e:
        raise ValueError(f"Failed to deserialize dict: {data}") from e
    return data