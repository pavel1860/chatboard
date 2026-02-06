import uuid
from pydantic._internal._model_construction import ModelMetaclass
from pydantic import PrivateAttr, create_model, ConfigDict, BaseModel, Field
from pydantic.fields import _Unset, AliasPath, AliasChoices, FieldInfo, JsonDict, Unpack, _EmptyKwargs, Deprecated # type: ignore
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Generic, List, Literal, Optional, Protocol, Self, Type, TypeVar, get_args, get_origin
from pydantic_core import PydanticUndefined

from ..algebra.vectors.base_vectorizer import BaseVectorizer
from ..algebra.vectors.empty_vectorizer import EmptyVectorizer

if TYPE_CHECKING:
    from .model3 import Model







def ModelField(
    default: Any = PydanticUndefined,
    *,
    foreign_key: bool = False,    
    index: Optional[str] = None,
    default_factory: Callable[[], Any] | None = _Unset,
    order_by: bool = False,
    db_type: str | None = None,
    description: str | None = _Unset,
    foreign_cls: "Type[Model] | None" = None,
    self_ref: bool = False,
    rel_name: str | None = None,
    enforce_foreign_key: bool = True,
) -> Any:
    """Define a model field with ORM-specific metadata"""
    # Create extra metadata for the field
    extra = {}
    extra["is_model_field"] = True
    extra["order_by"] = order_by
    extra["index"] = index    
    extra["self_ref"] = self_ref
    extra["rel_name"] = rel_name
    extra["enforce_foreign_key"] = enforce_foreign_key
    if rel_name and not foreign_key:
        raise ValueError("rel_name can only be set on foreign_key fields")
    if db_type:
        extra["db_type"] = db_type
    if foreign_key:
        extra["foreign_key"] = True
        default = None
        
    if foreign_cls:
        if not foreign_key:
            raise ValueError("foreign_key must be provided if foreign_cls is provided")
        if self_ref:
            raise ValueError("self_ref must be False if foreign_cls is provided")
        extra["foreign_cls"] = foreign_cls
        default = None
    
    # Create the field with the extra metadata
    params = {
        "json_schema_extra": extra,
        "description": description,
    }
    if default_factory and default_factory != _Unset:
        params["default_factory"] = default_factory
    return Field(default, **params)


def KeyField(
    default: Any = None,
    # *,
    default_factory: Callable[[], Any] | Callable[[dict[str, Any]], Any] | None = _Unset,
    primary_key: bool = False,
    order_by: bool = False,
    type: Literal["int", "uuid"] = "int",
    description: str | None = _Unset,
    # **kwargs
) -> Any:
    """Define a key field with ORM-specific metadata"""
    # Create extra metadata for the field
    extra = {}
    extra["is_model_field"] = True
    extra["primary_key"] = primary_key
    extra["type"] = type
    extra["is_key"] = True
    extra["order_by"] = order_by
    # if type == "uuid" and not primary_key:
        # return Field(default_factory=lambda: uuid.uuid4(), json_schema_extra=extra, description=description)
    # Create the field with the extra metadata
    params = {
        "json_schema_extra": extra,
        "description": description,
    }
    if default_factory and default_factory != _Unset:
        params["default_factory"] = default_factory
        return Field(**params)
    else:
        return Field(default, **params)


def RefField(
    default: Any = None,
    *,
    key: str = "id",
    description: str | None = _Unset,
    # **kwargs
) -> Any:
    """Define a reference field with ORM-specific metadata"""
    # Create extra metadata for the field
    extra = {}
    extra["is_model_field"] = True
    extra["key"] = key
    
    return Field(default, json_schema_extra=extra, description=description)

def RelationField(
    default: Any = None,
    *,
    foreign_key: str,
    primary_key: str | None = None,    
    junction_keys: list[str] | None = None,
    junction_model: "Type[Model] | None" = None,
    on_delete: str = "CASCADE",
    on_update: str = "CASCADE",
    description: str | None = _Unset,
    name: str | None = None,
    # **kwargs
) -> Any:
    """
    Define a relation field with ORM-specific metadata.
    
    Args:
        primary_key: The name of the primary key in the related model
        foreign_key: The name of the foreign key in the related model
        junction_keys: The names of the junction keys in the related model
        on_delete: The action to take when the referenced row is deleted
        on_update: The action to take when the referenced row is updated
    """
    # Create extra metadata for the field
    

    # if not default:
        # default = Relation()
    
    extra = {}
    extra["is_model_field"] = True
    extra["is_relation"] = True
    extra["primary_key"] = primary_key
    extra["foreign_key"] = foreign_key
    extra["junction_keys"] = junction_keys
    extra["on_delete"] = on_delete
    extra["on_update"] = on_update
    extra["junction_model"] = junction_model
    extra["name"] = name
    
    if junction_keys:
        if not primary_key or not foreign_key:
            raise ValueError("primary_key and foreign_key must be provided if junction_keys are provided")
        if not junction_model:
            raise ValueError("junction_model must be provided if junction_keys are provided")
    
    # Create the field with the extra metadata and make it optional
    return Field(
        default, 
        json_schema_extra=extra, 
        description=description, 
        # exclude=True,
        # exclude_none=True
    )





def VectorField(
    default: Any = None,
    *,
    dimension: int | None = None,
    vectorizer: Type[BaseVectorizer] | None = None,
    distance: Literal["cosine", "euclid"] = "cosine",
    description: str | None = _Unset,
) -> Any:
    extra = {}

    extra["is_model_field"] = True
    extra["dimension"] = dimension
    extra["is_vector"] = True    
    extra["vectorizer"] = vectorizer
    extra["distance"] = distance 
    return Field(default, json_schema_extra=extra, description=description)






def BlockField(
    default: Any = None,
    *,
    description: str | None = _Unset,
) -> Any:
    extra = {}
    extra["is_model_field"] = True
    extra["is_block"] = True
    return Field(default, json_schema_extra=extra, description=description)