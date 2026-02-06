



# chatboard/utils/model_field_utils.py

from typing import Any, Dict, ForwardRef, get_args, get_origin, List, Union
from pydantic.fields import FieldInfo
from .db_types import Tree

def unpack_extra(field_info: FieldInfo) -> dict[str, Any]:
    extra_json = field_info.json_schema_extra
    extra: Dict[str, Any] = {}
    if extra_json is not None:
        if isinstance(extra_json, dict):
            extra = dict(extra_json)
        elif callable(extra_json):
            extra = {}  # optionally call it if you want dynamic schema
    return extra







def resolve_annotation(ann, globalns):
    origin = get_origin(ann)
    args = get_args(ann)

    # Handle List[ForwardRef]
    if origin in (list, List, Tree) and args:
        resolved_args = tuple(resolve_annotation(a, globalns) for a in args)
        return origin[resolved_args[0]]  # rebuild List[T] type

    # Handle Optional[T] or Union[T, None]
    if origin is Union and args:
        resolved_args = tuple(resolve_annotation(a, globalns) for a in args)
        return Union[resolved_args]

    # Handle plain ForwardRef
    if isinstance(ann, ForwardRef):
        return ann._evaluate(globalns, None, recursive_guard=frozenset())

    # Handle string forward ref
    if isinstance(ann, str):
        return ForwardRef(ann)._evaluate(globalns, None, recursive_guard=frozenset())
    
        # If it's NoneType outside a Union — treat as Any or raise
    if ann is type(None):
        return ann
        # You can choose to raise here if this shouldn't happen
        raise ValueError("Annotation resolved to NoneType without optional context")
        # Or: return Any

    # Already a concrete type
    return ann
