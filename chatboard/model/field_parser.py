# model/field_parser.py

from enum import Enum
import inspect
from typing import TYPE_CHECKING, Literal, Type, Any, Dict, Union, get_args, get_origin
from pydantic.fields import FieldInfo
from .util import unpack_extra


if TYPE_CHECKING:
    from .model3 import Model

def get_field_extras(field_info: FieldInfo) -> Dict[str, Any]:
    """Extract extra metadata from a Pydantic FieldInfo."""
    extra_json = field_info.json_schema_extra
    if isinstance(extra_json, dict):
        return dict(extra_json)
    return {}

class FieldParser:
    def __init__(self, model_cls, model_name, db_type, namespace, reserved_fields=None):
        self.model_cls = model_cls
        self.model_name = model_name
        self.db_type = db_type
        self.namespace = namespace
        self.reserved_fields = reserved_fields or set()
        self.field_extras = {}
        
        
    @classmethod
    def parse_enum(cls, field_type: type[Any]) -> tuple[bool, list[Any] | None, bool]:
        """Detect enum/Literal fields.
        Returns: (is_enum, values, is_literal)
        """
        if get_origin(field_type) is Literal:
            return True, list(get_args(field_type)), True
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return True, [e.value for e in field_type], False
        return False, None, False

    def parse(self):
        for field_name, field_info in self.model_cls.model_fields.items():
            extra = unpack_extra(field_info)
            if not extra.get("is_model_field", False):
                continue

            # Detect Optional[T] or T | None
            annotation = field_info.annotation
            origin = get_origin(annotation)
            args = get_args(annotation)
            if origin is Union and type(None) in args:
                extra["is_optional"] = True
                field_type = next(a for a in args if a is not type(None))
            elif len(args) > 1 and type(None) in args:
                extra["is_optional"] = True
                field_type = next(a for a in args if a is not type(None))
            else:
                field_type = annotation

            self.field_extras[field_name] = extra

            if extra.get("is_relation", False):
                continue

            if extra.get("is_vector", False):
                self._register_vector_field(field_name, field_type, field_info, extra)
                continue

            self._register_scalar_field(field_name, field_type, field_info, extra)
            
    def validate_foreign_keys(self):
        for field_info in self.model_cls.get_namespace().iter_fields():            
            if field_info.is_foreign_key and field_info.foreign_cls is None and not field_info.self_ref and field_info.enforce_foreign_key:
                raise ValueError(f"Foreign class is not set for field '{field_info.name}' on model '{self.model_cls.__name__}'. eather add relation to foreign class or set foreign_cls.")

    def _register_scalar_field(self, field_name, field_type, field_info, extra):
        # Build backend-specific FieldInfo via namespace
        is_enum, enum_values, is_literal = self.parse_enum(field_type)
        field_obj = self.namespace.make_field_info(
            name=field_name,
            field_type=field_type,
            default=field_info.default,
            is_optional=extra.get("is_optional", False),
            is_primary_key=extra.get("primary_key", False),
            is_foreign_key=extra.get("foreign_key", False),
            is_vector=False,
            index=extra.get("index", False),
            on_delete=extra.get("on_delete", "CASCADE"),
            on_update=extra.get("on_update", "CASCADE"),
            is_key=extra.get("is_key", False),
            enum_values=enum_values if is_enum else None,
            order_by=extra.get("order_by", False),
            foreign_cls=extra.get("foreign_cls", None),
            sql_type=extra.get("db_type", None),
            self_ref=extra.get("self_ref", False),
            rel_name=extra.get("rel_name", None),
            enforce_foreign_key=extra.get("enforce_foreign_key", True),
        )
        self.namespace.add_field(field_obj)

    def _register_vector_field(self, field_name, field_type, field_info, extra):
        """Register a vector field with vectorizer."""
        from .namespace_manager2 import NamespaceManager

        vectorizer = extra.get("vectorizer")
        if vectorizer is None:
            raise ValueError(f"vectorizer is required for vector field: {field_type} on Model {self.model_name}")

        dimension = extra.get("dimension")
        distance = extra.get("distance", "cosine")

        # Register vector field with namespace (gets vectorizer instance)
        vectorizer_instance = self.namespace.register_vector_field(field_name, vectorizer)
        dimension = dimension or vectorizer_instance.dimension

        # Create field info for vector
        field_obj = self.namespace.make_field_info(
            name=field_name,
            field_type=field_type,
            default=field_info.default,
            is_optional=extra.get("is_optional", False),
            is_primary_key=False,
            is_foreign_key=False,
            is_vector=True,
            dimension=dimension,
            distance=distance,
            index=extra.get("index", False),
            is_key=False,
            order_by=False,
        )
        self.namespace.add_field(field_obj)

        # Register vector extension for Postgres
        if self.db_type == "postgres":
            NamespaceManager.register_extension(self.db_type, "vector")
