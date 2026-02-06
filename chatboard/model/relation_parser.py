from typing import TYPE_CHECKING, Type, Dict, Any, get_origin, get_args, List, Union
from pydantic.fields import FieldInfo

from .util import resolve_annotation

from .db_types import Tree
from .base.base_namespace import BaseNamespace
from .base.base_field_info import BaseFieldInfo
from .db_types import ContainerType

if TYPE_CHECKING:
    from .relation_info import RelationInfo
    from .model3 import Model


def get_relation_extras(field_info: FieldInfo) -> Dict[str, Any]:
    extra_json = field_info.json_schema_extra
    if isinstance(extra_json, dict):
        return dict(extra_json)
    return {}


class RelationParser:
    def __init__(self, model_cls: Type, namespace: BaseNamespace["Model", BaseFieldInfo]):
        self.model_cls = model_cls
        self.namespace = namespace
        self.relations: "Dict[str, RelationInfo]" = {}

    def parse(self):
        """Parse the relations for the model and instantiate all the foreign classes"""
        from .model3 import Model

        for field_name, field_info in self.model_cls.model_fields.items():
            if not isinstance(field_info, FieldInfo):
                continue

            extra = get_relation_extras(field_info)
            if not extra.get("is_relation", False):
                continue

            # --- 1) Unwrap Optional[T] / T | None ---
            annotation = field_info.annotation

            # Unwrap Optional[T] or T | None
            origin = get_origin(annotation)
            args = get_args(annotation)
            if origin is Union and type(None) in args:
                extra["is_optional"] = True
                annotation = next(a for a in args if a is not type(None))
            elif len(args) > 1 and type(None) in args:
                extra["is_optional"] = True
                annotation = next(a for a in args if a is not type(None))

            # Now detect if it's a list
            container_type = None
            origin = get_origin(annotation)
            if origin and issubclass(origin, ContainerType):
                container_type = origin
            args = get_args(annotation)
            # for a in args:
            #     if isinstance(a, ForwardRef):
            #         a = resolve_annotation(a, globalns)
            is_list = origin in (list, List)
            is_tree = origin in (Tree, Tree[Any])
            related_cls = args[0] if (is_list or is_tree) and args else annotation
            if not isinstance(related_cls, Type) or not issubclass(related_cls, Model):
                raise ValueError(f"Related class {related_cls} is not a subclass of Model")
            rel_ns = related_cls.get_namespace()

            junction_model = extra.get("junction_model")
            junction_keys = extra.get("junction_keys")

            # --- 3) Create the RelationInfo ---
            rel_info = self.namespace.add_relation(
                name=extra.get("name") or field_name,
                primary_key=extra.get("primary_key") or self.namespace.primary_key,
                foreign_key=extra.get("foreign_key") or rel_ns.primary_key,
                foreign_cls=related_cls,  # May be forward ref
                on_delete=extra.get("on_delete", "CASCADE"),
                on_update=extra.get("on_update", "CASCADE"),
                is_one_to_one=not is_list and not container_type,
                relation_model=junction_model,
                junction_keys=junction_keys,
                container_type=container_type
            )
            self.relations[field_name] = rel_info

            # --- 4) Set reverse FK metadata ---
            if junction_model and junction_keys:
                # Many-to-Many
                junction_ns = junction_model.get_namespace()
                if len(junction_keys) == 2:
                    # First key → current model
                    if junction_keys[0] in junction_ns._fields:
                        setattr(junction_ns._fields[junction_keys[0]], "foreign_cls", self.model_cls)
                    # Second key → related model
                    if junction_keys[1] in junction_ns._fields:
                        setattr(junction_ns._fields[junction_keys[1]], "foreign_cls", related_cls)
                else:
                    raise ValueError(f"junction_keys must be a list of length 2 for many-to-many relation: {junction_keys}")
            else:
                # One-to-One or One-to-Many
                fk_name = extra.get("foreign_key", "id")
                # The FK belongs to the related model, not self.model_cls
                try:
                    fk_ns = related_cls.get_namespace()
                except Exception:
                    # related_cls might be a ForwardRef or str; skip for now
                    fk_ns = None
                if fk_ns and fk_name in fk_ns._fields:
                    setattr(fk_ns._fields[fk_name], "foreign_cls", self.model_cls)
                    
                    
                    
                    
            # try:
            #     reverse_ns = related_cls.get_namespace()
            # except Exception:
            #     reverse_ns = None

            # if reverse_ns:
            #     reverse_name = extra.get("reverse_name") or f"{self.model_cls.__name__.lower()}s"
            #     if reverse_name not in reverse_ns._relations:
            #         if junction_model and junction_keys:
            #             # Reverse M:N
            #             reverse_ns.add_relation(
            #                 name=reverse_name,
            #                 primary_key=extra.get("foreign_key", "id"),
            #                 foreign_key=extra.get("primary_key", "id"),
            #                 foreign_cls=self.model_cls,
            #                 on_delete=extra.get("on_delete", "CASCADE"),
            #                 on_update=extra.get("on_update", "CASCADE"),
            #                 is_one_to_one=False,
            #                 relation_model=junction_model,
            #                 junction_keys=[junction_keys[1], junction_keys[0]],
            #                 is_reverse=True
            #             )
            #         else:
            #             # Reverse direct relation
            #             reverse_ns.add_relation(
            #                 name=reverse_name,
            #                 primary_key=extra.get("foreign_key", "id"),
            #                 foreign_key=extra.get("primary_key", "id"),
            #                 foreign_cls=self.model_cls,
            #                 on_delete=extra.get("on_delete", "CASCADE"),
            #                 on_update=extra.get("on_update", "CASCADE"),
            #                 is_one_to_one=not is_list,
            #                 is_reverse=True
            #             )
