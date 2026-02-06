# chatboard/model/base/relation_info.py
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, ForwardRef

from .db_types import ContainerType

if TYPE_CHECKING:
    from .model3 import Model
    from .base.base_namespace import BaseNamespace

class RelationInfo:
    def __init__(
        self,
        name: str,
        primary_key: str,
        foreign_key: str,
        primary_cls: Type["Model"],
        foreign_cls: Any,  # may be a type, str, or ForwardRef
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        is_one_to_one: bool = False,
        relation_model: Optional[Type["Model"]] = None,
        junction_keys: Optional[list[str]] = None,
        is_reverse: bool = False,
        container_type: Optional[ContainerType] = None
    ):
        self.name = name
        if primary_key is None:
            raise ValueError(f"primary_key is required for relation {name}")
        if foreign_key is None:
            raise ValueError(f"foreign_key is required for relation {name}")
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self.primary_cls = primary_cls
        self._foreign_cls_raw = foreign_cls
        self._foreign_cls_resolved: Optional[Type["Model"]] = (
            foreign_cls if isinstance(foreign_cls, type) else None
        )
        self.on_delete = on_delete
        self.on_update = on_update
        self.is_one_to_one = is_one_to_one
        self.relation_model = relation_model
        self.junction_keys = junction_keys or []
        self.is_reverse = is_reverse
        self.container_type = container_type
        
    @property
    def foreign_cls(self) -> Type["Model"]:
        if self._foreign_cls_resolved is None:
            raise RuntimeError(
                f"Relation '{self.name}' target model not resolved yet"
            )
        return self._foreign_cls_resolved
    
    @property
    def foreign_namespace(self) -> "BaseNamespace":
        return self.foreign_cls.get_namespace()
    
    @property
    def primary_namespace(self) -> "BaseNamespace":
        return self.primary_cls.get_namespace()
    
    @property
    def relation_namespace(self) -> "BaseNamespace":
        if self.relation_model is None:
            raise ValueError(f"No relation model for {self.name}")
        return self.relation_model.get_namespace()

    @property
    def is_many_to_many(self) -> bool:
        return self.relation_model is not None
    
    @property
    def type(self) -> Literal["one_to_one", "one_to_many", "many_to_many"]:
        if self.is_many_to_many:
            return "many_to_many"
        elif self.is_one_to_one:
            return "one_to_one"
        else:
            return "one_to_many"
    

    def resolve_foreign_cls(self, globalns: dict[str, Any]):
        from typing import ForwardRef
        import typing
        if self._foreign_cls_resolved:
            return
        raw = self._foreign_cls_raw
        if isinstance(raw, ForwardRef):
            self._foreign_cls_resolved = raw._evaluate(globalns, None, set())
        elif isinstance(raw, str):
            self._foreign_cls_resolved = typing.ForwardRef(raw)._evaluate(globalns, None, set())
        elif isinstance(raw, type):
            self._foreign_cls_resolved = raw
        else:
            raise ValueError(f"Cannot resolve foreign_cls for relation '{self.name}'")

    
    
    def create_junction(self, primary_key, foreign_key, **kwargs) -> "Model":
        if not self.is_many_to_many or not self.relation_model:
            raise ValueError(f"Cannot create junction for non-many-to-many relation '{self.name}'")
        kwargs.update({
            self.junction_keys[0]: primary_key,
            self.junction_keys[1]: foreign_key,
        })
        return self.relation_model(**kwargs)
    
    
    def get_foreign_key_value(self, primary_model: "Model") -> Any:
        return {self.foreign_key: getattr(primary_model, self.primary_key)}
    
        