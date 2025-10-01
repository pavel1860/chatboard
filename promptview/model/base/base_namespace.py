from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Literal, Optional, Self, Type, TypeVar, runtime_checkable, Protocol
import contextvars
import uuid

from promptview.model.base.types import VersioningStrategy

from .base_field_info import BaseFieldInfo
from ..relation_info import RelationInfo

if TYPE_CHECKING:
    from ..model3 import Model
    from ..field_parser import FieldParser
    from ..relation_parser import RelationParser

MODEL = TypeVar("MODEL", bound="Model")
FIELD = TypeVar("FIELD", bound=BaseFieldInfo)


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> dict[str, Any]:
        ...

    def deserialize(self, data: dict[str, Any]) -> Self:
        ...


@dataclass
class RelationPlan:
    name: str
    kind: Literal["direct","m2m"]
    rel_type: Literal["one_to_one","one_to_many","many_to_many"]
    outer_pk: str
    target_fk: str
    target_table: str
    target_pk: str
    junction_table: Optional[str] = None
    j_outer_key: Optional[str] = None
    j_target_key: Optional[str] = None
    
    



class BaseNamespace(Generic[MODEL, FIELD]):
    def __init__(self, name: str, db_type: str, versioning_strategy: VersioningStrategy = VersioningStrategy.NONE):
        self.name = name
        self.db_type = db_type
        self._fields: dict[str, FIELD] = {}
        self._relations: dict[str, RelationInfo] = {}
        self._model_cls: Optional[Type[MODEL]] = None
        self._ctx_model = contextvars.ContextVar(f"{name}_ctx", default=None)
        self._primary_key_field: Optional[FIELD] = None
        # Pending parsers for deferred execution
        self._pending_field_parser: Optional["FieldParser"] = None
        self._pending_relation_parser: Optional["RelationParser"] = None
        self._default_order_field = None
        self._versioning_strategy = versioning_strategy
    # -------------------------
    # Model class binding
    # -------------------------
    def set_model_class(self, model_cls: Type[MODEL]):
        self._model_cls = model_cls
        
    @classmethod
    async def install_extensions(cls):
        raise NotImplementedError("Install extensions is not supported for this backend.")

    # -------------------------
    # Field management
    # -------------------------
    def add_field(self, field: FIELD):
        if field.name in self._fields:
            raise ValueError(f"Field '{field.name}' already exists in '{self.name}' table")
        if field.is_primary_key:
            if self._primary_key_field:
                raise ValueError(f"Primary key already defined: {self._primary_key_field.name}")
            self._primary_key_field = field
        if field.order_by:
            self._default_order_field = field
        self._fields[field.name] = field
        
    @property
    def primary_key(self) -> str:
        return self.primary_key_field.name
    
    @property
    def primary_key_field(self) -> FIELD:
        if self._primary_key_field is None:
            raise ValueError(f"No primary key defined for namespace '{self.name}'")
        return self._primary_key_field
    
    @property
    def default_order_field(self) -> str:
        if self._default_order_field is None:
            raise ValueError(f"No default order field defined for namespace '{self.name}'")
        return self._default_order_field.name

    def get_field(self, name: str) -> FIELD:
        return self._fields[name]
    
    def make_field_info(self, **kwargs) -> FIELD:
        """Factory method to create backend-specific FieldInfo."""
        raise NotImplementedError

    def iter_fields(self):
        return self._fields.values()

    def get_field_names(self) -> list[str]:
        return list(self._fields.keys())

    # -------------------------
    # Relation management
    # -------------------------
    def add_relation(
        self,
        name: str,
        primary_key: str,
        foreign_key: str,
        foreign_cls: Type[MODEL],
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        is_one_to_one: bool = False,
        relation_model: Optional[Type[MODEL]] = None,
        junction_keys: Optional[list[str]] = None,
        is_reverse: bool = False
    ) -> RelationInfo:
        if not self._model_cls:
            raise ValueError("Model class must be set before adding a relation")
        if name in self._relations:
            raise ValueError(f"Relation {name} already exists in {self.name}")
        rel_info = RelationInfo(
            name=name,
            primary_key=primary_key,
            foreign_key=foreign_key,
            primary_cls=self._model_cls,
            foreign_cls=foreign_cls,
            on_delete=on_delete,
            on_update=on_update,
            is_one_to_one=is_one_to_one,
            relation_model=relation_model,
            junction_keys=junction_keys,
            is_reverse=is_reverse
        )
        self._relations[name] = rel_info
        return rel_info
    
    
    def has_field(self, name: str) -> bool:
        return name in self._fields
    
    def has_relation(self, name: str) -> bool:
        return name in self._relations

    def get_relation(self, name: str) -> Optional[RelationInfo]:
        return self._relations.get(name)

    def get_relation_by_type(self, cls: Type[MODEL]) -> Optional[RelationInfo]:
        for rel in self._relations.values():
            if rel.foreign_cls == cls:
                return rel
        return None
    
    def get_relation_for_namespace(self, namespace: "BaseNamespace", foreign_key: str | None = None) -> Optional[RelationInfo]:
        for rel in self._relations.values():
            if rel.foreign_cls == namespace._model_cls:
                if foreign_key is not None and rel.primary_key != foreign_key:
                    continue
                return rel
        return None
    

    # -------------------------
    # Context handling
    # -------------------------
    def set_ctx(self, model: MODEL):
        return self._ctx_model.set(model)

    def get_ctx(self) -> Optional[MODEL]:
        return self._ctx_model.get()
    
        
    def reset_ctx(self, token):
        """
        Restore the previous context model using the token from set_ctx().
        """
        self._ctx_model.reset(token)

    # -------------------------
    # Model instantiation
    # -------------------------
    def instantiate_model(self, data: dict[str, Any]) -> MODEL:
        if not self._model_cls:
            raise ValueError("Model class not set")
        return self._model_cls.from_dict(data)

    def validate_model_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        # Add foreign key default resolution here if needed
        return data

    # -------------------------
    # Deferred parsing
    # -------------------------
    def set_pending_parsers(self, field_parser: "FieldParser", relation_parser: "RelationParser"):
        self._pending_field_parser = field_parser
        self._pending_relation_parser = relation_parser

    # def run_pending_parsers(self):
    #     """Run deferred field & relation parsing."""
    #     if self._pending_field_parser:
    #         self._pending_field_parser.parse()
    #         # self._pending_field_parser = None
    #     if self._pending_relation_parser:
    #         self._pending_relation_parser.parse()
    #         self._pending_relation_parser = None
            
    #     if self._pending_field_parser:
    #         self._pending_field_parser.validate_foreign_keys()
    #         self._pending_field_parser = None
    
    def build_temp_relation(self, name: str, target_ns: "BaseNamespace", on: tuple[str, str]) -> RelationInfo:
        if not self._model_cls:
            raise ValueError("Model class not set")
        if not target_ns._model_cls:
            raise ValueError("Target model class not set")
        rel = RelationInfo(
            name=name,
            primary_key=on[0],
            foreign_key=on[1],
            primary_cls=self._model_cls,
            foreign_cls=target_ns._model_cls,
            on_delete="CASCADE",
            on_update="CASCADE",
            is_one_to_one=False,
            relation_model=None,
            junction_keys=None,
        )
        return rel
    
    def plan_relation(self, target_ns: "BaseNamespace") -> "RelationPlan":
        rel = self.get_relation_for_namespace(target_ns)
        if not rel:
            raise ValueError(f"No relation between {self.name} and {target_ns.name}")

        # normalize so OUTER uses its PK and TARGET exposes its FK
        if rel.primary_cls == self._model_cls:
            outer_pk, target_fk = rel.primary_key, rel.foreign_key
        else:
            outer_pk, target_fk = rel.foreign_key, rel.primary_key

        if rel.is_many_to_many:
            jns = rel.relation_model.get_namespace()
            return RelationPlan(
                name=rel.name,
                kind="m2m",
                rel_type="many_to_many",
                outer_pk=outer_pk,
                target_fk=rel.foreign_key,          # FK on target model
                target_table=tgt_ns.name,
                target_pk=tgt_ns.primary_key,
                junction_table=jns.name,
                j_outer_key=rel.junction_keys[0],   # junction -> OUTER
                j_target_key=rel.junction_keys[1],  # junction -> TARGET
            )

        return RelationPlan(
            name=rel.name,
            kind="direct",
            rel_type=("one_to_one" if rel.is_one_to_one else "one_to_many"),
            outer_pk=outer_pk,
            target_fk=target_fk,
            target_table=tgt_ns.name,
            target_pk=tgt_ns.primary_key,
        )
        
        
    def generate_fake_key(self) -> Any:
        if self.primary_key_field.field_type == int:
            return 1
        elif self.primary_key_field.field_type == uuid.UUID:
            return uuid.uuid4()
        else:
            return "a"
        
    def has_primary_key(self, data:  MODEL) -> bool:
        return getattr(data, self.primary_key) is not None
    
    def get_primary_key(self, data: MODEL) -> Any:
        return getattr(data, self.primary_key)
    
    def set_primary_key(self, data: MODEL, value) -> None:
        setattr(data, self.primary_key, value)

    
            
    def __repr__(self):
        return f"<BaseNamespace {self.name}>"

    
    def query(self) -> Any:
        raise NotImplementedError("Querying is not supported for this backend.")