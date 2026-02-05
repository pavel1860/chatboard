from pydantic import BaseModel, PrivateAttr
from typing import TYPE_CHECKING, Any, Type, Self, TypeVar, Generic, runtime_checkable, Protocol

from promptview.model.base.types import VersioningStrategy





from .model_meta import ModelMeta
if TYPE_CHECKING:
    # from .postgres2.pg_query_set import PgSelectQuerySet
    from .sql2.pg_query_builder import PgQueryBuilder
    from .base.base_namespace import BaseNamespace
    from .relation_info import RelationInfo

MODEL = TypeVar("MODEL", bound="Model")
JUNCTION_MODEL = TypeVar("JUNCTION_MODEL", bound="Model")

@runtime_checkable
class Modelable(Protocol, Generic[MODEL]):
    def to_model(self) -> MODEL:
        ...
        
    @classmethod
    def query(cls) -> "PgQueryBuilder[MODEL]":
        ...


class Model(BaseModel, metaclass=ModelMeta):
    """Base class for all ORM models."""

    _is_base: bool = True
    _db_type: str = "postgres"
    _namespace_name: str = PrivateAttr(default=None)
    _is_versioned: bool = PrivateAttr(default=False)
    _ctx_token: Any = PrivateAttr(default=None)
    _versioning_strategy: VersioningStrategy = VersioningStrategy.NONE
    # ...add other ORM-internal attrs as needed...

    @classmethod
    def get_namespace_name(cls) -> str:
        """Get the namespace name for this model."""
        # In Pydantic's PrivateAttr, the value is stored on the class as a plain attribute
        val = getattr(cls, "_namespace_name", None)
        if isinstance(val, str):
            return val
        raise ValueError(f"Namespace name not set for {cls.__name__}")

    @classmethod
    def get_namespace(cls) -> "BaseNamespace":
        from .namespace_manager2 import NamespaceManager
        return NamespaceManager.get_namespace(cls.get_namespace_name())

    @classmethod
    async def initialize(cls):
        """Create DB table/collection for this model."""
        await cls.get_namespace().create_namespace()
        
    @classmethod
    def is_saved(cls) -> bool:
        return cls.primary_id is not None
        
    def __enter__(self):
        """Enter the context"""
        ns = self.get_namespace()
        self._ctx_token = ns.set_ctx(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context"""
        ns = self.get_namespace()
        ns.set_ctx(None)
        self._ctx_token = None
        
    @classmethod
    def current(cls) -> Self:
        obj = cls.get_namespace().get_ctx()
        if obj is None:
            raise ValueError(f"{cls.__name__} not found in context")
        return obj
    
    @classmethod
    def current_or_none(cls) -> Self | None:
        return cls.get_namespace().get_ctx()
    
    @classmethod
    def resolve_target_id_or_none(cls, target: "Model | int | None", use_ctx: bool = True) -> int | None:
        if target is None and use_ctx:
            curr = cls.current_or_none()
            return curr.primary_id if curr else None
        elif isinstance(target, int):
            return target
        elif isinstance(target, Model):
            return target.primary_id
        else:
            return None
        
    @classmethod
    def resolve_target_id(cls, target: "Model | int | None", use_ctx: bool = True) -> int:
        res = cls.resolve_target_id_or_none(target, use_ctx)
        if res is None:
            raise ValueError(f"Could not resolve id for target for {cls.__name__}: {target}")
        return res
    
    @classmethod
    async def get(cls: Type[Self], id: Any) -> Self:
        res = await cls.query().where(id=id).one()
        if res is None:
            raise ValueError(f"{cls.__name__} with ID '{id}' not found")
        return res
    # @classmethod
    # async def get(cls: Type[Self], id: Any) -> Self:
    #     data = await cls.get_namespace().get(id)
    #     if not data:
    #         raise ValueError(f"{cls.__name__} with ID '{id}' not found")
    #     return cls(**data)
    
    @classmethod
    async def get_or_none(cls: Type[Self], id: Any) -> Self | None:
        return await cls.query().where(id=id).one()
    
    # @classmethod
    # async def get_or_none(cls: Type[Self], id: Any) -> Self | None:
    #     data = await cls.get_namespace().get(id)
    #     if not data:
    #         return None
    #     return cls(**data)


    # async def save(self, *args, **kwargs) -> Self:
        
    #     ns = self.get_namespace()
        
        
    #     pk_value = getattr(self, ns.primary_key, None)
        
    #     for field in ns.iter_fields():
    #         if field.is_foreign_key and getattr(self, field.name) is None:
    #             fk_cls = field.foreign_cls
    #             if fk_cls:
    #                 ctx_instance = fk_cls.get_namespace().get_ctx()
    #                 if ctx_instance:
    #                     setattr(self, field.name, ctx_instance.primary_id)

    #     dump = self.model_dump()

    #     if pk_value is None:
    #         # Insert new record
    #         result = await ns.insert(dump)
    #     else:
    #         # Update existing record
    #         result = await ns.update(pk_value, dump)

    #     for key, value in result.items():
    #         setattr(self, key, value)
    #     return self
    
    def insert(self):
        ns = self.get_namespace()
        return ns.insert(self.model_dump()).select("*")
    
    def update(self) -> Self:
        ns = self.get_namespace()
        return ns.update(self.primary_id, self.model_dump()).select("*").one()
    
    @classmethod
    async def update_query(cls, id: Any, data: dict[str, Any]) -> Self:
        """Update the model instance"""
        ns = cls.get_namespace()
        return await ns.update(id, data).select("*").one()
    
    def _load_context_vars(self):
        ns = self.get_namespace()
        for field in ns.iter_fields():
            if field.is_foreign_key and getattr(self, field.name) is None:
                fk_cls = field.foreign_cls
                if fk_cls:
                    ctx_instance = fk_cls.get_namespace().get_ctx()
                    if ctx_instance:
                        setattr(self, field.name, ctx_instance.primary_id)


    
    def transform(self) -> str:
        """
        Transform the model into text for vectorization.
        Override this method to customize how your model is converted to text.
        Default: returns JSON dump of the model.
        """
        return self.model_dump_json()

    async def save(self):
        ns = self.get_namespace()
        self._load_context_vars()

        # Prepare model dump
        dump = self.model_dump()

        # Handle vectorization if needed
        if ns.need_to_transform:
            # Check which vector fields need to be generated
            fields_to_vectorize = []
            for field_name in ns._vectorizers.keys():
                if dump.get(field_name) is None:
                    fields_to_vectorize.append(field_name)

            # Only vectorize if there are fields that need it
            if fields_to_vectorize:
                # For each field that needs vectorization, call its transformer or default transform
                for field_name in fields_to_vectorize:
                    # Check if there's a custom transformer for this field
                    if field_name in ns._transformers:
                        transform_func, vectorizer_cls = ns._transformers[field_name]
                        text_content = transform_func(self)
                    else:
                        text_content = self.transform()

                    # Get the vectorizer for this field and embed
                    vectorizer = ns._vectorizers[field_name]
                    vector = await vectorizer.embed_query(text_content)
                    dump[field_name] = vector

        pk_value = self.primary_id
        if pk_value is None:
            # Insert with vectors
            result = await ns.insert(dump).select("*").one().json()
        else:
            # Update with vectors
            result = await ns.update(pk_value, dump).select("*").one().json()

        for key, value in result.items():
            setattr(self, key, value)
        return self
    
    
    
    async def add(self, model: MODEL | Modelable[MODEL], should_append: bool = True, **kwargs) -> MODEL:
        """Add a model instance to the database"""
        ns = self.get_namespace()
        if isinstance(model, Modelable):
            model = model.to_model()
        relation = ns.get_relation_by_type(model.__class__)
        if not relation:
            raise ValueError(f"Relation model not found for type: {model.__class__.__name__}")
        if relation.is_many_to_many:
            result = await model.save()
            junction = relation.create_junction(
                primary_key=self.primary_id, 
                foreign_key=result.primary_id, 
                **kwargs
            )
            junction = await junction.save()
        else:
            key = getattr(self, relation.primary_key)
            setattr(model, relation.foreign_key, key)
            result = await model.save()
            
        field = getattr(self, relation.name)
        if should_append:
            if relation.is_one_to_one:
                setattr(self, relation.name, result)
            else:
                if field is None:
                    field = self._resulve_container_default(relation)
                    setattr(self, relation.name, field)
                field.append(result)
        return result
    
    def _resulve_container_default(self, relation: "RelationInfo"):
        from typing import List
        from .db_types import Tree
        if relation.container_type in (List, list):
            return []
        elif relation.container_type is Tree:
            return Tree([], relation)
        else:
            raise ValueError(f"Unsupported container type: {relation}")
        
    
    
    async def include(self, field: str, limit: int | None = None) -> Self:
        ns = self.get_namespace()
        relation = ns.get_relation(field)
        if not relation:
            raise ValueError(f"Relation model not found for type: {field}")
        if relation.is_one_to_one:
            result = await relation.foreign_cls.query().where(**{relation.foreign_key: getattr(self, relation.primary_key)}).one()
        # elif relation.is_many_to_many:
        #     result = await relation.foreign_cls.query().where(**{relation.foreign_key: self.primary_id})
        else:
            query = relation.foreign_cls.query().where(**{relation.foreign_key: getattr(self, relation.primary_key)})            
            if limit:
                query = query.limit(limit)
            result = await query
        setattr(self, field, result)
        return self
        
        
        
        
        


    async def delete(self):
        return await self.get_namespace().delete(self.primary_id)
    
    @classmethod
    def _get_context_fields(cls):
        ns = cls.get_namespace()
        where_keys = {}
        for field in ns.iter_fields():
            if field.is_foreign_key:
                if curr:= field.foreign_cls.current_or_none():
                    where_keys[field.name] = curr.primary_id
        return where_keys

    @property
    def primary_id(self):
        ns = self.get_namespace()
        return getattr(self, ns.primary_key)
    

    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        **kwargs
    ) -> "PgQueryBuilder[Self]":
        # from .postgres2.pg_query_set import PgSelectQuerySet
        # query = PgSelectQuerySet(cls, alias=alias).select(*fields if fields else "*")
        from .sql2.pg_query_builder import PgQueryBuilder
        # query = select(cls)
        return PgQueryBuilder().select(cls)
        
        
    @classmethod
    def q(cls) -> "PgQueryBuilder[Self]":
        return cls.query()