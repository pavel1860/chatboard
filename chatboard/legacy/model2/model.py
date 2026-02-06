import contextvars
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Iterator, List, Literal, Optional, Protocol, Self, Set, Tuple, Type, TypeVar, Callable, cast, ForwardRef, get_args, get_origin, runtime_checkable
import uuid
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.config import JsonDict
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
import pydantic_core
import datetime as dt



from chatboard.model.context import Context
from chatboard.model.fields import KeyField, ModelField
from chatboard.model.namespace_manager import NamespaceManager
from chatboard.model.base_namespace import DatabaseType, Distance, NSFieldInfo, NSManyToManyRelationInfo, NSRelationInfo, Namespace, QuerySet, QuerySetSingleAdapter
from chatboard.model.postgres.query_set3 import SelectQuerySet
from chatboard.model.query_filters import SelectFieldProxy



from chatboard.model.relation import Relation
from chatboard.resource_manager import ResourceManager
from chatboard.utils.model_utils import unpack_list_model
from chatboard.utils.string_utils import camel_to_snake


if TYPE_CHECKING:
    from chatboard.algebra.vectors.base_vectorizer import BaseVectorizer

T = TypeVar('T', bound=BaseModel)



def get_dct_private_attributes(dct: dict[str, Any], key: str, default: Any=None) -> Any:
    """Get the private attributes of a class"""
    # res = dct['__private_attributes__'].get(key)
    res = dct.get(key)
    if res is not None and hasattr(res, "default"):
        res = res.default
    if not res:
        return default
    return res



def build_namespace(model_cls_name: str, db_type: DatabaseType = "postgres"):
    if db_type == "postgres":
        return f"{camel_to_snake(model_cls_name)}s"
    elif db_type == "neo4j":
        return model_cls_name
    elif db_type == "qdrant":
        return model_cls_name
    else:
        raise ValueError(f"Invalid database type: {db_type}")


ARTIFACT_RESERVED_FIELDS = ["id", "artifact_id", "version", "branch_id", "turn_id", "created_at", "updated_at", "deleted_at"]

AUTH_RESERVED_FIELDS = ["id", "name", "email", "emailVerified", "image", "is_admin"]


def check_reserved_fields(bases: Any, name: str, field_name: str):
    if bases[0].__name__ == "ArtifactModel":
        if field_name in ARTIFACT_RESERVED_FIELDS:
            raise ValueError(f"""Field "{field_name}" in ArtifactModel "{name}" is reserved for internal use. use a different name. {", ".join(ARTIFACT_RESERVED_FIELDS)} are reserved.""")
    elif bases[0].__name__ == "AuthModel":
        if field_name in AUTH_RESERVED_FIELDS:
            raise ValueError(f"""Field "{field_name}" in AuthModel "{name}" is reserved for internal use. use a different name. {", ".join(AUTH_RESERVED_FIELDS)} are reserved.""")

def check_if_artifact_model(bases: Any, name: str):
    if len(bases) >= 1 and bases[0].__name__ == "ArtifactModel":
        return True
    return False

def check_if_context_model(bases: Any, name: str):
    if len(bases) >= 1 and bases[0].__name__ == "ContextModel":
        return True
    return False


def unpack_extra(field_info: FieldInfo) -> dict[str, Any]:
    extra_json = field_info.json_schema_extra
    extra: Dict[str, Any] = {}
    
    # Convert json_schema_extra to a dictionary
    if extra_json is not None:
        if isinstance(extra_json, dict):
            extra = dict(extra_json)
        elif callable(extra_json):
            # If it's a callable, create an empty dict
            # In a real implementation, we might want to call it
            extra = {}
    return extra

def unpack_relation_extra(extra: dict[str, Any], field_origin: Type[Any], is_versioned: bool, is_artifact: bool) -> tuple[str, str, list[str] | None, str, Type[Any] | None, str, str]:
    primary_key = extra.get("primary_key") or ("artifact_id" if is_artifact else "id")
    foreign_key = extra.get("foreign_key")
    on_delete=extra.get("on_delete", "CASCADE")
    on_update=extra.get("on_update", "CASCADE")
    junction_keys = extra.get("junction_keys", None)
    junction_model = extra.get("junction_model", None)
    relation_type = "one_to_one"                
    if junction_keys is not None:
        if junction_model is not None:
            relation_type = "many_to_many"
        else:
            relation_type = "one_to_many"
    
    # if not foreign_key:
    #     raise ValueError("foreign_key is required for one_to_many relation")
    
    # print(primary_key, foreign_key, on_delete, on_update)
    # if not foreign_key:
    #     raise ValueError("foreign_key is required for one_to_many relation")
    
    return primary_key, foreign_key, junction_keys, relation_type, junction_model, on_delete, on_update


def unpack_vector_extra(extra: dict[str, Any]) -> "tuple[int, Type[BaseVectorizer] | None, Distance]":
    dimension = extra.get("dimension", None)
    vectorizer = extra.get("vectorizer", None)
    distance = Distance(extra.get("distance", "cosine"))
    
    return dimension, vectorizer, distance

class ModelMeta(ModelMetaclass, type):
    """Metaclass for Model
    
    This metaclass handles the registration of models with the namespace manager
    and the processing of fields.
    """
    
    def __new__(cls, name, bases, dct):
        # Use the standard Pydantic metaclass to create the class
        
        
        # Skip processing for the base Model class
        # if name == "Model" or name == "RepoModel" or name == "ArtifactModel" or name == "ContextModel" or name == "AuthModel" or name == "TurnModel":
        if dct.get("_is_base", False):
            cls_obj = super().__new__(cls, name, bases, dct)
            return cls_obj
        
        # Get model name and namespace
        model_name = name
        db_type = get_dct_private_attributes(dct, "_db_type", "postgres")
        namespace_name = get_dct_private_attributes(dct, "_namespace_name", build_namespace(model_name, db_type))
        
        
        # Check if this is -a repo model or an artifact model
        is_repo = len(bases) >= 1 and bases[0].__name__ == "RepoModel"
        is_artifact = check_if_artifact_model(bases, name)
        is_context = check_if_context_model(bases, name)
        # If _repo is specified, the model is automatically versioned
        is_versioned = get_dct_private_attributes(dct, "_is_versioned", False) or (len(bases) >= 1 and (bases[0].__name__ == "ArtifactModel" or bases[0].__name__ == "ContextModel" or bases[0].__name__ == "TurnModel"))
        repo_namespace = get_dct_private_attributes(dct, "_repo", "main" if is_versioned else None)
        if repo_namespace is None and is_versioned:
            raise ValueError("RepoModel must have a repo namespace")
        
        if is_repo and is_versioned:
            raise ValueError("RepoModel cannot be versioned")
        
        # Build namespace
        ns = NamespaceManager.build_namespace(
            model_name=namespace_name, 
            db_type=db_type, 
            is_versioned=is_versioned,
            is_context=is_context,
            is_repo=is_repo, 
            is_artifact=is_artifact, 
            repo_namespace=repo_namespace
        )
        # Process fields and relations
        relations = {}
        field_extras = {}
    
        for field_name, field_info in dct.items():            
            if callable(field_info) and hasattr(field_info, "_transformer_field_name"):
                field_name = getattr(field_info, "_transformer_field_name")
                vectorizer_cls = getattr(field_info, "_vectorizer_cls")
                # register it on the class
                ns.register_transformer(field_name, field_info, vectorizer_cls)
                # ResourceManager.register_vectorizer(field_name, vectorizer_cls)
        
        
        cls_obj = super().__new__(cls, name, bases, dct)
        
        for field_name, field_info in cls_obj.model_fields.items():
            # Skip fields without a type annotation
            # check_reserved_fields(bases, name, field_name)
            if not isinstance(field_info, FieldInfo):
                continue
            
            # field_type = dct["__annotations__"].get(field_name)
            field_type = field_info.annotation
            if field_type is None:
                continue
            
            extra = unpack_extra(field_info)
            if not extra.get("is_model_field", False):
                continue
            
            # Check if this is a relation field
            if extra.get("is_key", False) and extra.get("primary_key", False):
                if db_type == "postgres" and field_type == uuid.UUID:
                    NamespaceManager.register_extension(db_type, '"uuid-ossp"')
            
            if extra.get("is_relation", False):
                # Get the model class from the relation                                
                field_origin = get_origin(field_type)
                foreign_cls = None
                is_one_to_one = False
                if field_origin is Relation:                    
                    foreign_cls = get_relation_model(field_type)
                    if not issubclass(foreign_cls, Model) and not isinstance(foreign_cls, ForwardRef):
                        raise ValueError(f"foreign_cls must be a subclass of Model: {foreign_cls}")
                elif field_origin is None and issubclass(field_type, Model):
                    foreign_cls = field_type
                    is_one_to_one = True
                if not foreign_cls:
                    raise ValueError(f"foreign_cls is required for relation: {field_type} on Model {name}")
                             
                relation_field = ns.add_relation(
                    name=extra.get("name") or field_name,
                    primary_key=extra.get("primary_key") or "id",
                    foreign_key=extra.get("foreign_key") or "id",
                    foreign_cls=foreign_cls,
                    junction_keys=extra.get("junction_keys", None),
                    junction_cls=extra.get("junction_model", None),                    
                    on_delete=extra.get("on_delete") or "CASCADE",
                    on_update=extra.get("on_update") or "CASCADE",
                    is_one_to_one=is_one_to_one
                )    
                # primary_key, foreign_key, junction_keys, relation_type, junction_cls, on_delete, on_update = unpack_relation_extra(extra, field_origin, ns.is_versioned, ns.is_artifact)
                                
                # if relation_type == "one_to_one" or relation_type == "one_to_many":                    
                #     # foreign_cls = get_one_to_many_relation_model(field_type)                    
                #     relation_field = ns.add_relation(
                #         name=field_name,
                #         primary_key=primary_key,
                #         foreign_key=foreign_key,
                #         foreign_cls=foreign_cls,
                #         on_delete=on_delete,
                #         on_update=on_update,
                #     )

                # else:  # many to many relation
                #     if junction_cls is None:
                #         raise ValueError(f"junction_cls is required for many to many relation: {field_type} on Model {name}")
                #     if not junction_keys:
                #         raise ValueError(f"junction_keys is required for many to many relation: {field_type} on Model {name}")
                    
                #     relation_field = ns.add_many_relation(
                #         name=field_name,
                #         primary_key=primary_key,
                #         foreign_key=foreign_key,
                #         foreign_cls=foreign_cls,
                #         junction_cls=junction_cls,
                #         junction_keys=junction_keys,
                #         on_delete=on_delete,
                #         on_update=on_update,
                #     )                
                
                relations[field_name] = relation_field
                # Skip adding this field to the namespace
                continue
            elif extra.get("is_vector", False):
                # vector field, need to add it to the vectorizers
                dimension, vectorizer, distance = unpack_vector_extra(extra)
                if vectorizer is None:
                    raise ValueError(f"vectorizer is required for vector field: {field_type} on Model {name}")
                # if dimension is None:
                    # raise ValueError(f"dimension is required for vector field: {field_type} on Model {name}")
                # if not ns.get_transformer(field_name):
                    # raise ValueError(f"transformer is required for vector field: {field_type} on Model {name}")
                # ResourceManager.register_vectorizer(field_name, vectorizer)
                
                # transformer = ns.vector_fields.get_transformer(field_name)
                vectorizer = ns.register_vector_field(field_name, vectorizer)
                dimension = dimension or vectorizer.dimension
                ns.add_field(
                    field_name, 
                    field_type, 
                    dimension=dimension, 
                    distance=distance,
                    is_optional=False,
                    foreign_key=False,
                    is_key=False,
                    is_vector=True,
                    is_primary_key=False,
                    is_default_temporal=False,
                )
                NamespaceManager.register_extension(db_type, "vector")
            else:
                # ns.add_field(field_name, field_type, **extra)
                ns.add_field(
                    field_name, 
                    field_type, 
                    default=field_info.default,
                    is_optional=extra.get("is_optional", False),
                    foreign_key=extra.get("foreign_key", False),
                    is_key=extra.get("is_key", False),
                    is_vector=False,
                    is_primary_key=extra.get("primary_key", False),
                    is_default_temporal=extra.get("is_default_temporal", False),
                )
            
            # Add field to namespace
            field_extras[field_name] = extra
            # ns.add_field(field_name, field_type, extra)
        
        # Set namespace name and relations on the class
        
        
        
            # # if field_name in relations:
            # #     continue
            # extra = get_extra(field_info)            
            # field_type = field_info.annotation
            
            # if not extra.get("is_model_field", False) or extra.get("is_relation", False):
            #     continue
            # ns.add_field(field_name, field_type, extra)
        
        
        ns.set_model_class(cls_obj)
        cls_obj._namespace_name = namespace_name
        cls_obj._is_versioned = is_versioned

        
        return cls_obj
    
    # def __getattribute__(cls, name: str) -> Any:
    #     # print(cls, name)
    #     if NamespaceManager._is_initialized and name not in {"__name__", "model_fields", "__pydantic_fields__", "get_namespace", "get_namespace_name", "_namespace_name"}:        
    #         # print("entered")
    #         if name in cls.model_fields:
    #             ns = NamespaceManager.get_namespace_or_none(build_namespace(cls.__name__))
    #             if ns:
    #                 if not ns.has_field(name):
    #                     raise ValueError(f"Field {name} not found in namespace {ns.table_name}")
    #                 return SelectField(ns.get_field(name))
    #                 # return FieldComparable(name, cls.model_fields[name])
    #                 # print("table",ns.table_name, name)
    #             else:
    #                 raise ValueError(f"Namespace {build_namespace(cls.__name__)} not found")
    #     return super().__getattribute__(name)
    
    # def get_field(cls, name: str) -> SelectField:
    #     if name in cls.model_fields:
    #         ns = NamespaceManager.get_namespace_or_none(build_namespace(cls.__name__))
    #         if ns:
    #             if not ns.has_field(name):
    #                 raise ValueError(f"Field {name} not found in namespace {ns.table_name}")
    #             return SelectField(ns.get_field(name))                    
    #     raise ValueError(f"Namespace {build_namespace(cls.__name__)} not found")
    @property
    def f(cls: "Type[Model]") -> SelectFieldProxy:
        return SelectFieldProxy(cls, cls.get_namespace())
    
    
MODEL = TypeVar("MODEL", bound="Model", covariant=True)
JUNCTION_MODEL = TypeVar("JUNCTION_MODEL", bound="Model")


@runtime_checkable
class Modelable(Protocol, Generic[MODEL]):
    def to_model(self) -> MODEL:
        ...
        
    @classmethod
    def query(cls) -> "SelectQuerySet[MODEL]":
        ...

class Model(BaseModel, metaclass=ModelMeta):
    """Base class for all models
    
    This class is a simple Pydantic model with a custom metaclass.
    The ORM functionality is added by the metaclass.
    """
    # Namespace reference - will be set by the metaclass
    _is_base: bool = True
    _db_type: Literal["postgres", "qdrant", "neo4j"] = "postgres"
    _namespace_name: str = PrivateAttr(default=None)
    # _db_type: DatabaseType = PrivateAttr(default="postgres")
    _is_versioned: bool = PrivateAttr(default=False)
    _relations: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict) 
    _ctx_token: contextvars.Token | None = PrivateAttr(default=None)    
    
    
    @classmethod
    def get_namespace_name(cls) -> str:
        """Get the namespace name for this model"""
        return cls._namespace_name
    
    @classmethod
    def get_namespace(cls) -> Namespace:
        """Get the namespace for this model"""
        return NamespaceManager.get_namespace(cls.get_namespace_name())
    
    @classmethod
    def get_key_field(cls) -> str:
        """Get the key field for this model"""
        ns = cls.get_namespace()
        return ns.primary_key.name
    
    @property
    def primary_id(self) -> Any:
        """Get the key for this model"""
        return getattr(self, self.get_key_field())
    
    @classmethod
    def current(cls) -> MODEL:
        """Get the current model in context"""
        ns = cls.get_namespace()
        return ns.get_ctx()
    
    @classmethod
    def current_or_none(cls) -> MODEL | None:
        """Get the current model in context or None if it is not set"""
        ns = cls.get_namespace()
        return ns.get_ctx_or_none()
    
    @classmethod
    async def initialize(cls):
        """Initialize the model (create table)"""
        ns = cls.get_namespace()
        await ns.create_namespace()
    
    def _get_relation_fields(self):
        """Get the names of relation fields."""
        ns = self.get_namespace()
        return list(ns._relations.keys())
    
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
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a model instance from a dictionary"""
        obj = cls(**data)
        obj._update_relation_instance()
        return obj
    
    
    def _update_relation_instance(self):
        """Update the instance for all relations."""
        # Get the ID from the model instance
        # The ID field should be defined by the user
        # if not hasattr(self, "id"):
        #     return
        
        for field_name in self._get_relation_fields():
            relation = getattr(self, field_name)
            if relation is not None and isinstance(relation, Relation):
                relation.set_primary_instance(self)
                
    def model_dump(self, *args, **kwargs):
        # relation_fields = self._get_relation_fields()
        exclude = kwargs.get("exclude", set())
        if not isinstance(exclude, set):
            exclude = set()
        # exclude.update(relation_fields)
        if len(exclude) > 0:
            kwargs["exclude"] = exclude
        res = super().model_dump(*args, **kwargs)
        res["_type"] = self.__class__.__name__
        return res
    
                
    def _payload_dump(self):
        relation_fields = self._get_relation_fields()
        version_fields = ["turn_id", "branch_id", "main_branch_id", "head_id"]
            
        dump = self.model_dump(exclude={"id", "score", "_vector", *relation_fields, *version_fields})
        return dump
    
    @classmethod
    def iter_fields(cls, keys: bool = True, select: Set[str] | None = None) -> Iterator[NSFieldInfo]:
        ns = cls.get_namespace()
        return ns.iter_fields(keys, select)
    
    def transform(self) -> str:
        return self.model_dump_json()
    
    async def save(self, overwrite_relations: bool = False, *args, **kwargs) -> Self:
        """
        Save the model instance to the database
        """
                
        ns = self.get_namespace()
        # result = await ns.save(self)
        dump = self.model_dump()
        if ns.need_to_transform:
            text_content = self.transform()
            vectors = await ns.batch_vectorizer.embed_query(text_content)
            # vectors = await ns.vector_fields.transform(self)
            dump.update(vectors)
        result = await ns.insert(dump)
        # Update instance with returned data (e.g., ID)
        for key, value in result.items():
            field = ns.get_field(key, False)
            if field is None:
                rel = ns.get_relation(key)
                if rel is None:
                    raise ValueError(f"Field {key} not found in namespace {ns.table_name}")
                if not overwrite_relations and not value:
                    continue
                setattr(self, key, value)
            else:
                dv = field.deserialize(value)
                setattr(self, key, dv)
            # setattr(self, key, value)
        
        
        # Update relation instance IDs
        self._update_relation_instance()
        
        return self
    
    
    @classmethod
    async def update_query(cls, id: Any, data: dict[str, Any]) -> "SelectQuerySet[Self]":
        """Update the model instance"""
        ns = cls.get_namespace()
        return await ns.update(id, data)

    async def update(self, **kwargs) -> Self:
        """Update the model instance"""
        ns = self.get_namespace()
        result = await ns.update(self.primary_id, kwargs)
        if result is None:
            raise ValueError(f"Model '{self.__class__.__name__}' with ID '{self.primary_id}' not found")
        for key, value in result.items():
            setattr(self, key, value)
        return self
    
    
    async def delete(self, *args, **kwargs):
        """
        Delete the model instance from the database
        """
        ns = self.get_namespace()
        result = await ns.delete(id=self.primary_id)
        return result
    
    
    async def add(self, model: MODEL | Modelable[MODEL], **kwargs) -> MODEL:
        """Add a model instance to the database"""
        ns = self.get_namespace()
        if isinstance(model, Modelable):
            model = model.to_model()
        relation = ns.get_relation_by_type(model.__class__)
        if not relation:
            raise ValueError(f"Relation model not found for type: {model.__class__.__name__}")
        if isinstance(relation, NSManyToManyRelationInfo):
            result = await model.save()
            junction = relation.inst_junction_model_from_models(self, result, kwargs)
            junction = await junction.save()
        else:
            key = getattr(self, relation.primary_key)
            setattr(model, relation.foreign_key, key)
            result = await model.save()
        field = getattr(self, relation.name)
        if field is not None:
            field.append(result)
        return result
    
    
    async def add_rel(self, relation: JUNCTION_MODEL, model: MODEL) -> Tuple[JUNCTION_MODEL, MODEL]:
        ns = self.get_namespace()
        relation = ns.get_relation_by_type(model.__class__)
        if not relation:
            raise ValueError(f"Relation model not found for type: {model.__class__.__name__}")
        if isinstance(relation, NSManyToManyRelationInfo):
            result = await model.save()
            junction = relation.create_junction(self, result)            
            junction = await junction.save()
        else:
            raise ValueError("Relation is not a many to many relation")
        return junction, result
    
    @classmethod
    async def get(cls: Type[MODEL], id: Any) -> MODEL:
        """Get a model instance by ID"""
        ns = cls.get_namespace()
        data = await ns.get(id)
        if data is None:
            raise ValueError(f"Model '{cls.__name__}' with ID '{id}' not found")
        instance = cls(**data)
        instance._update_relation_instance()
        return instance
    
    @classmethod
    async def get_or_none(cls: Type[MODEL], id: Any) -> MODEL | None:
        """Get a model instance by ID or None if it doesn't exist"""
        ns = cls.get_namespace()
        data = await ns.get(id)
        instance = cls(**data) if data else None
        if not instance:
            return None
        instance._update_relation_instance()
        return instance
    
    
    @classmethod
    def query(cls: Type[Self], parse: Callable[[Self], Any] | None = None, **kwargs) -> "SelectQuerySet[Self]":
        """
        Create a query for this model
        
        Args:
            branch: Optional branch ID to query from
        """
        ns = cls.get_namespace()
        return ns.query(parse=parse, **kwargs)
    
    
    async def fetch(self, *fields: str) -> Self:
        ns = self.get_namespace()
        for field in fields:
            relation = ns.get_relation(field)
            if relation is None:
                raise ValueError(f"Relation {field} is not found")
            where_dict = {relation.foreign_key: self.primary_id}
            rel_objs = await relation.foreign_cls.query().where(**where_dict).order_by("created_at")
            setattr(self, field, rel_objs)
        return self
    

        


# No need for the ModelFactory class anymore
def get_extra(field_info: FieldInfo) -> dict[str, Any]:
    extra_json = field_info.json_schema_extra
    extra: Dict[str, Any] = {}
    if extra_json is not None:
        if isinstance(extra_json, dict):
            extra = dict(extra_json)
    return extra








def get_relation_model(cls):
    """Get the model class from a relation type."""
    args = get_args(cls)
    if len(args) != 1:
        raise ValueError("Relation model must have exactly one argument")
    return args[0]




def get_one_to_many_relation_model(cls):
    """Get the model class from a relation type."""
    args = get_args(cls)
    if len(args) != 1:
        raise ValueError("Relation model must have exactly one argument")
    return args[0]


def get_many_to_many_relation_model(cls):
    """Get the model class from a relation type."""
    args = get_args(cls)
    if len(args) != 2:
        raise ValueError("Relation model must have exactly one argument")
    return args[0], args[1]



def make_relation_default_factory(namespace: Namespace, relation_field: NSRelationInfo):
    """Create a default factory for a relation field."""
    def relation_default_factory():
        return Relation(namespace, relation_field)
    return relation_default_factory
    
    
    
    
def make_many_relation_default_factory(namespace: Namespace, relation_field: NSManyToManyRelationInfo):
    """Create a default factory for a many relation field."""
    def many_relation_default_factory():
        return ManyRelation(namespace, relation_field)
    return many_relation_default_factory

