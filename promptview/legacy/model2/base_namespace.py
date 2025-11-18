from abc import abstractmethod
import asyncio
import contextvars
from enum import Enum, StrEnum
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Generic, Iterator, List, Literal, Protocol, Self, Set, Type, TypeVar, TypedDict, Optional, get_args, get_origin, runtime_checkable
import uuid
from pydantic import BaseModel
from pydantic.fields import FieldInfo
import datetime as dt



from promptview.algebra.vectors.batch_vectorizer import BatchVectorizer
from promptview.utils.model_utils import is_list_type, unpack_list_model
from promptview.utils.string_utils import camel_to_snake

DatabaseType = Literal["qdrant", "postgres", "neo4j"]

if TYPE_CHECKING:
    from promptview.model.namespace_manager import NamespaceManager
    from promptview.model.model import Model
    from promptview.model.versioning import Branch, Turn, Partition
    from promptview.model.query_filters import QuerySetProxy
    from promptview.algebra.vectors.base_vectorizer import BaseVectorizer
INDEX_TYPES = TypeVar("INDEX_TYPES", bound=str)



class SelectFields(TypedDict):
    namespace: "Namespace"
    fields: "list[NSFieldInfo]"




class Distance(StrEnum):
    COSINE = "cosine"
    EUCLID = "euclid"
    DOT = "dot"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"
    HAMMING = "hamming"


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> dict[str, Any]:
        ...

    def deserialize(self, data: dict[str, Any]) -> Self:
        ...

    
class NSFieldInfo:
    name: str
    field_type: Type[Any]
    default: Any | None = None
    origin_type: Type[Any]
    extra: dict[str, Any] | None = None
    is_optional: bool = False
    is_list: bool = False    
    list_origin_type: Type[Any] | None = None
    is_temporal: bool = False
    is_enum: bool = False
    is_foreign_key: bool = False
    is_literal: bool = False
    enum_values: List[Any] | None = None
    enum_name: str | None = None
    is_key: bool = False
    is_primary_key: bool = False
    db_type: str | None = None
    is_vector: bool = False
    dimension: int | None = None
    distance: Distance | None = None
    
    
    def __init__(
        self,
        name: str,
        field_type: type[Any],
        default: Any | None = None,
        is_optional: bool = False,
        foreign_key: bool = False,
        is_key: bool = False,
        is_vector: bool = False,
        dimension: int | None = None,
        distance: Distance | None = None,
        namespace: "Namespace | None" = None,
        is_primary_key: bool = False,
    ):
        self.name = name   
        self.default = default
        self.is_foreign_key = foreign_key
        self.field_type = field_type
        self.origin_type, type_is_optional = NSFieldInfo.parse_optional(field_type)
        self.is_optional = is_optional or type_is_optional
        self.list_origin_type, self.is_list = NSFieldInfo.parse_list(self.origin_type)
        if self.is_list and self.list_origin_type is not None:
            self.is_enum, self.enum_values, self.is_literal = NSFieldInfo.parse_enum(self.list_origin_type)
        else:
            self.is_enum, self.enum_values, self.is_literal = NSFieldInfo.parse_enum(self.origin_type)            
        
        if self.is_enum:
            ns_name = namespace.name + "_" if namespace else ""
            if self.is_literal:
                self.enum_name = ns_name + camel_to_snake(name) + "_literal"
            else:
                self.enum_name = ns_name + camel_to_snake(self.data_type.__name__) + "_enum"

        if self.origin_type is dt.datetime:
            self.is_temporal = True
        else:
            self.is_temporal = False
            
        self.is_key = is_key
        self.is_primary_key = is_primary_key
        if is_key:
            self.key_type = "uuid" if self.field_type is uuid.UUID else "int"
        if is_vector:
            if not dimension:
                raise ValueError("Dimension is required for vector field")
            if not distance:
                raise ValueError("Distance is required for vector field")
        
        self.dimension = dimension
        self.is_vector = is_vector
        self.distance = distance
        
    # def __init__(
    #     self,
    #     name: str,
    #     field_type: type[Any],
    #     extra: dict[str, Any] | None = None,
    # ):
    #     self.name = name        
    #     self.extra = extra
    #     self.is_foreign_key = extra and extra.get("foreign_key", False)
    #     self.field_type = field_type
    #     self.db_type = extra and extra.get("db_type", None)
    #     self.origin_type, self.is_optional = NSFieldInfo.parse_optional(field_type)
    #     self.list_origin_type, self.is_list = NSFieldInfo.parse_list(self.origin_type)
    #     if self.is_list and self.list_origin_type is not None:
    #         self.is_enum, self.enum_values, self.is_literal = NSFieldInfo.parse_enum(self.list_origin_type)
    #     else:
    #         self.is_enum, self.enum_values, self.is_literal = NSFieldInfo.parse_enum(self.origin_type)            
        
    #     if self.is_enum:
    #         if self.is_literal:
    #             self.enum_name = camel_to_snake(name) + "_literal"
    #         else:
    #             self.enum_name = camel_to_snake(self.data_type.__name__) + "_enum"

    #     if self.origin_type is dt.datetime:
    #         self.is_temporal = True
    #     else:
    #         self.is_temporal = False
            
    #     if extra and extra.get("is_default_temporal", False):
    #         if not field_type is dt.datetime:
    #             raise ValueError("is_default_temporal can only be used with datetime")
    #         self.is_default_temporal = True
        
    #     if extra:
    #         if not extra.get("is_relation", False):
    #             self.is_primary_key = extra.get("primary_key", False)
                    
    #     self.is_key = extra and extra.get("is_key", False)
    #     if self.is_key:
    #         self.key_type = extra and extra.get("type", None)
    #     self.dimension = extra and extra.get("dimension", None)
    #     self.is_vector = extra and extra.get("is_vector", False)
        
    @property
    def data_type(self) -> Type[Any]:
        if self.is_list:
            if self.list_origin_type is None:
                raise ValueError("List origin type is not set")
            return self.list_origin_type
        field_type = self.field_type
        if self.is_optional:
            field_type = self.origin_type
        if origin_type := get_origin(field_type):
            return origin_type
        return field_type
    
    def get_enum_values_safe(self) -> List[str]:
        if self.is_enum and self.enum_values is not None:
            return self.enum_values
        else:
            raise ValueError("Field is not an enum")
        
    
    @classmethod
    def parse_enum(cls, field_type: type[Any]) -> tuple[bool, List[Any] | None, bool]:
        if get_origin(field_type) is Literal:            
            return True, list(get_args(field_type)), True
        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            return True, [e.value for e in field_type], False
        return False, None, False
    
    
    
    @classmethod
    def parse_optional(cls, field_type: type[Any]) -> tuple[type[Any], bool]:
        is_optional = False
        inner_type = field_type
        if type(None) in get_args(field_type):
            nn_types = [t for t in get_args(field_type) if t is not type(None)]
            if len(nn_types) != 1:
                raise ValueError(f"Field has multiple non-None types: {nn_types}")
            inner_type = nn_types[0]
            is_optional = True
        return inner_type, is_optional
    
    
    @classmethod
    def parse_list(cls, field_type: type[Any]) -> tuple[type[Any] | None, bool]:
        if is_list_type(field_type):
            inner_type = unpack_list_model(field_type)
            return inner_type, True
        else:
            return None, False
        
    
    
    def validate_value(self, value: Any) -> bool:
        """Validate the value"""
        if value is None:
            if not self.is_optional and not self.is_foreign_key and not self.is_key:
                return False
            return True
        return True
    
    @abstractmethod
    def serialize(self, value: Any) -> Any:
        """Serialize the value for the database"""
        raise NotImplementedError("Not implemented")
    
    @abstractmethod
    def deserialize(self, value: Any) -> Any:
        """Deserialize the value from the database"""
        raise NotImplementedError("Not implemented")
    
    
    def _param_repr_list(self) -> list[str]:
        params = []
        if self.is_primary_key:
            params.append("is_primary_key=True")
        if self.is_key:
            params.append("is_key=True")
        if self.is_optional:
            params.append("is_optional=True")
        if self.is_list:
            params.append("is_list=True")
        if self.is_temporal:
            params.append("is_temporal=True")
        if self.is_enum:
            params.append("is_enum=True")
        if self.is_literal:
            params.append("is_literal=True")
        if self.is_foreign_key:
            params.append("is_foreign_key=True")  
        return params
    
    def __repr__(self) -> str:
        params = self._param_repr_list()        
        return f"NSFieldInfo(name={self.name}, data_type={self.data_type.__name__}, {', '.join(params)})"


MODEL = TypeVar("MODEL", bound="Model")
FOREIGN_MODEL = TypeVar("FOREIGN_MODEL", bound="Model")
JUNCTION_MODEL = TypeVar("JUNCTION_MODEL", bound="Model")


class NSRelationInfo(Generic[MODEL, FOREIGN_MODEL, JUNCTION_MODEL]):
    name: str
    foreign_cls: Type[FOREIGN_MODEL]
    primary_key: str
    foreign_key: str
    on_delete: str
    on_update: str
    namespace: "Namespace"
    _primary_cls: Type[MODEL] | None = None
    is_one_to_one: bool = False
    def __init__(
        self, 
        namespace: "Namespace",
        name: str,         
        primary_key: str,
        foreign_key: str,
        foreign_cls: Type[FOREIGN_MODEL],
        junction_keys: list[str] | None = None,
        junction_cls: Type[JUNCTION_MODEL] | None = None,        
        on_delete: str = "CASCADE", 
        on_update: str = "CASCADE",        
        is_one_to_one: bool = False,
    ):
        self.name = name
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self.foreign_cls = foreign_cls
        self.junction_keys = junction_keys
        self.junction_cls = junction_cls
        self.on_delete = on_delete
        self.on_update = on_update
        self.namespace = namespace
        self._primary_cls = None
        self.is_one_to_one = is_one_to_one
        
    @property
    def primary_cls(self) -> Type[MODEL]:
        if self._primary_cls is None:
            raise ValueError("Primary class not set")
        return self._primary_cls
    
    def set_primary_cls(self, primary_cls: Type[MODEL]):
        self._primary_cls = primary_cls
        
    @property
    def foreign_table(self) -> str:
        return self.foreign_namespace.table_name
    
    @property
    def primary_table(self) -> str:
        return self.namespace.table_name
    
    @property
    def primary_namespace(self) -> "Namespace":
        return self.namespace
    
    @property
    def foreign_namespace(self) -> "Namespace":
        from promptview.model.namespace_manager import NamespaceManager
        return NamespaceManager.get_namespace_by_model_cls(self.foreign_cls)
        # return self.foreign_cls.get_namespace()
    
    
    @property
    def junction_table(self) -> str:
        return self.junction_namespace.table_name

    @property  
    def junction_namespace(self) -> "Namespace":
        from promptview.model.namespace_manager import NamespaceManager
        return NamespaceManager.get_namespace_by_model_cls(self.junction_cls)

    @property
    def junction_primary_key(self) -> str:
        return self.junction_keys[0]
    
    
    @property
    def junction_foreign_key(self) -> str:
        return self.junction_keys[1]
    
    def inst_junction_model(self, data: dict[str, Any]) -> JUNCTION_MODEL:
        return self.junction_namespace.instantiate_model(data)
    
    @property
    def primary_key_field(self):
        return self.namespace.get_field(self.primary_key)
    
    @property
    def foreign_key_field(self):
        return self.foreign_namespace.get_field(self.foreign_key)
    
    @property
    def junction_primary_key_field(self):
        return self.junction_namespace.get_field(self.junction_primary_key)
    
    @property
    def junction_foreign_key_field(self):
        return self.junction_namespace.get_field(self.junction_foreign_key)
    
    
    def inst_junction_model_from_models(self, primary_model: MODEL, forreign_model: FOREIGN_MODEL, data: dict[str, Any]) -> JUNCTION_MODEL:
        data.update({
            self.junction_primary_key: getattr(primary_model, self.primary_key),
            self.junction_foreign_key: getattr(forreign_model, self.foreign_key),
        })
        return self.inst_junction_model(data)
    
    
    
    def deserialize(self, value: Any) -> Any:
        if isinstance(value, str):
            json_value = json.loads(value)
            return json_value
        return value
    
    
    def inst_foreign_model(self, data: dict[str, Any]) -> FOREIGN_MODEL:
        return self.foreign_namespace.instantiate_model(data)
        # return self.foreign_cls(**data)
        
    def get_foreign_ctx_or_none(self) -> FOREIGN_MODEL | None:
        return self.foreign_namespace.get_ctx_or_none()
    
    def get_primary_ctx_value_or_none(self) -> Any:
        ctx_obj = self.primary_namespace.get_ctx()
        if ctx_obj is None:
            return None
        return getattr(ctx_obj, self.primary_key)
    
    def __repr__(self) -> str:
        return f"NSRelationInfo(name={self.name}, primary_key={self.primary_key}, foreign_key={self.foreign_key}, foreign_cls={self.foreign_cls.__name__}, primary_cls={self.primary_cls.__name__})"
     
    
JUNCTION_MODEL = TypeVar("JUNCTION_MODEL", bound="Model")

class NSManyToManyRelationInfo(Generic[MODEL, FOREIGN_MODEL, JUNCTION_MODEL], NSRelationInfo[MODEL, FOREIGN_MODEL, JUNCTION_MODEL]):
    """Many to many relation"""
    junction_cls: Type[JUNCTION_MODEL]
    junction_keys: list[str]

    def __init__(
        self,
        namespace: "Namespace",
        name: str,
        primary_key: str,
        foreign_key: str,
        foreign_cls: Type[FOREIGN_MODEL],
        junction_cls: Type[JUNCTION_MODEL],
        junction_keys: list[str],
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE"
    ):
        super().__init__(namespace, name, primary_key, foreign_key, foreign_cls, on_delete, on_update)
        self.junction_cls = junction_cls
        self.junction_keys = junction_keys

    
    @property
    def junction_table(self) -> str:
        return self.junction_namespace.table_name

    @property  
    def junction_namespace(self) -> "Namespace":
        from promptview.model.namespace_manager import NamespaceManager
        return NamespaceManager.get_namespace_by_model_cls(self.junction_cls)

    @property
    def junction_primary_key(self) -> str:
        return self.junction_keys[0]
    
    @property
    def junction_foreign_key(self) -> str:
        return self.junction_keys[1]
    
    def inst_junction_model(self, data: dict[str, Any]) -> JUNCTION_MODEL:
        return self.junction_namespace.instantiate_model(data)
    
    def inst_junction_model_from_models(self, primary_model: MODEL, forreign_model: FOREIGN_MODEL, data: dict[str, Any]) -> JUNCTION_MODEL:
        data.update({
            self.junction_primary_key: getattr(primary_model, self.primary_key),
            self.junction_foreign_key: getattr(forreign_model, self.foreign_key),
        })
        return self.inst_junction_model(data)
        
    
    def __repr__(self) -> str:
        return f"NSManyToManyRelationInfo(name={self.name}, primary_key={self.primary_key}, foreign_key={self.foreign_key}, foreign_cls={self.foreign_cls.__name__}, junction_cls={self.junction_cls.__name__}, junction_keys={self.junction_keys})"


    
    
T_co = TypeVar("T_co", covariant=True)

class QuerySetSingleAdapter(Generic[T_co]):
    def __init__(self, queryset: "QuerySet[T_co]"):
        self.queryset = queryset

    def __await__(self) -> Generator[Any, None, T_co]:
        async def await_query():
            results = await self.queryset.execute()
            if results:
                return results[0]
            return None
            # raise ValueError("No results found")
            # return None
            # raise DoesNotExist(self.queryset.model)
        return await_query().__await__()  


# MODEL = TypeVar("MODEL", bound="Model")
    
class QuerySet(Generic[MODEL]):
    """Base query set interface"""
    model_class: Type[MODEL]
    
    def __init__(self, model_class: Type[MODEL]):
        """
        Initialize the query set
        
        Args:
            model_class: Optional model class to use for instantiating results
        """
        self.model_class = model_class
        
    def __await__(self):
        """Make the query awaitable"""
        return self.execute().__await__()
    
    @property
    def q(self) -> "QuerySetProxy[MODEL, FIELD_INFO]":
        """Get the query proxy"""
        from promptview.model.query_filters import QuerySetProxy
        # raise NotImplementedError("Not implemented")
        return QuerySetProxy[MODEL, FIELD_INFO](self)
    
    def filter(self, filter_fn: Callable[[MODEL], bool] | None = None, **kwargs) -> "QuerySet[MODEL]":
        """Filter the query"""
        raise NotImplementedError("Not implemented")
    
    def set_filter(self, *args, **kwargs) -> "QuerySet[MODEL]":
        """Set the filters for the query"""
        raise NotImplementedError("Not implemented")
    
    def limit(self, limit: int) -> "QuerySet[MODEL]":
        """Limit the query results"""
        raise NotImplementedError("Not implemented")
    
    def order_by(self, field: str, direction: Literal["asc", "desc"] = "asc") -> "QuerySet[MODEL]":
        """Order the query results"""
        raise NotImplementedError("Not implemented")
    
    def turn_limit(self, limit: int, order_direction: Literal["asc", "desc"] = "desc") -> "QuerySet[MODEL]":
        """Limit the query results to the last N turns"""
        raise NotImplementedError("Not implemented")
    
    def offset(self, offset: int) -> "QuerySet[MODEL]":
        """Offset the query results"""
        raise NotImplementedError("Not implemented")
    
    def last(self)-> "QuerySetSingleAdapter[MODEL]":
        """Get the last result"""
        raise NotImplementedError("Not implemented")
    
    def tail(self, limit: int = 10) -> "QuerySet[MODEL]":
        """Get the last N results"""
        raise NotImplementedError("Not implemented")
    
    def first(self) -> "QuerySetSingleAdapter[MODEL]":
        """Get the first result"""
        raise NotImplementedError("Not implemented")
    
    def head(self, limit: int = 10) -> "QuerySet[MODEL]":
        """Get the first N results"""
        raise NotImplementedError("Not implemented")
    
    async def execute(self) -> List[MODEL]:
        """Execute the query"""
        raise NotImplementedError("Not implemented")

    def join(self, *models: "Type[Model]") -> "QuerySet[MODEL]":
        """Join the query"""
        raise NotImplementedError("Not implemented")
    
    def sub_query(self, query_set: "QuerySet") -> "QuerySet[MODEL]":
        """Create a sub query"""
        raise NotImplementedError("Not implemented")
    
    def cte(self, key: str, q1_labels: dict[str, str] | None = None, q2_labels: dict[str, str] | None = None) -> "QuerySet[MODEL]":
        """Create a common table expression"""
        raise NotImplementedError("Not implemented")
    
    
    def raw_query(self, *args, **kwargs) -> "QuerySet[MODEL]":
        """Execute a raw query"""
        raise NotImplementedError("Not implemented")

FIELD_INFO = TypeVar("FIELD_INFO", bound=NSFieldInfo)




# class Transformer(Generic[MODEL]):
#     def __init__(self, field_name: str, transform_fn: Callable[[MODEL], Any], vectorizer_cls: "Type[BaseVectorizer]"):
#         self.field_name = field_name
#         self.transform_fn = transform_fn
#         self.vectorizer_cls = vectorizer_cls
    
#     @property
#     def vectorizer(self) -> "BaseVectorizer":
#         from promptview.resource_manager import ResourceManager
#         return ResourceManager.get_vectorizer_by_cls(self.vectorizer_cls)
        
#     async def __call__(self, models: list[MODEL]) -> list[Any]:        
#         docs = [self.transform_fn(model) for model in models]
#         return await self.vectorizer.embed_documents(docs)
class Transformer:
    def __init__(self, field_name: str, transform_fn: Callable[[dict[str, Any]], Any], vectorizer_cls: "Type[BaseVectorizer] | None" = None):
        self.field_name = field_name
        self.transform_fn = transform_fn
        self.vectorizer_cls = vectorizer_cls
    
    @property
    def vectorizer(self) -> "BaseVectorizer | None":
        from promptview.resource_manager import ResourceManager
        if self.vectorizer_cls is None:
            return None
        return ResourceManager.get_vectorizer_by_cls(self.vectorizer_cls)
        
    async def __call__(self, data_list: list[dict[str, Any]]) -> list[Any]:        
        docs = [self.transform_fn(data) for data in data_list]
        return await self.vectorizer.embed_documents(docs)
        

class VectorFields(Generic[FIELD_INFO]):
    fields: dict[str, FIELD_INFO]
    
    def __init__(self):
        self.fields = {}
        self.transformers = {}
        
    def add_vector_field(self, field_name: str, field_info: FIELD_INFO):
        self.fields[field_name] = field_info
    
    def add_transformer(self, field_name: str, transformer: Transformer):
        self.transformers[field_name] = transformer
        
    def get(self, field_name: str) -> FIELD_INFO:
        return self.fields[field_name]
    
    def get_transformer(self, field_name: str) -> Transformer:
        return self.transformers[field_name]
    
    def __iter__(self) -> Iterator[FIELD_INFO]:
        return iter(self.fields.values())
    
    def __len__(self) -> int:
        return len(self.fields)
    
    def __contains__(self, field_name: str) -> bool:
        return field_name in self.fields
    
    def __getitem__(self, field_name: str) -> FIELD_INFO:
        return self.fields[field_name]
    
    def __setitem__(self, field_name: str, field_info: FIELD_INFO):
        self.fields[field_name] = field_info
        
    def __delitem__(self, field_name: str):
        del self.fields[field_name]
    
    def __repr__(self) -> str:
        return f"VectorFields(fields={self.fields})"
    
    async def transform_field(self, field_name: str, model: MODEL) -> Any:
        transformer = self.transformers[field_name]
        return await transformer([model])
    
    
    def get_vectorizer(self, field_name: str) -> "BaseVectorizer":
        from promptview.resource_manager import ResourceManager
        vectorizer = ResourceManager.get_vectorizer_by_name(field_name)
        if vectorizer is None:
            raise ValueError(f"Vectorizer for field {field_name} not found")
        return vectorizer
    
    
    async def transform_many(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vector_data = await asyncio.gather(*[transformer(data) for transformer in self.transformers.values()])
        vectors = []
        for v in vector_data:
            vectors.append({vec_name: value for vec_name, value in zip(self.transformers.keys(), v)})        
        return vectors
    
    async def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        vectors = await self.transform_many([data])
        return vectors[0]
    
    def split_vector_data(self, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        vector_data = {}
        for field_name, field_info in self.fields.items():
            if field_info.is_vector:
                vector_data[field_name] = data[field_name]
                del data[field_name]
        return data, vector_data
    
    
    
    


class Namespace(Generic[MODEL, FIELD_INFO]):
    _model_cls: Type[MODEL] | None = None
    _name: str
    _fields: dict[str, FIELD_INFO]
    _relations: dict[str, NSRelationInfo]
    is_versioned: bool
    is_artifact: bool
    is_repo: bool
    db_type: DatabaseType
    _primary_key: FIELD_INFO | None = None
    _curr_ctx_model: contextvars.ContextVar
    _vector_fields: dict[str, FIELD_INFO]
    _transformers: dict[str, Transformer]   
    default_temporal_field: FIELD_INFO | None = None
    def __init__(
        self, 
        name: str, 
        db_type: DatabaseType,
        is_versioned: bool = False, 
        is_repo: bool = False, 
        is_artifact: bool = False,
        is_context: bool = False,
        repo_namespace: Optional[str] = None,         
        namespace_manager: Optional["NamespaceManager"] = None
    ):
        self._name = name
        self._fields = {}
        self._relations = {}
        self._vector_fields = {}
        self.is_versioned = is_versioned
        self.is_repo = is_repo
        self.is_context = is_context
        self.is_artifact = is_artifact
        self.repo_namespace = repo_namespace
        self.namespace_manager = namespace_manager
        self.vector_fields = VectorFields()
        self.batch_vectorizer = BatchVectorizer()
        self._transformers = {}
        self.db_type = db_type
        self._model_cls = None
        self._primary_key = None
        self.default_temporal_field = None
        self._curr_ctx_model = contextvars.ContextVar(f"curr_ctx_{self._name}")

       
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def table_name(self) -> str:
        """Get the table name for this namespace."""
        return self._name
    
    
    @property
    def model_class(self) -> Type[MODEL]:
        if self._model_cls is None:
            raise ValueError("Model class not set")
        return self._model_cls
    
    
    @property
    def primary_key(self) -> FIELD_INFO:
        # if self.is_artifact:
        #     return self._fields["artifact_id"]
        if self._primary_key is None:
            raise ValueError("Primary key not found")
        return self._primary_key
    
    def get_dump_primary_key(self, data: dict[str, Any], remove_key: bool = True) -> Any:
        if self._primary_key is None:
            raise ValueError("Primary key not found")
        primary_key = data.get(self._primary_key.name)
        # if primary_key is None:
            # return None
        if remove_key:
            del data[self._primary_key.name]
        return primary_key
    
    def find_primary_key(self) -> FIELD_INFO | None:
        for field_name, field_info in self._fields.items():
            if field_info.is_primary_key:
                return field_info
        return None
    
    def set_model_class(self, model_class: Type[MODEL]):
        self._model_cls = model_class
        for relation_info in self._relations.values():
            relation_info.set_primary_cls(model_class)    
            
    def instantiate_model(self, data: dict[str, Any]) -> MODEL:
        if self._model_cls is None:
            raise ValueError("Model class not set")
        model_dump = self.validate_model_fields(data)
        return self._model_cls.from_dict(model_dump)
        # return self._model_cls(**model_dump)
    
    
    def set_ctx(self, model: MODEL):
        self._curr_ctx_model.set(model)
        
    def get_ctx(self) -> MODEL:
        return self._curr_ctx_model.get(None)
    
    def get_ctx_or_none(self, throw_error: bool = True) -> MODEL | None:
        try:
            model = self._curr_ctx_model.get(None)
        except LookupError:
            if throw_error:
                raise ValueError(f"Context model for {self._name} not set")
            return None
        return model
    
    def get_field(self, name: str, throw_error: bool = True) -> FIELD_INFO | None:
        field = self._fields.get(name, None)
        if field is None and throw_error:
            raise ValueError(f"""Field "{name}" not found in "{self._name}" namespace""")
        return field
    
    def has_field(self, name: str) -> bool:
        return name in self._fields
    
    def has_relation(self, name: str) -> bool:
        return name in self._relations
    
    def get_relation(self, name: str) -> NSRelationInfo | None:
        return self._relations.get(name, None)
    
    
    def get_relation_by_type(self, model: Type[MODEL]) -> NSRelationInfo | None:
        for rel in self._relations.values():
            if rel.foreign_cls.__name__ == model.__name__:
                return rel
        return None
    
    
    def get_repo_namespace(self) -> Optional["Namespace"]:
        if not self.namespace_manager:
            raise ValueError("Namespace manager not set")
        if self.repo_namespace:
            return self.namespace_manager.get_namespace(self.repo_namespace)
        return None
    
    # def iter_fields(self, keys: bool = True) -> Iterator[FIELD_INFO]:
    #     for field in self._fields.values():
    #         if field.is_key and not keys:
    #             continue
    #         yield field
    
    def iter_fields(
        self, 
        keys: bool = True, 
        select: Set[str] | None = None, 
        is_vector: bool = True, 
        is_optional: bool | None = None,
        exclude: Set[str] | None = None,
        default: bool | None = None
        ) -> "Iterator[FIELD_INFO]":
        for field in self._fields.values():
            if not keys and field.is_key:
                continue
            if select is not None and field.name not in select:
                continue
            if not is_vector and field.is_vector:
                continue
            if is_optional is not None and field.is_optional != is_optional:
                continue
            if exclude is not None and field.name in exclude:
                continue
            if default is not None:
                if default and field.default is None:
                    continue
                if not default and field.default is not None:  
                    continue
            yield field
            
    def iter_relations(self, is_one_to_one: bool | None = None) -> Iterator[NSRelationInfo]:
        for relation in self._relations.values():
            if is_one_to_one is not None and relation.is_one_to_one != is_one_to_one:
                continue
            yield relation
            
    def get_field_names(self) -> list[str]:
        return list(self._fields.keys())
    
    def select_fields(self, fields: list[str] | None = None) -> SelectFields:
        
        if fields is None:
            return SelectFields(
                namespace=self,
                fields=[f for f in self._fields.values()]
            )
        return SelectFields(
            namespace=self,
            fields=[self._fields[field] for field in fields]
        )
        
        
    def get_foreign_key_ctx_value(self, field: FIELD_INFO) -> Any:
        """Get the value of the foreign key field from the context"""
        from promptview.model.namespace_manager import NamespaceManager
        if not field.is_foreign_key:
            raise ValueError(f"""Field "{field.name}" on model "{self.model_class.__name__}" is not a foreign key""")
        relation = NamespaceManager.get_reversed_relation(self.table_name, field.name)
        if relation is None:
            raise ValueError(f"""Field "{field.name}" on model "{self.model_class.__name__}" is a key field but has no reversed relation""")
        return relation.get_primary_ctx_value_or_none()
        
        
    def validate_model_fields(self, dump: dict[str, Any]) -> dict[str, Any]:
        """Validate the model fields"""
        from promptview.model.namespace_manager import NamespaceManager
        for field in self._fields.values():
            if field.is_foreign_key:
                if dump[field.name] is None:
                    relation = NamespaceManager.get_reversed_relation(self.table_name, field.name)
                    if relation is None:
                        # raise ValueError(f"""Field "{field.name}" on model "{self.model_class.__name__}" is a key field but has no reversed relation""")                        
                        raise ValueError(f"""Field "{field.name}" in "{self.name}" is a key field but has no reversed relation""")
                    curr_rel_model = relation.primary_cls.current()
                    if curr_rel_model is None and not field.is_optional:
                        # raise ValueError(f"""Field "{field.name}" on model "{self.model_class.__name__}" is a key field but no context was found for class "{relation.primary_cls.__name__}".""")
                        raise ValueError(f"""Field "{field.name}" in "{self.name}" is a key field but no context was found for class "{relation.name}".""")
                    dump[field.name] = getattr(curr_rel_model, relation.primary_key) if curr_rel_model is not None else None
        return dump
    
    def add_field(
        self,
        name: str,
        field_type: type[Any],
        default: Any | None = None,
        is_optional: bool = False,
        foreign_key: bool = False,
        is_key: bool = False,
        is_vector: bool = False,
        dimension: int | None = None,
        distance: Distance | None = None,
        is_primary_key: bool = False,
        is_default_temporal: bool = False,        
    ):
        """Add a field to the namespace"""
        raise NotImplementedError("Not implemented")
    
    def add_relation(
        self,
        name: str,
        primary_key: str,
        foreign_key: str,
        foreign_cls: Type["Model"], 
        junction_cls: Type["Model"] | None = None,
        junction_keys: list[str] | None = None,       
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        is_one_to_one: bool = False,
    ):
        """
        Add a relation to the namespace.
        
        Args:
            name: The name of the relation
            primary_key: The name of the primary key in the target model
            foreign_key: The name of the foreign key in the target model
            foreign_cls: The class of the target model
            on_delete: The action to take when the referenced row is deleted
            on_update: The action to take when the referenced row is updated
        """
        raise NotImplementedError("Not implemented")
    
    
    
    def add_many_relation(
        self,
        name: str,
        primary_key: str,
        foreign_key: str,
        foreign_cls: Type["Model"],
        junction_cls: Type["Model"],
        junction_keys: list[str],
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
    ):
        """
        Add a many to many relation to the namespace.
        
        Args:
            name: The name of the relation
            primary_key: The name of the primary key in the target model
            foreign_key: The name of the foreign key in the target model
            foreign_cls: The class of the target model
            junction_cls: The class of the junction model
            junction_keys: The keys of the junction model
            on_delete: The action to take when the referenced row is deleted
            on_update: The action to take when the referenced row is updated
        """
        raise NotImplementedError("Not implemented")
    
    @property
    def need_to_transform(self) -> bool:
        return bool(len(self.vector_fields) > 0)
    
    def register_transformer(self, field_name: str, transformer: Callable[[MODEL], Any], vectorizer_cls: "Type[BaseVectorizer] | None" = None):
        from promptview.resource_manager import ResourceManager
        if vectorizer_cls is not None:
            ResourceManager.register_vectorizer(vectorizer_cls)
        self.vector_fields.add_transformer(field_name, Transformer(field_name, transformer, vectorizer_cls))
        # self._transformers[field_name] = Transformer(field_name, transformer, vectorizer)
        
        
    # def register_vectorizer(self, field_name: str, vectorizer_cls: "Type[BaseVectorizer]"):
    #     from promptview.resource_manager import ResourceManager
    #     ResourceManager.register_vectorizer(vectorizer_cls)
    #     self.vector_fields.add_vector_field(field_name, vectorizer_cls())
    def register_vector_field(self, field_name: str, vectorizer_cls: "Type[BaseVectorizer]"):
        from promptview.resource_manager import ResourceManager
        vectorizer = ResourceManager.register_vectorizer(vectorizer_cls)
        self.batch_vectorizer.add_vectorizer(field_name, vectorizer)
        return vectorizer
        
    def get_transformers(self) -> dict[str, Transformer]:
        return self._transformers
    
    def get_transformer(self, field_name: str) -> Transformer | None:
        return self._transformers.get(field_name, None)
    
    # def transform_model(self, model: MODEL) -> dict[str, Any]:
    #     vector_payload = {}
    #     for field_name, transformer in self._transformers.items():
    #         vector_payload[field_name] = transformer(model)
    #     return vector_payload
    
    async def transform_field(self, field_name: str, model: MODEL) -> Any:
        transformer = self._transformers[field_name]
        return await transformer([model])
    
    
    def get_vectorizer(self, field_name: str) -> "BaseVectorizer":
        from promptview.resource_manager import ResourceManager
        vectorizer = ResourceManager.get_vectorizer_by_name(field_name)
        if vectorizer is None:
            raise ValueError(f"Vectorizer for field {field_name} not found")
        return vectorizer
    

    
    async def transform_model_list(self, models: list[MODEL]) -> dict[str, list[Any]]:
        res_list = await asyncio.gather(*[transformer(models) for transformer in self._transformers.values()])
        return {field_name: res for field_name, res in zip(self._transformers.keys(), res_list)}
    
    async def transform_model(self, model: MODEL) -> dict[str, Any]:
        res = await self.transform_model_list([model])
        return {field_name: res[field_name][0] for field_name in res}
    
    async def get_current_ctx_head(self, turn: "int | Turn | None" = None, branch: "int | Branch | None" = None) -> tuple[int | None, int | None]:
        from promptview.model.context import Context
        turn_id, branch_id = await Context.get_current_head(turn, branch)
        if self.is_versioned:
            if turn_id is None:
                raise ValueError(f"""Turn is required when saving {self.model_class.__name__} to an artifact""")
            if branch_id is None:
                raise ValueError(f"""Branch is required when saving {self.model_class.__name__} to an artifact""")
        return turn_id, branch_id
    
        
    async def get_current_ctx_branch(self, branch: "int | Branch | None" = None) -> int | None:
        from promptview.model.context import Context
        return await Context.get_current_branch(branch)
    
    async def get_current_ctx_partition_branch(self, partition: "int| Partition | None", branch: "int | Branch | None" = None):        
        from promptview.model.context import Context
        partition = await Context.get_current_partition(partition)
        branch = await Context.get_current_branch(branch)        
        return partition, branch
    

    
    # async def save(self, data: Dict[str, Any], id: Any | None = None, artifact_id: uuid.UUID | None = None, version: int | None = None, turn: "int | Turn | None" = None, branch: "int | Branch | None" = None) -> Dict[str, Any]:
    #     """Save data to the namespace"""
    #     raise NotImplementedError("Not implemented")
    
    async def save(self, model: MODEL) -> MODEL:
        """Save data to the namespace"""
        raise NotImplementedError("Not implemented")
    
    async def insert(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a model into the namespace"""
        raise NotImplementedError("Not implemented")
    
    async def update(self, id: Any, data: dict[str, Any]) -> MODEL | None:
        """Update a model in the namespace"""
        raise NotImplementedError("Not implemented")
    
    async def get(self, id: Any) -> MODEL | None:
        """Get data from the namespace by ID"""
        raise NotImplementedError("Not implemented")
    
    async def get_artifact(self, artifact_id: uuid.UUID, version: int | None = None) -> MODEL | None:
        """Get data from the namespace by artifact ID and version"""
        raise NotImplementedError("Not implemented")
    
    async def delete(self, id: Any) -> MODEL | None:
        """Delete data from the namespace"""
        raise NotImplementedError("Not implemented")
    
    async def delete_model(self, model: MODEL) -> MODEL | None:
        """Delete data from the namespace"""
        raise NotImplementedError("Not implemented")

    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute a raw query"""
        raise NotImplementedError("Not implemented")
    
    async def fetch(self, *args, **kwargs) -> List[MODEL]:
        """Execute a raw model query"""
        raise NotImplementedError("Not implemented")
    
    def query(
        self, 
        parse: Callable[[MODEL], Any] | None = None,
        **kwargs
    ) -> QuerySet:
        """
        Create a query for this namespace
        
        Args:
            branch: Optional branch or branch ID to query from            
        """
        raise NotImplementedError("Not implemented")
    

    
    async def recreate_namespace(self):
        """Recreate the namespace in the database"""
        raise NotImplementedError("Not implemented")
    
    
    
    def create_namespace(self):
        """Create the namespace in the database"""
        raise NotImplementedError("Not implemented")
    
    def drop_namespace(self):
        """Drop the namespace from the database"""
        raise NotImplementedError("Not implemented")