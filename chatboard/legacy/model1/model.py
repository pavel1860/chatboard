import asyncio
from dataclasses import dataclass
import inspect
import copy
import json
from typing import Any, Callable, Dict, ForwardRef, Generic, List, Optional, Protocol, Self, Type, TypeVar,  get_args, get_origin
from uuid import uuid4
from pydantic import PrivateAttr, create_model, ConfigDict, BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic._internal._model_construction import ModelMetaclass
from pydantic._internal._config import ConfigWrapper
from pydantic_core import core_schema
from chatboard.artifact_log.artifact_log3 import ArtifactLog
from chatboard.model.head_model import HeadModel
from chatboard.utils.string_utils import camel_to_snake
from chatboard.model.query_types import QueryListType
from .query import AllVecs, ModelFilterProxy, QueryFilter, ALL_VECS, QueryProxy, QuerySet, FusionType, QuerySetSingleAdapter, QueryType, parse_query_params
from .vectors.base_vectorizer import BaseVectorizer
from .fields import VectorSpaceMetrics, get_model_indices
from .resource_manager import VectorSpace, connection_manager, DatabaseType

def unpack_list_model(pydantic_model):
    return get_args(pydantic_model)[0]




def get_model_fields(model_instance, model_class):
    try:
        fields = {field: getattr(model_instance, field) for field in model_class.__fields__.keys()}
        return model_class(**fields)
    except:
        return None



MetadataArg = BaseModel | dict | str | None

class DefaultInput(BaseModel):
    input: str

class DefaultOutput(BaseModel):
    output: str


TInput = TypeVar('TInput', bound=BaseModel | str | None)
TOutput = TypeVar('TOutput', bound=BaseModel | str )
TMetadata = TypeVar('TMetadata', bound=BaseModel)

def serialize_field(field):
    if field is None:
        return None
    if type(field) == str:
        return field
    if isinstance(field, BaseModel):
        return field.dict()


def copy_field(field):
    if field is None:
        return None
    if type(field) == str:
        return field
    if isinstance(field, BaseModel):
        return field.model_copy()


class ModelManager:
    
    _assets: Dict[str, "Model"] = {}
    
    def __init__(self):
        self._assets = {}
        
    def get_asset(self, asset_name: str):
        return self._assets.get(asset_name, None)
    
    def add_asset(self, asset: "Model"):
        self._assets[asset.__name__] = asset
        return asset
        

model_manager = ModelManager()  


def build_namespace(model_cls_name: str, db_type: DatabaseType):
    return f"{camel_to_snake(model_cls_name)}s" if db_type == "postgres" else model_cls_name

@dataclass
class ForeignKey:
    key: str # the name of the foreign key in the model
    referenced_table: str # the name of the table that is referenced
    referenced_column: str # the column of the referenced table
    on_delete: str = "CASCADE" # the action to take when the referenced row is deleted
    on_update: str = "CASCADE" # the action to take when the referenced row is updated
    target_db_type: DatabaseType = "postgres"


@dataclass
class RelationBlueprint(ForeignKey):
    namespace: str | None = None            
    target_forward_ref: ForwardRef | None = None
    target_db_type: DatabaseType = "qdrant"
    
    
    def get_target_namespace(self) -> str:
        if self.namespace:
            return self.namespace
        elif self.target_forward_ref:
            return build_namespace(self.target_forward_ref.__forward_arg__, self.target_db_type)
        else:
            raise ValueError("Target namespace not defined")
    
    # target_type.__forward_arg__

# AssetBase = TypeVar('AssetBase')


# def ModelRelation(
#     # model: "Type[Model]",
#     key: str,
# ):
#     json_schema_extra={
#         "type": "relation",
#         "is_relation": True,
#         "partition": key,
#         # "model": model,
#     }
#     return Field(
#         # model,
#         json_schema_extra=json_schema_extra,
#     )




MODEL = TypeVar("MODEL", bound="Model")


def get_relation_model(cls):
    args = get_args(cls)
    if len(args) != 1:
        raise ValueError("Relation model must have exactly one argument")
    return args[0]



def make_default_factory(model_cls, rel_field):
    return lambda: Relation(model_cls=model_cls, rel_field=rel_field)

class Relation(Generic[MODEL]):
    model_cls: Type[MODEL]
    rel_field: str
    instance_id: str | None = None
    
    def __init__(self, model_cls: Type[MODEL], rel_field: str):
        self.model_cls = model_cls
        self.rel_field = rel_field
        self.instance_id = None
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, 
        _source_type: Any, 
        _handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        from pydantic_core import core_schema
        # Use a simple any schema since we're handling serialization ourselves
        return core_schema.any_schema()
    
    
    def _build_query_set(self, query_type: QueryType = "scroll"):
        if not self.instance_id:
            raise ValueError("Instance ID is not set")
        return QuerySet(self.model_cls, query_type, {self.rel_field: self.instance_id})
    
    def all(self, partitions=None, limit=10, start_from=None, offset=0, ascending=False, ids=None):
        qs = self._build_query_set()        
        return qs
    
    def limit(self, limit: int):
        qs = self._build_query_set()
        qs.limit(limit)
        return qs
    
    def filter(self, filters: Callable[[MODEL], bool]):
        qs = self._build_query_set()
        qs.filter(filters)
        return qs
    
    async def add(self, obj: MODEL):
        # setattr(obj, self._rel_field, self._cls)
        setattr(obj, self.rel_field, self.instance_id)
        await obj.save()
        return obj
        
    
    def get_or_create(self):
        pass
    
    def upsert(self):
        pass
    
    def create(self,):
        pass
    
    def first(self, partitions=None):
        pass
        

    def last(self, partitions=None):
        pass
        
    def set_instance_id(self, instance_id: str):
        self.instance_id = instance_id





class ModelMeta(ModelMetaclass, type):
    
    # def __new__(cls, name, bases, dct):
    #     print("Creating class", name)
    #     return super().__new__(cls, name, bases, dct)
    
    def __new__(cls, name, bases, dct):
        print("Creating class", name)
        cls_partitions = {}
        default_temporal_field = None
        default_temporal_type = None
        namespace = None
        db_type = "qdrant"
        versioned = False
        is_head = False
        is_detached_head = False
        vec_field_map = {}
        vector_spaces = []
        relations = {}
        is_abstract = False
        config = None
        if name == "Model":
            is_abstract = True
        
        
        if "Config" in dct:
            config = dct["Config"]
            if hasattr(config, "is_abstract"):
                is_abstract = config.is_abstract
        # extract config from base class
        elif bases and type(bases) == tuple and hasattr(bases[0], "Config"):
            config = bases[0].Config            
            
        if config:        
            if hasattr(config, "database_type"):
                db_type = config.database_type             
            if hasattr(config, "versioned"):
                versioned = config.versioned
            if hasattr(config, "is_head"):
                is_head = config.is_head 
            if hasattr(config, "namespace"):
                namespace = config.namespace
            if hasattr(config, "is_detached_head"):
                is_detached_head = config.is_detached_head
            
        for base in bases:
            if base == HeadModel:
                is_head = True

            
        
        if not is_abstract:
            #? add model partition
            if not bases:
                raise ValueError(f"Vector Space not defined in {name} and no base class")
            model_base = bases[0]
            if not namespace:
                if model_base == Model:
                    # namespace = f"{camel_to_snake(name)}s" if db_type == "postgres" else name
                    namespace = build_namespace(name, db_type)
                    dct["_subspace"] = None
                else:
                    if not hasattr(model_base, "_namespace"):
                        raise ValueError(f"Namespace not defined in base class of {name}")
                    namespace = model_base._namespace.default
                    dct["_subspace"] = name
            else:
                dct["_subspace"] = None
            # Get database type from Config class if it exists
            # if "Config" in dct:
            #     if hasattr(dct["Config"], "database_type"):
            #         db_type = dct["Config"].database_type             
            #     if hasattr(dct["Config"], "versioned"):
            #         versioned = dct["Config"].versioned
            #     if hasattr(dct["Config"], "is_head"):
            #         is_head = dct["Config"].is_head
            
            
            
            if "VectorSpace" in dct:
                field_type = dct["VectorSpace"]
                if not inspect.isclass(field_type.__class__):
                    raise ValueError(f"VectorSpace must be a class")
                #? vector space extraction
                vectorizers = field_type.__annotations__
                for vec_name, vec_cls in vectorizers.items():
                    vector_spaces.append(
                        VectorSpace(
                            name=vec_name,
                            namespace=namespace,
                            vectorizer_cls=vec_cls,
                            metric=VectorSpaceMetrics.COSINE                                
                        ))
                #? namespace and indices extraction
                indices = get_model_indices(dct)
                connection_manager.add_namespace(
                    namespace=namespace,
                    # subspace=dct.get("_subspace"),
                    vector_spaces=vector_spaces,
                    indices=indices,
                    db_type=db_type,
                    versioned=versioned,
                    is_head=is_head,
                    is_detached_head=is_detached_head
                )
            else:
                if db_type == "qdrant":                    
                    ns = connection_manager.get_namespace(namespace)                
                    if not ns:
                        raise ValueError("Vector Space not defined for Qdrant database")
                    vector_spaces = list(ns.vector_spaces.values())
                elif db_type == "postgres":
                    
                    indices = get_model_indices(dct)
                    connection_manager.add_namespace(
                        namespace=namespace,
                        vector_spaces=[],
                        indices=indices,
                        db_type=db_type,
                        versioned=versioned,
                        is_head=is_head,
                        is_detached_head=is_detached_head
                    )
                else:
                    raise ValueError(f"Unsupported database type: {db_type}")
            # for field, field_type in dct.items():
            #     if inspect.isclass(field_type.__class__) and field == "VectorSpace":    
            #         #? vector space extraction
            #         vectorizers = field_type.__annotations__
            #         for vec_name, vec_cls in vectorizers.items():
            #             vector_spaces.append(
            #                 VectorSpace(
            #                     name=vec_name,
            #                     namespace=namespace,
            #                     vectorizer_cls=vec_cls,
            #                     metric=VectorSpaceMetrics.COSINE                                
            #                 ))
            #         #? namespace and indices extraction
            #         indices = get_model_indices(dct)
            #         connection_manager.add_namespace(
            #             namespace=namespace,
            #             # subspace=dct.get("_subspace"),
            #             vector_spaces=vector_spaces,
            #             indices=indices,
            #             db_type=db_type
            #         )
            #         break
            # else:
            #     ns = connection_manager.get_namespace2(namespace)
            #     vector_spaces = list(ns.vector_spaces.values())
            
            for field, field_type in dct.items():
                #? temporal field extraction
                if inspect.isclass(field_type.__class__):
                    # if isinstance(field_type, FieldInfo) and field_type.json_schema_extra:
                    #     if field_type.json_schema_extra.get("auto_now_add", None):
                    #         default_temporal_field = field
                    #         default_temporal_type = field_type                    
                    
                    # #? vector field extraction. the vector has to be a field of type VectorSpace
                    # if type(field_type) == FieldInfo:
                    #     if not field_type.json_schema_extra:
                    #         raise ValueError(f"You should use ModelField instead of pydantic Field for {field}")
                    #     if vec:= field_type.json_schema_extra.get("vec"):                            
                    #         for v in vec: # type: ignore
                    #             if v not in vec_field_map:
                    #                 vec_field_map[v] = []
                    #             vec_field_map[v] += [field]
                    if isinstance(field_type, FieldInfo):
                        if not field_type.json_schema_extra:
                            raise ValueError(f"You should use ModelField instead of pydantic Field for {field}")
                        if field_type.json_schema_extra.get("auto_now_add", None):
                            default_temporal_field = field
                            default_temporal_type = field_type                    
                    #? vector field extraction. the vector has to be a field of type VectorSpace
                        if vec:= field_type.json_schema_extra.get("vec"):                            
                            for v in vec: # type: ignore
                                if v not in vec_field_map:
                                    vec_field_map[v] = []
                                vec_field_map[v] += [field]
                        # if field_type.json_schema_extra.get("type", None) == "relation":
                        #     to_model_namespace = field_type.json_schema_extra.get("model", None)._namespace.default
                        #     key = field_type.json_schema_extra.get("key", None)
                        #     if not key:
                        #         raise ValueError(f"Relation key not defined for {field}")
                        #     connection_manager.add_relation(namespace, to_model_namespace, key)
                                
                    
                        #TODO validate that the vector exists in the vector space
            if vec_field_map:
                for vs in vector_spaces:
                    if vs.name not in vec_field_map:
                        raise ValueError(f"Vector Space '{vs.name}' not defined in model fields. if you are using explicit vector field mapping, ensure that all vectorizers are defined in the model fields")

            if dct.get("_subspace", None):
                if not hasattr(bases[0], "_namespace"):
                    raise ValueError(f"Namespace not defined in base class of {name}")
                indices = get_model_indices(dct)
                connection_manager.add_subspace(bases[0]._namespace.default, dct["_subspace"], indices)
                
            #? partition extraction
            if annotations:= dct.get("__annotations__", None):
                for field, field_type in annotations.items():       
                    field_origin = get_origin(field_type)
                    if field_origin == list:
                        target_type = unpack_list_model(field_type)
                    else:
                        target_type = field_type
                    if field_origin and inspect.isclass(field_origin) and issubclass(field_origin, Relation): 
                        target_type = get_relation_model(target_type)
                        field_info = dct.get(field)                            
                        if not field_info:
                            raise ValueError(f"Field {field} is not defined")
                        key = field_info.json_schema_extra.get("key")
                        if not key:
                            raise ValueError(f"Model Field {field} does not have partition")
                        _target_type, _forward_ref = (target_type, None) if not isinstance(target_type, ForwardRef) else (None, target_type)
                        
                        cls_partitions[field] = RelationBlueprint(                            
                            referenced_column= field,
                            referenced_table= namespace,
                            key= key,
                            # target_model_cls= _target_type,
                            namespace= target_type._namespace.default if _target_type is not None else None,                            
                            target_db_type= db_type, #TODO: check if this is correct
                            on_delete= field_info.json_schema_extra.get("on_delete", "CASCADE"),
                            on_update= field_info.json_schema_extra.get("on_update", "CASCADE"),
                            target_forward_ref= _forward_ref
                        )
                        # cls_partitions[field] = {                            
                            # "type": _target_type,
                            # "origin": field_origin,
                            # "partition": partition,
                            # "field": field,
                            # "forward_ref": _forward_ref,
                            # "source_namespace": namespace,
                            # "target_namespace": target_type._namespace.default if target_type._namespace else None,
                            # "field_info": field_info
                        # }
                        connection_manager.add_relation(cls_partitions[field])
                        
                        field_info.default_factory = make_default_factory(target_type, key)
                        print("Found",field, key)
                
            dct["_partitions"] = cls_partitions
            dct["_default_temporal_field"] = default_temporal_field
            dct["_vec_field_map"] = vec_field_map
            dct["_db_type"] = db_type
            dct["_foreign_keys"] = {}
            if namespace:
                dct["_namespace"] = namespace
            # dct["_vectorizers"] = connection_manager.get_vectorizers(namespace)
        model_cls_inst = super().__new__(cls, name, bases, dct)
        if not is_abstract:
            if not namespace:
                raise ValueError("Namespace not defined")
            connection_manager.add_model(model_cls_inst)
            model_manager.add_asset(model_cls_inst)# type: ignore
        return model_cls_inst
    
    
    # def __getattr__(cls, name):
    #     if field_info:= cls.model_fields.get(name, None):
    #         # print("Getting attribute",cls.__name__, name)
    #         return FieldComparable(name, field_info)
    #     return super().__getattr__(name)





class Model(BaseModel, metaclass=ModelMeta):
    # _id: str = PrivateAttr(default_factory=lambda: str(uuid4()))
    # _score: float = PrivateAttr(default=-1)
    # id: str | int = Field(default_factory=lambda: str(uuid4()), description="Unique Identifier")
    # id: str | int = Field(..., description="Unique Identifier")
    score: float = Field(default=-1, description="Score of the document. Default is -1")
    turn_id: str | int = Field(default=-1, description="Unique Identifier of the turn")
    branch_id: str | int = Field(default=-1, description="Unique Identifier of the branch")
    # head_id: str | int = Field(default=-1, description="Unique Identifier of the head")
    _partitions: dict[str, str] = PrivateAttr(default_factory=dict)
    _default_temporal_field: str = PrivateAttr(default=None)
    _namespace: str | None = PrivateAttr(default=None)
    _subspace: str | None = PrivateAttr(default=None)
    _vec_field_map: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _vector: dict[str, Any] = PrivateAttr(default_factory=dict)
    _db_type: DatabaseType = PrivateAttr(default="qdrant")
    _id: str | int | None = PrivateAttr(default=None)
    _foreign_keys: dict[str, ForeignKey] = {}
    # _partitions: dict[str, str] = {}
    # _default_temporal_field: str | None = None
    # _namespace: str | None = None
    # _subspace: str | None = None
    # _vec_field_map: dict[str, list[str]] = {}
    
    # class Config:
    #     arbitrary_types_allowed = True
    #     database_type: DatabaseType = "qdrant"  # Default to qdrant for backward compatibility
    #     versioned: bool = False
    #     is_head: bool = False
    
    
    # def __init__(
    #         self,
    #         # _id=None, 
    #         # _score=None,
    #         _vector=None,
    #         **data
    #     ):
    #     if "id" not in data:
    #         if self.__class__.Config.database_type == "qdrant":
    #             data["id"] = str(uuid4())
    #         else:
    #             data["id"] = -1
    #     super().__init__(**data)
    #     # if _id:
    #     #     self._id = _id
    #     # if _score:
    #     #     self._score = _score 
    #     # if id:
    #     #     self.id = id
    #     if _vector:
    #         self._vector = _vector

    
        
        
    # @property
    # def id(self):
    #     return self._id
    
    # @property
    # def score(self):
    #     return self._score
    
    @property
    def id(self):
        if self._id is None:
            raise ValueError("Id not set. Please use save() method to set id.")
        return self._id
    
    @property
    def vector(self):
        return self._vector
    
    
    
    @classmethod
    async def get_client(cls: Type["Model"]):
        if not cls._namespace:
            raise ValueError("Namespace not defined")
        ns = await connection_manager.get_namespace_or_create(cls._namespace)
        return ns.conn
    
    
    @classmethod
    def add_foreign_key(cls: Type["Model"], key: str, referenced_table: str, referenced_column: str, on_delete: str = "CASCADE", on_update: str = "CASCADE"):
        if key in cls.model_fields:
            raise ValueError(f"Foreign key {key} already exists")
        cls._foreign_keys[key] = ForeignKey(key, referenced_table, referenced_column, on_delete, on_update)
        
    @classmethod
    def get_foreign_key(cls: Type["Model"], key: str):
        return cls._foreign_keys.get(key)
        
    
    def _payload_dump(self):
        relation_fields = self._get_relation_fields()
        version_fields = ["turn_id", "branch_id", "main_branch_id", "head_id"]
            
            
        dump = self.model_dump(exclude={"id", "score", "_vector", *relation_fields, *version_fields})
        dump["_subspace"] = self._subspace
        return dump
    
    
    def model_dump(self, *args, **kwargs):
        relation_fields = self._get_relation_fields()
        exclude = kwargs.get("exclude", set())
        if not isinstance(exclude, set):
            exclude = set()
        exclude.update(relation_fields)
        kwargs["exclude"] = exclude
        res = super().model_dump(*args, **kwargs)
        if "id" not in exclude:
            if self._id is not None:
                res["id"] = self._id
        if "head_id" not in exclude and issubclass(self.__class__, HeadModel):
            # res["head_id"] = self.head.id
            if self.head:
                res["head"] = self.head.model_dump()
        return res
        
    # async def verify_namespace(self):
        # await self.rag_documents.verify_namespace()
    def _get_relation_fields(self):
        if not self.__class__._partitions:
            return []
        return list(self.__class__._partitions.default.keys())
    
    
    def _update_relation_instance_id(self):
        for field in self._get_relation_fields():
            if not getattr(self, field):
                continue
            if isinstance(getattr(self, field), Relation):
                getattr(self, field).set_instance_id(self.id)
    
    @classmethod
    async def get_namespace(cls):
        if not cls._namespace:
            raise ValueError("Namespace not defined")
        namespace = cls._namespace.default # type: ignore
        return await connection_manager.get_namespace_or_create(namespace)
    
    @classmethod
    def _get_vec_field_map(cls: Type[MODEL]):
        return cls._vec_field_map.default# type: ignore

    def model_chunk(self, vectorizers: dict[str, BaseVectorizer])-> dict[str, str]:
        relation_fields = self._get_relation_fields()
        if not self._get_vec_field_map():
            model_dump = self.model_dump_json(exclude=set(relation_fields))
            return {vec_name: model_dump for vec_name in vectorizers}
        vec_strings = {}
        for vec, fields in self._get_vec_field_map().items():
            fields_str = [getattr(self, field) for field in fields]            
            vec_strings[vec] = "\n".join([str(f) for f in fields_str])
        return vec_strings
    
    
    @classmethod 
    async def query_embed(cls: Type[MODEL], query: str, vec: list[str]):
        ns = await cls.get_namespace()                
        vectorizers = {name: vs.vectorizer for name, vs in  ns.vector_spaces.items() if name in vec}
        tasks = [vectorizer.embed_documents([query]) for _, vectorizer in vectorizers.items()]
        vector_list = await asyncio.gather(*tasks)
        return {vec_name: vec[0] for vec_name, vec in zip(vectorizers.keys(), vector_list)}
    
    
    @classmethod
    async def model_batch_embed(cls: Type[MODEL], objs: List[MODEL]):
        ns = await cls.get_namespace()
        vectorizers = {name: vs.vectorizer for name, vs in  ns.vector_spaces.items()}
        chunks = [obj.model_chunk(vectorizers=vectorizers) for obj in objs]
        vector_list = await asyncio.gather(*[
            vectorizer.embed_documents([c[vec_name] for c in chunks])
            for vec_name, vectorizer in vectorizers.items()
        ])
        vec_names = list(vectorizers.keys())
        embeddings = []
        for vecs in zip(*vector_list):
            embeddings.append({vec_names[i]: vec for i, vec in enumerate(vecs)})
        return embeddings
        
        
    async def save(self):
        if not self._namespace:
            raise ValueError("Namespace not defined")
        # namespace = await connection_manager.get_namespace(self._namespace)
        
        ns = await self.get_namespace()
        vectors = await self.__class__.model_batch_embed([self])
        metadata = self._payload_dump()
        
        if self._id is None:
            res = await ns.conn.upsert(
                    namespace=self._namespace,
                    vectors=vectors,
                    metadata=[metadata],
                    # ids=[self._id]
                    model_cls=self.__class__,
                    is_versioned=ns.versioned,
                    is_head=ns.is_head,
                    is_detached_head=ns.is_detached_head,
                    field_mapper=ns.field_mapper
                )
        else:
            res = await ns.conn.update(
                namespace=self._namespace,
                id=self.id,
                vectors=vectors,
                metadata=[metadata],
                is_versioned=ns.versioned,
                field_mapper=ns.field_mapper
            )
        if not res:
            raise ValueError("Failed to save model")
        if not res[0].get('id'):
            raise ValueError("No id returned from database")
        # for field, field_info in self.model_fields.items():
        self._id = res[0].get('id')
        # self.branch_id = res[0].get('branch_id', -1)
        # self.turn_id = res[0].get('turn_id', -1)
        # self.head_id = res[0].get('head_id', -1)
        if hasattr(self, "after_save") and callable(self.after_save):
            await self.after_save(**res[0])
        self._update_relation_instance_id()
        return self

        
    
    # async def delete(self):
    #     ns = await self.get_namespace()
    #     return await ns.conn.delete(ns.name, ids=[self._id])
    
    @classmethod
    async def batch_delete(cls: Type[MODEL], ids: List[str] | None = None, filters: Callable[[QueryProxy[MODEL]], QueryFilter] | None = None):
        ns = await cls.get_namespace()
        query_filter = None
        if filters:
            query_proxy = QueryProxy[MODEL](cls)
            query_filter = filters(query_proxy)
        return await ns.conn.delete(ns.name, ids=ids, filters=query_filter)
    
    @classmethod
    async def delete(cls: Type[MODEL], id: str | int):
        ns = await cls.get_namespace()
        return await ns.conn.delete(ns.name, ids=[id])
    
    
    
        
        
    @classmethod
    async def batch_upsert(cls: Type[MODEL], points: List[MODEL]):
        vectors = await cls.model_batch_embed(points)
        metadata = [point._payload_dump() for point in points]
        # ids = [point._id for point in points]        
        # ids = [point.id for point in points]
        ns = await cls.get_namespace()
        res = await ns.conn.upsert(
            namespace=cls._namespace.default,# type: ignore
            vectors=vectors,
            metadata=metadata,
            # ids=ids,
            model_cls=cls,
            is_versioned=ns.versioned,
            is_head=ns.is_head,
            is_detached_head=ns.is_detached_head,
            field_mapper=ns.field_mapper
        )
        return res
    
    
    @classmethod
    def pack_search_result(cls: Type[MODEL], search_result) -> MODEL:
        db_type = cls._db_type.default # type: ignore
        inst_id = None
        if db_type == "qdrant":
            result_dict = {
                # "id": search_result.id,
                "score": search_result.score if hasattr(search_result, "score") else -1,
                "vector": search_result.vector if hasattr(search_result, "vector") else {},
                **search_result.payload
            }
            inst_id = search_result.id
        elif db_type == "postgres":
            result_dict = {}
            for field, field_info in cls.model_fields.items():
                field_type = field_info.annotation
                field_value = search_result.get(field)
                if field_value:
                    if field_type == dict or (inspect.isclass(field_type) and issubclass(field_type, BaseModel)):
                        result_dict[field] = json.loads(field_value)
                    else:
                        result_dict[field] = field_value
            inst_id = search_result.get("id")
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        instance = cls(
                **result_dict
            )
        instance._id = inst_id
        instance._update_relation_instance_id()
        return instance
    
    @classmethod
    async def pack_search_result_with_hooks(cls: Type[MODEL], search_result):
        instance = cls.pack_search_result(search_result)
        if hasattr(instance, "after_load") and callable(instance.after_load):
            await instance.after_load(**search_result)
        return instance
    
    @classmethod
    def _get_default_temporal_field(cls):
        if cls._default_temporal_field.default is not None: # type: ignore
            return cls._default_temporal_field.default # type: ignore
        else:
            parent_field = cls.__bases__[0]._get_default_temporal_field()            
            return parent_field
    
    
    @classmethod
    async def get_assets(cls: Type[MODEL], limit: int=10, filters=None, start_from=None, offset=0, ascending=False, ids=None):
        sort_key = cls._get_default_temporal_field()
        if not sort_key:
            raise ValueError("Temporal Field not defined")
        order_by = {
            "key": sort_key,
            "direction": "asc" if ascending else "desc",
            "start_from": start_from
        }
        namespace = cls._namespace.default# type: ignore
        ns = await connection_manager.get_namespace_or_create(namespace)
        res = await ns.conn.scroll(
            collection_name=namespace,
            limit=limit,
            filters=filters,
            order_by=order_by,
            offset=offset,
            ids=ids,
            with_payload=True,
            with_vectors=True
        )
        # res = await rag_documents.get_documents(top_k=top_k, filters=filters, order_by=order_by, offset=offset, ids=ids, with_vectors=True)
        return [cls.pack_search_result(r) for r in res]
    
    @classmethod
    async def get(cls: Type[MODEL], id: str) -> MODEL | None:
        ns = await cls.get_namespace()
        res = await ns.conn.retrieve(
            namespace=ns.name,
            ids=[id],
            field_mapper=ns.field_mapper
        )
        if not res:
            return None
        return await cls.pack_search_result_with_hooks(res[0])
    
    @classmethod
    async def get_many(cls: Type[MODEL], ids: List[str] | List[int]) -> List[MODEL]:
        ns = await cls.get_namespace()
        res = await ns.conn.retrieve(
            namespace=ns.name,
            ids=ids,
            field_mapper=ns.field_mapper
        )
        return await asyncio.gather(*[cls.pack_search_result_with_hooks(r) for r in res])
        # return [cls.pack_search_result(r) for r in res]
    
    @classmethod
    def build_query(cls: Type[MODEL], query_type: QueryType, partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
        partitions = partitions or {}
        if cls._subspace:
            partitions["_subspace"] = cls._subspace.default# type: ignore
        return QuerySet(cls, query_type=query_type, partitions=partitions)
        
    
    @classmethod
    def similar(cls: Type[MODEL], query: str, threshold: float | None = None, vec: list[str] | str | AllVecs = ALL_VECS, partitions: dict[str, str] | None = None, fusion: FusionType="rff") -> QuerySet[MODEL]:
        return cls.build_query("vector").similar(query, threshold, vec)
        # return QuerySet(cls, query_type="vector").similar(query, vec)    
    
    # @classmethod
    # def filter(cls: Type[MODEL], filters: Callable[[Type[MODEL]], QueryFilter]):
    #     return QuerySet(cls, query_type="scroll").filter(filters)
    @classmethod
    # def filter(cls: Type[MODEL], filters: Callable[[QueryProxy[MODEL]], QueryFilter], partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
    def filter(cls: Type[MODEL], filters: Callable[[MODEL], bool], partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
        return cls.build_query("scroll", partitions).filter(filters)
        # return QuerySet(cls, query_type="scroll").filter(filters)
        
    def filter_list(cls: Type[MODEL], filters: QueryListType, partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
        return cls.build_query("scroll", partitions).filter_list(filters)
    
    @classmethod
    def fusion(cls: Type[MODEL], *args, type: FusionType="rff", partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
        return cls.build_query("vector", partitions).fusion(*args, type=type)
        # return QuerySet(cls, query_type="vector").fusion(*args, type=type)
    
    @classmethod
    def first(cls: Type[MODEL], partitions: dict[str, str] | None = None) -> QuerySetSingleAdapter[MODEL]:
        return cls.build_query("scroll", partitions).first()
        # return QuerySet(cls, query_type="scroll").first()
    
    @classmethod
    def last(cls: Type[MODEL], partitions: dict[str, str] | None = None)  -> QuerySetSingleAdapter[MODEL]:
        return cls.build_query("scroll", partitions).last()
        # return QuerySet(cls, query_type="scroll").last()

    
    # @classmethod
    # def all(cls: Type[MODEL], partitions: dict[str, str] | None = None):
        # return cls.build_query("scroll", partitions).all()
        # return QuerySet(cls, query_type="scroll").all()
        
    @classmethod
    def limit(cls: Type[MODEL], limit: int, partitions: dict[str, str] | None = None) -> QuerySet[MODEL]:
        return cls.build_query("scroll", partitions).limit(limit)
     
    
    @classmethod
    def partition(cls: Type[MODEL], partitions: dict[str, str]) -> QuerySet[MODEL]:
        return cls.build_query("scroll", partitions)
    
    
    @classmethod
    async def add(cls: Type[MODEL]):
        pass
    
    
    @classmethod
    async def get_or_create(cls: Type[MODEL]):
        pass
    
    @classmethod
    async def upsert(cls: Type[MODEL]):
        pass
    
    
    @classmethod
    async def create(cls: Type[MODEL], **kwargs: Any) -> MODEL:
        obj = await cls(**kwargs).save()
        return obj
    
    # @classmethod
    # async def update_payload(cls: Type[MODEL], id: str, update: dict):
    #     pass
        
    # def copy(self, with_ids: bool = True):
    #     payload = self._payload_dump()
    #     if not with_ids:
    #         payload.pop("id")
    #     obj = self.__class__(**payload)
    #     if self._vector:
    #         obj._vector = copy.deepcopy(self._vector)        
    #     return obj
    
    def copy(self, *args, with_ids: bool = False, **kwargs):
        payload = self._payload_dump()
        if not with_ids:
            payload.pop("id")
        obj = self.__class__(**payload)
        if self._vector:
            obj._vector = copy.deepcopy(self._vector)        
        return obj
    
    @classmethod
    async def batch_copy(cls: Type[MODEL], objs: List[MODEL]) -> List[MODEL]:
        new_objs = [obj.copy() for obj in objs]
        return await cls.batch_upsert(new_objs)
        
        
    
    @classmethod
    async def delete_namespace(cls):
        await connection_manager.delete_namespace(cls._namespace.default)# type: ignore
    
    
    
    async def get_head(self):        
        head = await ArtifactLog.get_head(self.head_id)
        return head
    

    @classmethod
    def to_block(cls: Type[MODEL]):
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def from_block(cls, block: Any):
        raise NotImplementedError("Subclasses must implement this method")
    