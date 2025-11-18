


from typing import TYPE_CHECKING, Callable, Generic, List, Type, TypeVar, Any, get_args
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler

from .base_namespace import NSManyToManyRelationInfo





if TYPE_CHECKING:
    from promptview.model.base_namespace import NSRelationInfo, Namespace
    from promptview.model.model import Model
    from promptview.model.postgres.query_set3 import SelectQuerySet


FOREIGN_MODEL = TypeVar("FOREIGN_MODEL", bound="Model")

class Relation(Generic[FOREIGN_MODEL], list):
    
    
    def __init__(self, *args, relation: "NSRelationInfo | None" = None, **kwargs):        
        items = []
        self._is_initialized = False 
        self._primary_instance = None
        self._relation = relation
        if relation is not None:
            if len(args) == 1 and isinstance(args[0], list):
                self._is_initialized = True
                
                items = [
                    self.pack_cls(item) if not isinstance(item, relation.foreign_cls) else item
                    for item in args[0]]
        
        super().__init__(items)
        
        
    # def __repr__(self) -> str:
    #     if not self._is_initialized:
    #         return f"Relation({self._relation.primary_cls.__name__} -> {self._relation.foreign_cls.__name__})"
    #     return "Relation[\n" + ",\n".join("  " +repr(item) for item in self) + "\n]"
    
    @property
    def primary_id(self) -> Any:
        if self._primary_instance is None:
            raise ValueError("Primary instance is not set")
        return self._primary_instance.id
    
    @property
    def foreign_table_name(self) -> str:
        if self._relation is None:
            raise ValueError("Relation is not set")
        return self._relation.foreign_namespace.table_name
    
    def set_primary_instance(self, primary_instance: "Model"):
        self._primary_instance = primary_instance
    
    def pack_cls(self, item: Any) -> Any:  
        if self._relation is None:
            raise ValueError("Relation is not set")
        item = self._relation.foreign_namespace.pack_record(item)
        return self._relation.inst_foreign_model(item)


    def query(self) -> "SelectQuerySet[FOREIGN_MODEL]":
        if self._relation is None:
            raise ValueError("Relation is not set")
        
        if isinstance(self._relation, NSManyToManyRelationInfo):
            foreign_cls = self._relation.foreign_cls
            junction_cls = self._relation.junction_cls
            where_clouse = {self._relation.junction_primary_key: self.primary_id}
            return foreign_cls.query().include(junction_cls.query().where(**where_clouse))
        else: 
            where_dict = {self._relation.foreign_key: self.primary_id}
            return self._relation.foreign_cls.query().where(**where_dict).order_by("created_at")
    
    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    #     return core_schema.no_info_plain_validator_function(
    #         cls._validate,
    #         serialization=core_schema.plain_serializer_function_ser_schema(
    #             cls._serialize
    #         )
    #     )
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # extract the generic type parameter
        foreign_model = get_args(source_type)[0]
        list_schema = handler(List[foreign_model])

        return core_schema.with_info_wrap_validator_function(
            cls._validate_with_info,
            list_schema,
            field_name=handler.field_name,            
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )
        
        
    @staticmethod
    def _validate_with_info(value, next_validator, info: core_schema.ValidationInfo) -> "Relation":
        from promptview.model.namespace_manager import NamespaceManager
        if isinstance(value, Relation):
            return value
        if not info.config:
            raise ValueError("Config is not set")        
        model_name = info.config.get('title')
        if not model_name:
            raise ValueError("Model name is not set")        
        field_name = info.field_name
        if not field_name:
            raise ValueError("Field name is not set")
        ns = NamespaceManager.get_namespace_by_model_cls(model_name)   
        relation = ns.get_relation(field_name)     
        if not relation:
            raise ValueError(f"Relation {field_name} not found")
        validated_list = next_validator(value)
        return Relation(validated_list, relation=relation)


    
    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    #     foreign_model = get_args(source_type)[0]
    #     list_schema = handler(List[foreign_model])
    #     return core_schema.with_info_plain_validator_function(
    #         function=cls._validate,
    #         metadata={"field_name": list_schema.metadata["field_name"]},
    #         serialization=core_schema.plain_serializer_function_ser_schema(
    #             cls._serialize
    #         )
    #     )
        
    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type, handler):
    #     foreign_model_type = get_args(source_type)[0]
    #     list_schema = handler(List[foreign_model_type])
    #     return core_schema.general_wrap_validator_function(
    #         cls._validate_with_info,
    #         list_schema,
    #         serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
    #     )

        
        
    # @staticmethod
    # def _validate_with_info(value: Any, next_validator: Callable[[Any], Any], info: core_schema.ValidationInfo) -> "Relation":
    #     parsed = next_validator(value)
    #     relation = info.context.get("relation") if info.context else None
    #     return Relation(parsed, relation=relation)
        
    @staticmethod
    def _validate(value: Any, info: core_schema.ValidationInfo) -> "Relation":
        if isinstance(value, Relation):
            return value
        if isinstance(value, list):
            return Relation[FOREIGN_MODEL](value)
        raise ValueError(f"Invalid value: {value}")
    
    @staticmethod
    def _serialize(instance: "Relation | None") -> list | None:
        if instance is None:
            return None
        return instance
    
    