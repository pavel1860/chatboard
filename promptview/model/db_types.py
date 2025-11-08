from typing import TYPE_CHECKING, Callable, Dict, Generic, Iterator, List, Literal, Type, TypeVar, Any, get_args
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler


if TYPE_CHECKING:
    from .model3 import Model
    from .relation_info import RelationInfo



FOREIGN_MODEL = TypeVar("FOREIGN_MODEL", bound="Model")
PathType = list[int] | tuple[int, ...] | int | str



class Tree(Generic[FOREIGN_MODEL]):
    items: Dict[str, FOREIGN_MODEL]  # Keys are normalized string paths
    
    @staticmethod
    def _normalize_path(path: PathType) -> str:
        """Normalize path to string for comparison and lookup."""
        if isinstance(path, list) or isinstance(path, tuple):
            return '.'.join(str(p) for p in path)
        elif isinstance(path, int):
            return str(path)
        return path
    
    def __init__(self, items: List[FOREIGN_MODEL], relation: "RelationInfo"):
        # Sort items by path before storing in dict to maintain order
        sorted_items = sorted(items, key=lambda item: self._normalize_path(item.path))  # type: ignore[attr-defined]
        # Normalize paths to strings when storing to ensure consistent lookup
        self.items = {self._normalize_path(item.path): item for item in sorted_items}  # type: ignore[attr-defined]
        self.relation = relation
        
    def __getitem__(self, path: PathType) -> FOREIGN_MODEL:
        normalized_path = self._normalize_path(path)
        return self.items[normalized_path]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Iterator[FOREIGN_MODEL]:
        return iter(self.items.values())
    
    def print(self, show_input: bool = False):
        for path, item in self.items.items():
            if not show_input and path.endswith(".input"):
                continue
            print(f"{path}: {item}")
            
    def print_values(self, show_input: bool = False):
        for path, item in self.items.items():
            if not show_input and path.endswith(".input"):
                continue
            print(f"{path}: {item.kind} {item.artifact_id}")
    
    def __repr__(self):
        s = "Tree(\n"
        for path, item in self.items.items():
            s += f"  {path}: {item},\n"
        s += ")"
        return s
    
    def __str__(self):
        return self.__repr__()
    
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
    def _validate_with_info(value, next_validator, info: core_schema.ValidationInfo) -> "Tree[FOREIGN_MODEL]":
        from .namespace_manager2 import NamespaceManager
        if isinstance(value, Tree):
            return value
        if not info.config:
            raise ValueError("Config is not set")        
        model_name = info.config.get('title')
        if not model_name:
            raise ValueError("Model name is not set")        
        field_name = info.field_name
        if not field_name:
            raise ValueError("Field name is not set")
        ns = NamespaceManager.get_namespace_for_model(model_name)   
        relation = ns.get_relation(field_name)     
        if not relation:
            raise ValueError(f"Relation {field_name} not found")
        validated_list = next_validator(value)
        return Tree[FOREIGN_MODEL](validated_list, relation=relation)
    
    @staticmethod
    def _validate(value: Any, info: core_schema.ValidationInfo) -> "Tree[FOREIGN_MODEL]":
        # This method is not used - _validate_with_info is used instead
        # Keeping for compatibility but it requires relation which we don't have here
        if isinstance(value, Tree):
            return value
        raise ValueError(f"Invalid value: {value}. Use _validate_with_info instead.")
    
    @staticmethod
    def _serialize(instance: "Tree | None") -> list | None:
        if instance is None:
            return None
        # Return list of items in sorted order (dict maintains insertion order)
        return list(instance.items.values())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
ContainerType = Tree | List