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
        return self.match(normalized_path)
        # return self.items[normalized_path]
    
    def get(self, path: PathType) -> FOREIGN_MODEL:
        normalized_path = self._normalize_path(path)
        res = self.match(normalized_path)
        return res[0] if res else None
    
    def get_many(self, path: PathType) -> List[FOREIGN_MODEL]:
        normalized_path = self._normalize_path(path)
        return self.match(normalized_path)
    
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
    
    def append(self, item: FOREIGN_MODEL):
        path = self._normalize_path(item.path)
        if path in self.items:
            raise ValueError(f"Item already exists at path {path}")
        self.items[path] = item
        return item
    
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
    def _serialize(instance: "Tree | None") -> dict | None:
        if instance is None:
            return None
        # Serialize each item in the tree
        # If items are Pydantic models, call model_dump() on them
        d = {}
        for path, item in instance.items.items():
            if hasattr(item, 'model_dump'):
                d[path] = item.model_dump()
            elif hasattr(item, 'to_dict'):
                d[path] = item.to_dict()
            else:
                d[path] = item
        return d

    # ========================================================================
    # Pattern Matching Methods
    # ========================================================================

    def match(self, pattern: str) -> List[FOREIGN_MODEL]:
        """
        Get all items matching a pattern

        Examples:
            tree.match("1.*") -> all direct children of "1"
            tree.match("**.message") -> all paths ending with "message"
            tree.match("1.{0,1,2}") -> items at paths "1.0", "1.1", "1.2"
        """
        from .tree_utils import is_pattern, path_to_regex

        if not is_pattern(pattern):
            # Exact match
            return [self.items[pattern]] if pattern in self.items else []

        regex = path_to_regex(pattern)
        matches = [item for path, item in self.items.items() if regex.match(path)]
        return sorted(matches, key=lambda x: getattr(x, 'id', 0))

    def get_children(
        self,
        parent_pattern: str,
        io_kind: Literal["input", "output"] | None = None
    ) -> List[FOREIGN_MODEL]:
        """
        Get direct children of a parent path (supports patterns!)

        Examples:
            tree.get_children("1")
            tree.get_children("1.*")
            tree.get_children("1.1", io_kind="input")
            tree.get_children("1.{0,1}", io_kind="output")
        """
        from .tree_utils import is_pattern, parse_path_level, path_to_regex

        # If it's a simple path (no pattern), use optimized logic
        if not is_pattern(parent_pattern):
            parent_level = parse_path_level(parent_pattern)
            children = [
                item for path, item in self.items.items()
                if path.startswith(parent_pattern + '.') and
                   parse_path_level(path) == parent_level + 1
            ]
        else:
            # For patterns, match parents then find their direct children
            regex = path_to_regex(parent_pattern)
            matched_parents = [path for path in self.items.keys() if regex.match(path)]
            child_paths = set()

            for parent_path in matched_parents:
                parent_level = parse_path_level(parent_path)
                for path in self.items.keys():
                    if (path.startswith(parent_path + '.') and
                        parse_path_level(path) == parent_level + 1):
                        child_paths.add(path)

            children = [self.items[path] for path in child_paths if path in self.items]

        # Filter by io_kind if specified
        if io_kind:
            children = [c for c in children if getattr(c, 'io_kind', None) == io_kind]

        return sorted(children, key=lambda x: getattr(x, 'id', 0))

    def get_descendants(self, parent_pattern: str) -> List[FOREIGN_MODEL]:
        """
        Get all descendants (recursive) of a path (supports patterns!)

        Examples:
            tree.get_descendants("1")
            tree.get_descendants("1.*")
            tree.get_descendants("1.**.message")
        """
        from .tree_utils import is_pattern, path_to_regex

        # If it's a simple path, use optimized logic
        if not is_pattern(parent_pattern):
            descendants = [
                item for path, item in self.items.items()
                if path.startswith(parent_pattern + '.')
            ]
        else:
            # For patterns, match parents then find all their descendants
            regex = path_to_regex(parent_pattern)
            matched_parents = [path for path in self.items.keys() if regex.match(path)]
            descendant_paths = set()

            for parent_path in matched_parents:
                for path in self.items.keys():
                    if path.startswith(parent_path + '.'):
                        descendant_paths.add(path)

            descendants = [self.items[path] for path in descendant_paths if path in self.items]

        return sorted(descendants, key=lambda x: getattr(x, 'id', 0))

    def find_by_kind(
        self,
        kind: str | List[str],
        pattern: str = "**"
    ) -> List[FOREIGN_MODEL]:
        """
        Find items by kind using pattern matching

        Examples:
            tree.find_by_kind("span")
            tree.find_by_kind("block", "1.**")
            tree.find_by_kind(["span", "block"], "1.{0,1}.**")
        """
        kinds = [kind] if isinstance(kind, str) else kind
        values = self.match(pattern)

        result = []
        for v in values:
            item_kind = getattr(v, 'kind', None)
            if item_kind:
                # Handle both single kind and list of kinds
                if isinstance(item_kind, list):
                    if any(k in kinds for k in item_kind):
                        result.append(v)
                elif item_kind in kinds:
                    result.append(v)

        return result

    def find_by_tags(
        self,
        tags: List[str],
        pattern: str = "**",
        match_all: bool = False
    ) -> List[FOREIGN_MODEL]:
        """
        Find items with specific tags

        Examples:
            tree.find_by_tags(["pirate"])
            tree.find_by_tags(["answer", "user"], "1.**")
            tree.find_by_tags(["answer", "pirate"], match_all=True)
        """
        values = self.match(pattern)
        result = []

        for v in values:
            item_tags = None
            kind = getattr(v, 'kind', None)

            if kind in ['block', 'span']:
                value_obj = getattr(v, 'value', None)
                if value_obj:
                    item_tags = getattr(value_obj, 'tags', None)

            if item_tags:
                if match_all:
                    # All tags must be present
                    if all(tag in item_tags for tag in tags):
                        result.append(v)
                else:
                    # Any tag must be present
                    if any(tag in item_tags for tag in tags):
                        result.append(v)

        return result

    def find_spans_by_name(
        self,
        span_name: str,
        pattern: str = "**"
    ) -> List[FOREIGN_MODEL]:
        """
        Find spans with a specific name

        Examples:
            tree.find_spans_by_name("pirate_talk")
            tree.find_spans_by_name("llm", "1.1.**")
        """
        values = self.match(pattern)
        result = []

        for v in values:
            if getattr(v, 'kind', None) == 'span':
                value_obj = getattr(v, 'value', None)
                if value_obj and getattr(value_obj, 'name', None) == span_name:
                    result.append(v)

        return result

    def get_root_items(self) -> List[FOREIGN_MODEL]:
        """
        Get all root level items (paths with no dots)

        Examples:
            tree.get_root_items() -> items at paths "1", "2", "3", etc.
        """
        return self.match("*")

    def get_items_at_level(self, level: int) -> List[FOREIGN_MODEL]:
        """
        Get items at a specific depth level

        Examples:
            tree.get_items_at_level(1) -> root items ("1", "2", "3")
            tree.get_items_at_level(2) -> second level ("1.0", "1.1", "2.0")
            tree.get_items_at_level(3) -> third level ("1.0.message", "1.1.0")
        """
        from .tree_utils import parse_path_level

        items = [
            item for path, item in self.items.items()
            if parse_path_level(path) == level
        ]
        return sorted(items, key=lambda x: getattr(x, 'id', 0))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tree structure

        Returns:
            Dictionary with: total_paths, max_depth, min_depth, avg_depth,
            kind_counts, paths
        """
        from .tree_utils import parse_path_level

        paths = list(self.items.keys())
        if not paths:
            return {
                'total_paths': 0,
                'max_depth': 0,
                'min_depth': 0,
                'avg_depth': 0,
                'kind_counts': {},
                'paths': []
            }

        depths = [parse_path_level(p) for p in paths]
        kinds = [getattr(item, 'kind', 'unknown') for item in self.items.values()]

        # Count kinds (handle both single kind and list of kinds)
        kind_counts: Dict[str, int] = {}
        for kind in kinds:
            if isinstance(kind, list):
                k = '+'.join(kind)
            else:
                k = str(kind)
            kind_counts[k] = kind_counts.get(k, 0) + 1

        return {
            'total_paths': len(paths),
            'max_depth': max(depths),
            'min_depth': min(depths),
            'avg_depth': sum(depths) / len(depths),
            'kind_counts': kind_counts,
            'paths': sorted(paths)
        }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
ContainerType = Tree | List