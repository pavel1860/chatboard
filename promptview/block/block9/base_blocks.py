from typing import Any, Generic, Type, TypeVar
from uuid import uuid4




BaseContent = str | int | float | bool | None

class UnsetType:
    __slots__ = ()

    def __repr__(self):
        return "<UNSET>"

CONTENT = TypeVar("CONTENT")


class BaseBlock(Generic[CONTENT]):
    __slots__ = [
        "content",
        "parent",
        "prefix",
        "postfix", 
        "id",      
    ]   
    
    def __init__(
        self,
        content: CONTENT,
        *,
        prefix: CONTENT,
        postfix: CONTENT,
        parent: "BlockSequence | None" = None,
        id: str | None = None,        
    ):
        self.content: CONTENT = content
        self.parent: "BlockSequence | None" = parent
        self.prefix: CONTENT = prefix
        self.postfix: CONTENT = postfix
        self.id: str = id or uuid4().hex[:8]
    
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return []
        if self.index is None:
            raise ValueError("Block has no index")
        return self.parent.path + [self.index]
    
    @property
    def index(self) -> int | None:
        if self.parent is None:
            return None
        return self.parent.index_of(self)
    
    
    def model_dump(self):
        return {
            "_type": self.__class__.__name__,
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "path": [p for p in self.path],
            "parent_id": self.parent.id if self.parent else None,
        }
        
    def repr_tree(self, verbose: bool = False):
        return f"{self.path}  BaseBlock({self.content})"
    
    def print_tree(self, verbose: bool = False):
        print(self.repr_tree(verbose=verbose))
        
        
    def __eq__(self, other: object):
        if isinstance(other, BaseBlock):
            return self.id == other.id
        else:
            return False
        
    def traverse_path(self):
        curr = self
        while curr is not None:
            yield curr
            curr = curr.parent
            
            
    def index_in_parent(self, kind: "Type[BaseBlock] | None" = None) -> int:
        if self.parent is None:
            return 0 if kind is None or type(self) is kind else -1
        idx = -1
        for b in self.parent:
            if kind is None or type(b) is kind:
                idx += 1
            if b == self:
                return idx
        return idx
                
            
            
    


CHILD = TypeVar("CHILD", bound=BaseBlock)
PathType = list[int] | int | str


class BlockSequence(Generic[CONTENT,CHILD], BaseBlock[CONTENT]):
    
    __slots__ = [
        "children", 
        "is_wrapper" 
    ]
    
    def __init__(
        self,
        content: CONTENT,
        children: list[CHILD],
        prefix: CONTENT,
        postfix: CONTENT,
        parent: "BlockSequence | None" = None,
        id: str | None = None,
        is_wrapper: bool = False,

    ):
        BaseBlock.__init__(self, content=content, parent=parent, prefix=prefix, postfix=postfix, id=id)
        self.children: list[CHILD] = []
        self.parent: "BlockSequence | None" = parent
        self.id: str = uuid4().hex[:8]
        self.is_wrapper = is_wrapper
        if children is not None:
            for child in children:
                self.append(child)
                
    def promote_content(self, content: Any, prefix: CONTENT | None = None, postfix: CONTENT | None = None) -> CHILD:
        raise NotImplementedError("promote_content is not implemented")
    
    
    @property
    def is_no_content(self) -> bool:
        if isinstance(self.content, list):
            return len(self.content) == 0
        elif isinstance(self.content, str):
            return len(self.content) == 0
        elif self.content is None:
            return True
        else:
            return False
    
    @property
    def is_empty(self) -> bool:
        return self.is_no_content and len(self.children) == 0

        
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return [0]
        if self.index is None:
            return self.parent.path
        return self.parent.path + [self.index]
    
    
    @property
    def rel_path(self) -> list[int]:       
        if self.is_wrapper:
            return []
        if self.parent is None:
            if self.is_no_content:
                return []
            return [0]
        if self.index is None or self.is_no_content:
            return self.parent.rel_path
        return self.parent.rel_path + [self.index]

    
    @property
    def index(self) -> int | None:
        if self.parent is None:
            return None
        return self.parent.index_of(self)
        
    def index_of(self, child: CHILD) -> int | None:        
        return self.children.index(child)
    
    def _parse_path(self, path: PathType):
        if isinstance(path, str):
            return [int(p) for p in path.split(".")]
        if isinstance(path, int):
            return [path]
        return [p for p in path]
    
    
    # def path_traverse(self, path: list[int] | int):
    
    def path_get(self, path: PathType) -> "BlockSequence | BaseBlock | None":
        _path = self._parse_path(path)
        target = self
        while _path:
            idx = _path.pop(0)
            if not isinstance(target, BlockSequence):
                if len(_path) == 0:
                    return target
                raise ValueError(f"Invalid path: {path}, target is not a BlockSequence and not supporting insert")
            if idx >= len(target.children):
                return None            
            target = target.children[idx]
        return target
    
    
    def path_exists(self, path: PathType) -> bool:
        path = self._parse_path(path)
        target = self
        while path:
            idx = path.pop(0)
            if not isinstance(target, BlockSequence):
                return False
            if idx >= len(target.children):
                return False
            target = target.children[idx]
        return True
    
    
    @property
    def last_child(self) -> CHILD | None:
        if len(self.children) == 0:
            return None
        return self.children[-1]
    
    
    def append_child(self, child: CHILD):
        self.children.append(child)
        child.parent = self        
        return child
    
    def insert_child(self, index: int, child: CHILD):
        self.children.insert(index, child)
        child.parent = self        
        return child
    
    
    def replace_child(self, index: int, child: CHILD):
        child.parent = self
        self.children[index] = child
        return child
    
    def append(
        self, 
        child: CHILD | BaseContent, 
        path: PathType | None = None, 
        prefix: BaseContent | None = None, 
        postfix: BaseContent | None = None
    ):
        child = self.promote_content(child, prefix=prefix, postfix=postfix)
        if not path:
            self.append_child(child)
            return child        
        path = self._parse_path(path)
        target = self.path_get(path)
        if target is None:
            raise ValueError(f"Invalid path: {path}, target not found")
        if not isinstance(target, BlockSequence):
            raise ValueError(f"Invalid path: {path}, target is not a BlockSequence and not supporting append")
        target.append_child(child)
        return child
    
    def extend(
        self, 
        children: list[CHILD],
        path: PathType | None = None,
        prefix: BaseContent | None = None,
        postfix: BaseContent | None = None,
    ):
        for child in children:
            self.append(child, path=path, prefix=prefix, postfix=postfix)
        return self
    
    
    def insert(self, child: CHILD | BaseContent, path: PathType):
        child = self.promote_content(child)
        path = self._parse_path(path)
        if len(path) == 1:
            self.insert_child(path[0], child)
            return child
        idx = path.pop(-1)        
        target = self.path_get(path)
        if target is None:
            raise ValueError(f"Invalid path: {path}, target not found")
        if not isinstance(target, BlockSequence):
            raise ValueError(f"Invalid path: {path}, target is not a BlockSequence and not supporting insert")
        target.insert_child(idx, child)
        return child
    
    
    def replace(self, child: CHILD | BaseContent, path: PathType):
        child = self.promote_content(child)
        path = self._parse_path(path)
        if len(path) == 1:
            self.replace_child(path[0], child)
            return child
        idx = path.pop(-1)
        target = self.path_get(path)
        if target is None:
            raise ValueError(f"Invalid path: {path}, target not found")
        if not isinstance(target, BlockSequence):
            raise ValueError(f"Invalid path: {path}, target is not a BlockSequence and not supporting replace")
        target.replace_child(idx, child)
        return child
    
    
    
    def __iter__(self):
        return iter(self.children)
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, path: PathType):
        path = self._parse_path(path)
        target = self.path_get(path)
        if target is None:
            raise ValueError(f"Invalid path: {path}, target not found")
        return target
    
    # def __eq__(self, other: object):
    #     if isinstance(other, BlockSequence):
    #         return self.children == other.children
    #     else:
    #         return False
        
        
        
    def model_dump(self):
        dump = {
            **super().model_dump(),
            "children": [child.model_dump() for child in self.children],
        }
        
        return dump

    
    
    def repr_tree(self, verbose: bool = False):           
        res =  f"{self.path}  BlockSequence({self.id})"
        for child in self.children:
            res += f"\n{child.repr_tree()}"
        return res
    

        
        
        




