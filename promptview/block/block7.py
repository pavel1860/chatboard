from collections import UserList
import copy
import json
import textwrap
from typing import TYPE_CHECKING, Annotated, Any, Callable, Generic, List, Literal, Protocol, Sequence, Set, Type, TypeVar, TypedDict, Unpack
import annotated_types
from pydantic_core import core_schema
from pydantic import BaseModel, GetCoreSchemaHandler
from .types import ContentType
if TYPE_CHECKING:
    from ..model import BlockModel





CHUNK_TYPE = TypeVar("CHUNK_TYPE", str, int, float, bool, None)


def process_basic_content(parent: "BlockSequence", content: "ContentType | BlockChunk"):
    if isinstance(content, BlockChunk):
        return content
    elif isinstance(content, str):
        return BlockChunk(content)
    elif isinstance(content, int):
        return BlockChunk(str(content))
    elif isinstance(content, float):
        return BlockChunk(str(content))
    elif isinstance(content, bool):
        return BlockChunk(str(content))
    # elif isinstance(content, list):
    #     return BlockList(content)
    # elif isinstance(content, BaseBlock):
    #     return content        
    else:
        # path = parent.path
        raise ValueError(f"Invalid content type: {type(content)}")


def process_sentence_content(parent: "BaseBlock", content: "ContentType | BlockChunk | BlockSent"):
    if isinstance(content, BlockSent):
        return content
    else: 
        block = process_basic_content(parent, content)
        return BlockSent(items=[block], sep="", parent=parent)







class ContextStack:
    """
    A stack-based context for managing nested block structures.
    """
    _ctx_stack: "list[BlockChunk]"
    
    def __init__(self):
        self._ctx_stack = []
        
        
    @property
    def top(self):
        return self._ctx_stack[-1]
    
    @property
    def root(self) -> "BlockChunk":
        if not self._ctx_stack:
            raise ValueError("No context stack")
        return self._ctx_stack[0]
        
    def __getitem__(self, idx: int) -> "BlockChunk":
        return self._ctx_stack[idx]
    
    def __len__(self) -> int:
        return len(self._ctx_stack)
    
    
    
    def push(self, block: "BlockChunk"):
        self._ctx_stack.append(block)
        
    def pop(self):
        return self._ctx_stack.pop()
    


def children_to_blocklist(children: list["BlockChunk"] | tuple["BlockChunk", ...] | None) -> "BlockList":
    if children is None:
        return BlockList([])
    if isinstance(children, BlockList):
        return children
    return BlockList(list(children))


def parse_style(style: str | List[str] | None) -> List[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []



class BaseBlockParams(TypedDict, total=False):
    index: int | None
    parent: "BaseBlock | None"

class BlockChunkParams(BaseBlockParams):
    content: ContentType | None
    sep: str
    logprob: float | None
    is_end_of_line: bool


class BlockParams(BaseBlockParams):
    role: str | None
    tags: list[str] | None
    style: str | None
    attrs: dict[str, "str | FieldAttrBlock"] | None     
    id: str | None
    styles: list[str] | None
    
    
class FieldBlockParams(BlockParams):
    type: Type
    attrs: dict[str, "str | FieldAttrBlock"] | None
    
def all_slots(cls):
    slots = []
    for base in cls.__mro__:
        if '__slots__' in base.__dict__:
            slots.extend(base.__slots__)
    return slots



def get_attrs(attrs: dict[str, "str | FieldAttrBlock"] | None) -> "dict[str, FieldAttrBlock]":
    if attrs is None:
        return {}
    return {k: v if isinstance(v, FieldAttrBlock) else FieldAttrBlock(name=k, description=v) for k, v in attrs.items()}

class BaseBlock:
    
    __slots__ = [
        "id",
        "index",
    ]
    
    
    def __init__(
        self, 
        index: int | None = None, 
        id: str | None = None,
    ):
        self.index = index if index is not None else 0
        self.id = id
    
    @property
    def path(self) -> list[int]:
        raise NotImplementedError("Not implemented")


    
        
    def model_dump(self):
        dump = {}
        for slot in all_slots(type(self)):
            if slot == "parent":
                continue
            if hasattr(self, slot):
                value = getattr(self, slot)
                if value is None:
                    continue
                dump[slot] = value
        dump["_type"] = self.__class__.__name__
        return dump
    
    
    def model_dump_json(self):
        import json
        return json.dumps(self.model_dump())
    

    
    def render(self, verbose: bool = False) -> str:
        from .renderers_base3 import render
        result = render(self)
        return result if result is not None else ""
    
    def print(self, verbose: bool = False):
        print(self.render(verbose=verbose))
        
        
    def set_index(self, index: int):
        self.index = index
        return self
    
    def __str__(self) -> str:
        out = self.render()
        if out is None:
            return "None"
        return out
    
    



class BlockList(UserList[BaseBlock], BaseBlock):
    __slots__ = [
        "default_sep",
    ]
    
    def __init__(
        self, 
        blocks: Sequence[BaseBlock] | None = None, 
        index: int | None = None,
        parent: "BaseBlock | None" = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        attrs: "dict[str, FieldAttrBlock] | None" = None,
        id: str | None = None,
        styles: list[str] | None = None,
        sep: str = " ",
    ):
        if blocks is None:
            blocks = []

        UserList.__init__(self, blocks)
        BaseBlock.__init__(self, parent=parent, index=index)
        for block in blocks:
            block.parent = self.parent
        self.default_sep = sep
            
            
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            raise ValueError("BlockList has no parent")
        return self.parent.path
    
    @property
    def logprob(self) -> float | None:
        return sum(block.logprob for block in self if block.logprob is not None)
    
    
    @property
    def last(self) -> "BlockSent":
        if not self:
            UserList.append(self, BlockSent(sep=self.default_sep, parent=self))
        return self[-1]
    
    
    def model_dump(self):
        dump = super().model_dump()
        dump["_type"] = "BlockList"
        dump["children"] = [b.model_dump() for b in self]
        return dump
    
    def append(self, content: "BlockSent | Block"):
        if isinstance(content, BlockSent):            
            UserList.append(self, content)
            self._connect(content)
        elif isinstance(content, Block):
            UserList.append(self, content)
            self._connect(content)
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        return self
    
    def append_inline(self, content: "BlockChunk"):
        if not isinstance(content, BlockChunk):
            raise ValueError(f"Invalid content type: {type(content)}")
        if self._should_add_sentence():
            self._add_sentence()
        self.last.append(content)
        return self
    
    def append_stream(self, content: "BlockChunk") -> tuple["BlockChunk", "BlockSent | None"]:
        if not isinstance(content, BlockChunk):
            raise ValueError(f"Invalid content type: {type(content)}")
        sent = None
        if self._should_add_sentence():
            sent = self._add_sentence()
        self.last.append(content)
        return content, sent
    
    
    def add_line(self):
        sent = self._add_sentence()
        return sent
    
    def _add_sentence(self):
        sent = BlockSent(sep=self.default_sep)
        UserList.append(self, sent)
        self._connect(sent)
        return sent
    
    def _connect(self, block: "BaseBlock"):
        block.parent = self
        block.index = len(self) - 1
        return block
    
    def _should_add_sentence(self):
        if not len(self):
            return True
        if not isinstance(self.last, BlockSent):
            return True
        if self.last.has_eol:
            return True
        return False
    
    @classmethod
    def model_validate(cls, data: dict):
        if "_type" not in data or data["_type"] != "BlockList":
            raise ValueError(f"Invalid or missing _type for BlockList: {data.get('_type')}")
        data = dict(data)
        data.pop("_type")
        data.pop("path")
        blocks_data = data.pop("blocks", [])
        blocks = [BlockChunk.model_validate(b) for b in blocks_data]
        return cls(blocks=blocks, **data)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    def get(self, tag: str):
        for block in self:
            if block.tags and tag in block.tags:
                return block
            elif isinstance(block, Block):
                return block.get(tag)
        return None
        
        
    
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, BlockList):
            return v
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, BlockChunk):
                    raise ValueError(f"Invalid block list: {v}")
            return BlockList(v)
        else:
            raise ValueError(f"Invalid block list: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, BlockList):
            return [item.model_dump() for item in v]
        else:
            raise ValueError(f"Invalid block list: {v}")
        
    def __repr__(self) -> str:
        return f"BlockList({super().__repr__()})"

 

class BlockChunk(BaseBlock):
    
    __slots__ = [
        "content",
        "children",
        "logprob",
    ]
    
    
    def __init__(
        self, 
        content: ContentType | None = None,
        logprob: float | None = None,   
        index: int | None = None,
        id: str | None = None,
    ):
        """
        Basic component of prompt building. 
        """
        super().__init__(id=id, index=index)
        self.content: ContentType = content        
        self.logprob: float | None = logprob
        # self.is_end_of_line = False
        # if is_end_of_line:
        #     self.is_end_of_line = is_end_of_line        
        # else:
        #     #! if content is a string, check if it ends with a newline
        #     if self._detect_end_of_line(content):
        #         self.sep = "\n"
        #         self.is_end_of_line = True
        
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return []
        return self.parent.path + [self.index]
        
    @property
    def has_eol(self) -> bool:
        if isinstance(self.content, str):
            return self.content.endswith("\n")
        return True
        
    
    def _detect_end_of_line(self, content: ContentType):
        if type(content) is str:
            if content.endswith("\n"):
                content = content[:-1]            
                return True
        return False
    
    def copy(self, with_parent: bool = True):
        if with_parent:
            return BlockChunk(self.content, logprob=self.logprob, parent=self.parent, index=self.index, id=self.id)
        else:
            return BlockChunk(self.content, logprob=self.logprob, index=self.index, id=self.id)
    
    def replace(self, old: str, new: str):
        if isinstance(self.content, str):
            self.content = self.content.replace(old, new)
        else:
            raise ValueError(f"Cannot replace content of type {type(self.content)}")
        return self
                 
    def __getitem__(self, key):
        if isinstance(key, slice) and isinstance(self.content, str):
            # Special-case for "[:-1]"
            if key.start is None and key.stop == -1 and key.step is None:
                return self.content[:-1]
            # Otherwise, fall back to normal slicing
            return self.content[key]
        else:
            # Normal indexing
            if isinstance(self.content, str):
                return self.content[key]
            else:
                raise ValueError(f"Cannot index content of type {type(self.content)}")
    
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    
    def __eq__(self, other: object):
        if isinstance(other, str):
            return self.content == other
        elif isinstance(other, BlockChunk):
            return self.content == other.content
        else:
            return False
        
    
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, BlockChunk):
            return v
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, BlockChunk):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")
    

        
    def to_model(self) -> "BlockModel":
        from ..model import BlockModel
        return BlockModel.from_block(self)
    
    def model_dump(self):
        dump = super().model_dump()
        dump["_type"] = "BlockChunk"
        dump["content"] = self.content
        dump['logprob'] = self.logprob
        # dump["path"] = self.path         
        return dump
    
        
        return {
            "_type": self.__class__.__name__,
            "content": self.content,
            "children": [c.model_dump() for c in self.children],
            "role": self.role,
            "tags": self.tags,
            "styles": self.styles,
            "attrs": self.attrs,
            "depth": self.depth,
            "parent": self.parent,
            "run_id": self.run_id,
            "model": self.model,
            "tool_calls": [tool.model_dump() for tool in self.tool_calls],
            "usage": self.usage.model_dump() if self.usage else None,
            "id": self.id,
            "db_id": self.db_id,
            "sep": self.sep,
        }
    
    
    @classmethod
    def model_validate(cls, data: dict):
        if "_type" not in data or data["_type"] != "BlockChunk":
            raise ValueError(f"Invalid or missing _type for BlockChunk: {data.get('_type')}")
        data = dict(data)  # copy
        data.pop("_type")
        content = data.pop("content", None)        
        block = cls(content, **data)
        return block
    
    
    def __str__(self):
        if isinstance(self.content, str):
            return self.content
        else:
            return str(self.content)
    
    def __repr__(self):
        return f"BlockChunk(content={self.content}, logprob={self.logprob})"




SEQUENCE_ITEM = TypeVar("SEQUENCE_ITEM", bound=BaseBlock)


class BlockSequence(Generic[SEQUENCE_ITEM], BaseBlock):
    
    __slots__ = [
        "sep_list",
        "has_eol",
        "children",
        "default_sep",
        "parent",
    ]
    
    
    def __init__(
        self, 
        items: list[SEQUENCE_ITEM] | None = None, 
        sep: str = " ", 
        wrap: tuple[str, str] | None = None,
        index: int | None = None,        
        has_eol: bool = False,
        sep_list: list[str] | None = None,  
        default_sep: str = " ",
        parent: "BlockSequence | None" = None,
        id: str | None = None,
    ):
        BaseBlock.__init__(self, index, id)
        if items is None:
            items = []        
        self.parent = parent
        self.sep_list = []
        self.default_sep = default_sep
        self.has_eol = has_eol
        self.children: list[SEQUENCE_ITEM] = []
        if sep_list is not None:
            for chunk, sep in zip(items, sep_list):
                self.append(chunk, sep)
        else:
            for chunk in items:
                self.append(chunk)
        if wrap:
            self.wrap(*wrap)
            
            
    def process_content(self, content: "ContentType | SEQUENCE_ITEM", sep: str, index: int | None = None) -> tuple[SEQUENCE_ITEM, str]:
        raise NotImplementedError("process_content is not implemented")
            
            
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, index: int):
        return self.children[index]
    
    
    def __iter__(self):
        return iter(self.children)
    
    
    def __eq__(self, other: object):
        if isinstance(other, BlockSequence):
            return self.children == other.children
        else:
            return False
        
        

        
    @property
    def logprob(self) -> float | None:
        return sum(item.logprob for item in self.children if item.logprob is not None)
    
    
    def wrap(self, start_token: str, end_token: str):
        self.prepend(start_token)
        self.append(end_token)
    
    
    def _connect(self, item: "SEQUENCE_ITEM", index: int | None = None):
        if hasattr(item, "parent"):
            item.parent = self
        item.index = index if index is not None else len(self.children)
        return item
    
    def append(self, content: ContentType | SEQUENCE_ITEM, sep: str | None = None):
        sep = sep if sep is not None else self.default_sep
        item, sep = self.process_content(content, sep)        
        # if item.is_eol:
        #     if self.has_eol:
        #         raise ValueError("Cannot append to a list that has an end of line")
        #     sep = ""
        #     self.has_eol = True
        #     item = item.replace("\n", "")
        item = self._connect(item)
        self.sep_list.append(sep)
        self.children.append(item)
        return self
    
    def prepend(self, content: ContentType | SEQUENCE_ITEM, sep: str | None = None):
        self.insert(0, content, sep=sep)
        return self
        
    # def join(self) -> str:
    #     def sep_gen():
    #         length = len(self.children)
    #         for i, (text, sep) in enumerate(zip(self.children, self.sep_list)):
    #             yield text
    #             if i < length - 1:
    #                 yield sep
    #     return "".join(sep_gen())
        
        
        
    def insert(self, index: int, content: ContentType | SEQUENCE_ITEM, sep: str | None = None):
        if self.has_eol and index == len(self.children) - 1:
            raise ValueError("Cannot insert to the end of a list that has an end of line")
        item, sep = self.process_content(content, sep, index)
        sep = sep if sep is not None else self.default_sep
        # if item.is_eol:
        #     if index != len(self.children) - 1:
        #         raise ValueError("end of line cannot be inserted in the middle of a list")
        #     if self.has_eol:
        #         raise ValueError("Cannot insert an end of line to a list that has an end of line")
        #     sep = "\n"
        #     self.has_eol = True
        self._shift_index(index)
        item = self._connect(item, index)
        self.sep_list.insert(index, sep)
        self.children.insert(index, item)
        return self
        
    def extend(self, seq: "BlockSequence"):
        for item in seq:
            self.append(item)
        
    # def _connect(self, item: "SEQUENCE_ITEM", index: int | None = None):
    #     item.parent = self
    #     item.index = index if index is not None else len(self.children) - 1
    #     return item
    
    def _shift_index(self, index: int):
        for i in range(index, len(self.children)):
            self.children[i].index += 1
        
        
    def iter_chunks(self, use_last_sep: bool = False):
        if len(self.children) != len(self.sep_list):
            raise ValueError(f"Number of chunks and separators must match: {len(self.children)} != {len(self.sep_list)}")
        for i, (chunk, sep) in enumerate(zip(self.children, self.sep_list)):
            # if not use_last_sep and i == len(self.children) - 1:
            #     yield chunk, ""
            if i == 0:
                yield "", chunk
            else:
                yield sep, chunk
        
    



class BlockSent(BlockSequence[BlockChunk]):
    
    
    def __init__(
        self,
        content: list[BlockChunk | str] | str | None = None,
        sep: str = " ",
        wrap: tuple[str, str] | None = None,
        index: int | None = None,
        has_eol: bool = False,
        sep_list: list[str] | None = None,
        parent: "BlockSequence | None" = None,
        id: str | None = None,
        default_sep: str = " ",
    ):
        chunks = []
        if isinstance(content, list):
            chunks = content
        elif content is not None:
            chunk, sep = self.process_content(content, sep, index)
            chunks = [chunk]
        super().__init__(chunks, sep, wrap, index, has_eol, sep_list, parent=parent, id=id)
        
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return []
        if self.index < 0:
            return self.parent.path + [self.index]
        return self.parent.path
        # return self.parent.path + [self.index]
        
    def process_content(self, content: "ContentType | BlockChunk", sep: str, index: int | None = None) -> tuple[BlockChunk, str]:
        chunk = process_basic_content(self, content)
        if chunk.has_eol:            
            if index is not None and index != len(self.children) - 1:
                raise ValueError("end of line cannot be inserted in the middle of a list")
            if self.has_eol:
                raise ValueError("Cannot insert an end of line to a list that has an end of line")
            chunk.content = chunk.content.replace("\n", "")
            # sep = "\n"
            self.has_eol = True
        return chunk, sep
    
    def __and__(self, other: ContentType | BlockChunk):
        self.append(other, sep="")
        return self
    
    def __iand__(self, other: ContentType | BlockChunk):
        self.append(other, sep="")
        return self
    
    def __add__(self, other: "ContentType | BlockChunk | BlockSent"):
        if isinstance(other, BlockSent):
            s = BlockSent()
            s.extend(self)
            s.extend(other)
            return s
        else:
            self.append(other)
        return self
    
    def __iadd__(self, other: "ContentType | BlockChunk | BlockSent"):
        if isinstance(other, BlockSent):
            self.extend(other)
        else:
            self.append(other)
        return self
    
    def __radd__(self, other: "ContentType | BlockChunk | BlockSent"):
        if isinstance(other, BlockSent):
            pass
        else:
            self.prepend(other)
        return self
    
    def __iradd__(self, other: ContentType):
        self.append(other)
        return self
    
        
        
    
    def model_dump(self):
        dump = super().model_dump()
        dump["_type"] = "BlockSent"
        # dump["children"] = [b.model_dump() for b in self]
        dump["children"] = []
        for child in self.children:
            cdump = child.model_dump()
            cdump["path"] = self.path + [self.index]
            dump["children"].append(cdump)
        dump["path"] = self.path
        return dump
    

    @classmethod
    def model_validate(cls, data: dict):
        if "_type" not in data or data["_type"] != "BlockSent":
            raise ValueError(f"Invalid or missing _type for BlockSent: {data.get('_type')}")
        data = dict(data)
        data.pop("_type")
        data.pop("path")
        blocks_data = data.pop("children", [])
        blocks = [BlockChunk.model_validate(b) for b in blocks_data]
        return cls(blocks, **data)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, BlockList):
            return v
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, BlockChunk):
                    raise ValueError(f"Invalid block list: {v}")
            return BlockList(v)
        else:
            raise ValueError(f"Invalid block list: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, BlockList):
            return [item.model_dump() for item in v]
        else:
            raise ValueError(f"Invalid block list: {v}")
        
    def __repr__(self) -> str:
        return f"BlockSent({super().__repr__()})"






def parse_content(content: ContentType | BaseBlock | list[str] | None = None, **kwargs: Unpack[BlockParams]) -> BlockChunk:
    if content is None:
        return BlockChunk(**kwargs)
    elif isinstance(content, BlockChunk):
        return content
    elif isinstance(content, list):
        return BlockChunk(content, **kwargs)
    elif isinstance(content, str):
        return BlockChunk(content, **kwargs)
    else:
        raise ValueError(f"Invalid content type: {type(content)}")
      
      
        
class Block(BlockSequence["Block"]):
    
    __slots__ = [
        "root",
        "children",
        "role",        
        "tags",
        "styles",      
        "attrs",
    ]
    

    def __init__(
        self, 
        root: ContentType | BlockChunk | BlockSent | None = None, 
        children: "list[Block | BlockSent] | None" = None,                 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        attrs: "dict[str, FieldAttrBlock] | None" = None,
        id: str | None = None,
        styles: list[str] | None = None,
        index: int | None = None,
        parent: "BlockSequence | None" = None,
        default_sep: str = "\n",        
    ):
        super().__init__(parent=parent, index=index, id=id, default_sep=default_sep)
        self.role: str | None = role
        self.tags: list[str] = tags or []
        self.styles: list[str] | None = styles or parse_style(style)
        self.attrs: dict[str, FieldAttrBlock] = get_attrs(attrs)        
        self.root = BlockSent(index=self.index, parent=self, id=id)
        # self.children = BlockList(children or [], index=self.index, parent=self)
        
        if isinstance(root, BlockSent):
            self.root = root
            self.root.parent = self            
        else:
            root_block = process_basic_content(self, root) if root is not None else None
            if root_block is not None:
                self.root.append(root_block)
                
                
    def process_content(self, content: "ContentType | Block | BlockSent", sep: str, index: int | None = None) -> "tuple[Block, str]":
        if isinstance(content, Block):
            pass
        elif isinstance(content, BlockSent):
            content = Block(root=content)
        else:
            chunk = process_basic_content(self, content)
            # content = BlockSent(content=[chunk], sep=sep, parent=self)
            content = Block(root=chunk)
        return content, sep
        
        
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return [self.index]
        return self.parent.path + [self.index]

    @classmethod
    def model_validate(cls, data: dict):
        if "_type" not in data or data["_type"] != "BlockContext":
            raise ValueError(f"Invalid or missing _type for BlockContext: {data.get('_type')}")
        data = dict(data)
        data.pop("_type")
        data.pop("path")
        root_data = data.pop("root", None)
        children_data = data.pop("children", [])
        root = BlockSent.model_validate(root_data) if root_data else BlockSent()
        # children = BlockList.model_validate({"_type": "BlockList", "blocks": children_data}) if children_data else BlockList()
        children = [Block.model_validate(c) for c in children_data]
        return cls(root=root, children=children, **data)
    
    @property
    def logprob(self) -> float | None:
        logprob = sum(block.logprob for block in self.children if block.logprob is not None) or 0
        root_logprob = self.root.logprob or 0
        return logprob + root_logprob
       
    def field(self, name: str, type: Type, attrs: dict[str, str] | None = None) -> "BlockSchemaField":
        # block = FieldBlock(name, type, attrs=attrs)
        ctx = BlockSchemaField(
            name,
            type=type,
            attrs=attrs,
            role=self.role,
            parent=self.parent,            
            styles=["xml"],
        )
        self.append(ctx)
        return ctx
    
    def attr(
        self, 
        name: str,
        type: Type, 
        description: str,
        gt: int | float | None = None,
        lt: int | float | None = None,
        ge: int | float | None = None,
        le: int | float | None = None,
    ) -> "None":
        if gt is not None: annotated_types.Gt(gt)
        if lt is not None: annotated_types.Lt(lt)
        if ge is not None: annotated_types.Ge(ge)
        if le is not None: annotated_types.Le(le)
        self.attrs[name] = FieldAttrBlock(
            name=name,
            type=type,
            description=description,
            gt=gt,
            lt=lt,
            ge=ge,
            le=le,
        )
        
    def _should_add_sentence(self):
        if not len(self):
            return True
        if not isinstance(self.last, BlockSent):
            return True
        if self.children[-1].has_eol:
            return True
        return False  
        
    # def iappend(self, content: "ContentType | Block | BlockSent", sep: str = " "):
    #     last = self.children[-1]
        
            
    #     else:
    
    
    def path_append(
        self, 
        path: list[int],
        content: "ContentType | Block | BlockSent",
        sep: str = " ",        
    ): 
        if path is not None:
            target = self.get_path(path)
            if target is None:
                raise ValueError(f"Path {path} not found")
            target.append(content, sep)
        else:
            super().append(content, sep)
        return self
    
    
    def is_last_eol(self) -> bool:
        if len(self.children) == 0:
            return True
        return self.children[-1].root.has_eol
    
    
    
    
    def inline_append(self, content: "ContentType | Block | BlockSent", sep: str = " "):        
        if isinstance(content, Block):
            self.append(content, sep)
            return self.children[-1]
        elif isinstance(content, BlockSent):
            block = Block(root=content)
            self.append(block, sep)
            return self.children[-1]
        else:
            if self.is_last_eol():
                self.append(content, sep)
                return self.children[-1]
            else:
                last = self.children[-1]
                if last.root.has_eol:
                    self.append(content, sep)
                    return self.children[-1]
                else:
                    last.root.append(content, sep)
                    return last.root.children[-1]
        
        
            
    
    # def path_insert(self, path: list[int], content: "ContentType | Block | BlockSent", sep: str = " "):        
    #     if len(path) > 1:
    #         target = self.get_path(path)
    #     else:
    #         target = self
        
    #     if target is None:
    #         raise ValueError(f"Path {path} not found")
    #     if isinstance(target, BlockSent) and not type(content) in (BlockSent, ContentType):
    #             raise ValueError(f"Cannot insert to a list that has an end of line")
    #     if path[-1] >= len(target):            
    #         target.append(content, sep)
    #     else:            
    #         target.insert(path[-1], content, sep)
        
    #     return self
    def path_insert(self, path: list[int], content: "ContentType | Block | BlockSent", sep: str = " "):        
        target = self        
        while path:
            index = path.pop(0)
            if index >= len(target.children):
                if not path:
                    target.append(content, sep)
                else:
                    raise ValueError(f"Path {path} not found")
            else:
                target = target.children[index]        
        
        return self
            
    
    def path_exists(self, path: list[int]) -> bool:
        if len(path) == 0:
            return True
        target = self.get_path(path)
        return target is not None

    
    def response_schema(self, name: str | None = None):
        name = name or "response_schema"
        block = BlockSchema(tags=[name])
        self.append(block)
        return block
    
    
    def traverse(self):
        yield self
        for child in self.children:
            if isinstance(child, Block):
                yield from child.traverse()
            else:
                yield child
    
    
    def get(self, tag: str):
        tag = tag.lower()
        if tag in self.tags:
            return self
        for child in self.children:
            if isinstance(child,Block):
                if tag in child.tags:
                    return child                            
                if (block:= child.get(tag)) is not None:
                    return block
        return None
    
    
    def get_path(self, path: list[int]) -> "Block | BlockSent | None":
        index, sub_path = path[0],path[1:]
        if len(self.children) <= index:
            return None
        target = self.children[index]
        if len(sub_path) == 0:
            return target
        if len(sub_path) == 1:
            if isinstance(target, Block):
                return target.get_path(sub_path)
            elif isinstance(target, BlockSent):
                return target[sub_path[0]]
        elif len(sub_path) > 1:
            if isinstance(target, Block):
                return target.get_path(sub_path)
            elif isinstance(target, BlockSent):
                raise ValueError(f"Invalid path: {path}")
        else:
            raise ValueError(f"Invalid path: {path}")
        
        
    # def insert(self, path: list[int], content: "ContentType | BaseBlock"):    
    #     target = self if not path else self.get_path(path)
    #     if isinstance(target, Block):
    #         target.append_child(content)
    #     elif isinstance(target, BlockSent):
    #         target.append(content)
    #     else:
    #         raise ValueError(f"Invalid path: {path}")
            
        
        

    # def get_field(self):
    #     for child in self.children:
    #         if isinstance(child, BlockSchemaField):
    #             return child
    #         elif isinstance(child, Block):
    #             for b in child.root:
    #                 if isinstance(b, BlockSchemaField):
    #                     return b
    #     return None
    
    def model_dump(self):
        dump = super().model_dump()
        # dump = self.base_model_dump()
        dump["_type"] = "Block"
        dump["root"] = self.root.model_dump()
        dump["children"] = [c.model_dump() for c in self.children]
        dump["styles"] = self.styles
        dump["tags"] = self.tags
        dump["attrs"] = self.attrs
        dump["role"] = self.role
        dump["id"] = self.id
        dump["path"] = self.path
        
        return dump
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, Block):
            return v
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, BlockChunk):
                    raise ValueError(f"Invalid block list: {v}")
            return BlockList(v)
        else:
            raise ValueError(f"Invalid block list: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block list: {v}")
    
    # def build_response(self) -> "ResponseBlock":
    #     def _clone_target_node(n: Block) -> ResponseBlock:
    #         return ResponseBlock(                
    #             schema=n, 
    #             tags=n.tags,                           
    #         )
        
    #     def _is_target(node: BaseBlock) -> bool:
    #         return isinstance(node, FieldSchemaBlock)
    #     res = self.gather_trees(_is_target, _clone_target_node)
    #     if len(res) == 1:
    #         return res[0]
    #     else:
    #         raise ValueError("Multiple target nodes found")
        
    def gather_trees(self, is_target: Callable[[BaseBlock], bool] | None = None, clone_target_node = None) -> "list[BaseBlock]":
        """Return a forest containing only target-type nodes, attached under their
        nearest target-type ancestor from the original tree."""
        dummy_children: List[BaseBlock] = []
        stack: List[Block] = []  # stack of cloned target nodes

        def dfs(u: Block):
            created = None
            if is_target(u):
                created = clone_target_node(u)
                if stack:
                    stack[-1].append(created)
                else:
                    dummy_children.append(created)
                stack.append(created)

            if isinstance(u, Block):
                for child in u.children:
                    dfs(child)
                    
            if created is not None:
                stack.pop()

        dfs(self)
        if not dummy_children:
            raise ValueError("No target nodes found")
        # res = Block(children=BlockList(dummy_children))
        return dummy_children
    
    
    def __enter__(self):        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    
    def __call__(
        self, 
        content: ContentType | BaseBlock | list[str] | None = None, 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        attrs: "dict[str, FieldAttrBlock] | None" = None,
    ) -> "Block":        
        block = Block(
            content,
            index=len(self.children),
            role=role,
            tags=tags,            
            parent=self,            
            style=style,
            attrs=attrs,
        )      
        self.append(block)
        return block
    
    def _process_tuple_content(self, other: tuple[ContentType, ...]):
        block = BlockSent(default_sep=" ")
        for o in other:
            block.append(o)
        return block
        
    
    
    def __iadd__(self, other: ContentType | BlockSent | tuple[ContentType, ...]):
        if isinstance(other, tuple):
            for o in other:
                self.root.append(o)
        else:
            self.root.append(other)
        return self
    
    def __iand__(self, other: ContentType | BlockSent | BlockChunk | tuple[ContentType, ...]):
        if isinstance(other, tuple):
            for o in other:
                self.root.append(o, sep="")
        else:
            self.root.append(other, sep="")
        return self
            
    
    
    def __itruediv__(self, other: "ContentType | BlockChunk | Block | BlockSent | tuple[ContentType, ...]"):        
        if isinstance(other, tuple):
            other =self._process_tuple_content(other)        
        self.append(other)
        return self
    
        
    def __repr__(self) -> str:
        root = self.root.render() if self.root else ''
        tags = ','.join(self.tags) if self.tags else ''
        tags = f"[{tags}] " if tags else ''
        return f"{self.__class__.__name__}({tags}root={root}, children={self.children})"


def FieldAttr(
    type: Type,
    description: str,
    name: str | None = None,
    gt: annotated_types.Gt | None = None,
    lt: annotated_types.Lt | None = None,
    ge: annotated_types.Ge | None = None,
    le: annotated_types.Le | None = None,
):
    return FieldAttrBlock(
        name=name,
        type=type,
        description=description,
        gt=gt,
        lt=lt,
        ge=ge,
        le=le,
    )


class FieldAttrBlock:
    name: str
    type: Type = str
    description: str
    gt: annotated_types.Gt | None = None
    lt: annotated_types.Lt | None = None
    ge: annotated_types.Ge | None = None
    le: annotated_types.Le | None = None
    
    def __init__(
        self, 
        name: str,          
        description: str, 
        type: Type = str,
        gt: annotated_types.Gt | None = None, 
        lt: annotated_types.Lt | None = None, 
        ge: annotated_types.Ge | None = None, 
        le: annotated_types.Le | None = None
    ):
        self.name = name
        self.type = type
        self.description = description
        self.gt = gt
        self.lt = lt
        self.ge = ge
        self.le = le
        
        
    def parse(self, content: str):
        content = content.strip()
        content = textwrap.dedent(content)
        if self.type == int:
            return int(content)
        elif self.type == float:
            return float(content)
        elif self.type == bool:
            return bool(content)
        elif self.type == str:
            return content
        elif self.type == list:
            return content.split(",")
        elif self.type == dict:
            return json.loads(content)
        else:
            raise ValueError(f"Invalid type: {self.type}")


class BlockSchemaField(Block):
    
    __slots__ = [
        "type",
        "name",
    ]
    
    def __init__(
        self, 
        name: str, 
        type: Type,
        attrs: dict[str, FieldAttrBlock] | None = None,        
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        id: str | None = None,
        index: int | None = None,
        parent: "BaseBlock | None" = None,  
        styles: list[str] | None = None,
    ):
        super().__init__(name, tags=tags or [] + [name], style=style, id=id, index=index, parent=parent, attrs=attrs, styles=styles)
        if not type:
            raise ValueError("type is required")
        self.type = type
        self.name = name
    
    
    # @property
    # def schema_path(self) -> list[int]:
    #     parent = self.parent
    #     for _ in range(10):
    #         if parent is None:
    #             return [self.index]
    #         if isinstance(parent, BlockSchemaField):
    #             return parent.schema_path + [self.index]
    #         elif isinstance(self.parent, BlockSchema):
    #             return [self.index]
    #         else:
    #             parent = parent.parent
    #     else:
    #         raise ValueError("Invalid parent")
            
       
    def partial_init(
        self, 
        root: list[BlockChunk] | None = None, 
        children: list[BlockChunk] | None = None, 
        attrs: dict[str, str] | None = None, 
        tags: list[str] | None = None
    ):
        res = ResponseBlock(
            self,
            tags=self.tags + tags or [],            
        )
        if root: 
            for chunk in root:
                res.root.append(chunk)
        if children:
            for chunk in children:
                res.append(chunk)
        if attrs:
            res.set_attributes(attrs)
        return res
    
    def shallow_copy(self) -> "BlockSchemaField":
        return BlockSchemaField(
            name=self.name,
            type=self.type,
            attrs=self.attrs,
            role=self.role,
            styles=self.styles,
            tags=[t for t in self.tags],
            index=self.index,
            id=self.id,
        )
    


class FieldContext(Block):
    pass



class ResponseBlock(Block):
    
    __slots__ = (
        "schema", 
        "_value", 
        "postfix"
    )
    
    def __init__(
        self, 
        schema: BlockSchemaField, 
        children: BlockList | None = None, 
        tags: list[str] | None = None, 
        style: str | None = None,
        id: str | None = None,
        index: int | None = None,
        parent: "BaseBlock | None" = None,
    ):
        if schema.name not in tags:
            tags = [schema.name] + (tags or [])
        super().__init__(children=children, role="assistant", tags=tags, style=style, id=id, index=index, parent=parent)
        self.root.default_sep = ""
        self.default_sep = ""
        self.schema = schema
        self._value = None
        self.postfix: BlockSent | None = None
        
    
    @property
    def name(self) -> str:
        return self.schema.name
    
    def _cast(self, content: str):
        if self.schema.type == str:
            return content
        elif self.schema.type == int:
            return int(content)
        elif self.schema.type == float:
            return float(content)
        elif self.schema.type == bool:
            return bool(content)
        elif self.schema.type == list:
            return content.split(",")
        elif self.schema.type == dict:
            return json.loads(content)
        
    def __getitem__(self, key: str):
        return self.attrs[key]
    
    def __setitem__(self, key: str, value: Any):
        self.attrs[key] = value
    
    def set_attributes(self, attrs: dict[str, str]):
        for k, v in attrs.items():
            if k in self.schema.attrs:
                value = self.schema.attrs[k].parse(v)
                self.attrs[k] = value
            else:
                raise ValueError(f"Attribute {k} not found in schema")
    
    def set_postfix(self, postfix: BlockSent): 
        postfix.parent = self
        self.postfix = postfix
        self.postfix.set_index(-1)
        
    
    def commit(self):
        from .renderers_base3 import SequenceRenderer
        # content = self.children.render()
        content = SequenceRenderer().render(self)
        content = content.strip()
        content = textwrap.dedent(content)
        self._value = self._cast(content)
        return self
    
    
    def model_dump(self):
        dump = super().model_dump()
        dump["_type"] = "Block"
        # dump["_type"] = "ResponseBlock"
        # dump["schema"] = self.schema.model_dump()
        del dump["schema"]
        # dump["value"] = self._value
        dump["postfix"] = self.postfix.model_dump() if self.postfix else None
        return dump
    
    def copy_without_children(self):
        c = copy.deepcopy(self)
        c.children = []
        return c
    
    
    @property
    def value(self):
        return self._value
    
    
    
    
    def __repr__(self) -> str:
        root = self.root.render() if self.root else ''
        tags = ','.join(self.tags) if self.tags else ''
        tags = f"[{tags}] " if tags else ''
        return f"ResponseBlock({tags}root={root}, children={self.children})"
    
    
    
# class BlockSchemaMeta(type):
#     def __new__(mcls, name, bases, dct):
#         return super().__new__(mcls, name, bases, dct)
    
#     def __call__(cls, *args, **kwargs):
#         return super().__call__(*args, **kwargs)
    
    
#     def __enter__(cls):
#         return cls
    
#     def __exit__(cls, exc_type, exc_value, traceback):
#         pass
    

        
SchemaTarget = str
SchemaFormat = Literal["xml"]

class BlockSchema(Block):
    
    __slots__ = [
        "inst",
        "pure_schema",
    ]
    
    def __init__(
        self, 
        root: ContentType | BlockChunk | BlockList | None = None, 
        children: BlockList | None = None,                 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        attrs: "dict[str, FieldAttrBlock] | None" = None,
        id: str | None = None,
        styles: list[str] | None = None,
        index: int | None = None,
        parent: "BaseBlock | None" = None,
    ):
        super().__init__(root=root, children=children, role=role, tags=tags, style=style, attrs=attrs, id=id, styles=styles, index=index, parent=parent)
        self.inst: ResponseBlock | None = None
        self.pure_schema: BlockSchemaField | None = None
        
        
    def build_response(self) -> "ResponseBlock":
        # def _clone_target_node(n: Block) -> ResponseBlock:
        #     return ResponseBlock(                
        #         schema=n, 
        #         tags=n.tags,                           
        #     )
        def _clone_target_node(n: BlockSchemaField) -> BlockSchemaField:            
            r = n.shallow_copy()
            # parent = n.parent
            # while parent is not None:
            #     if isinstance(parent, BlockSchemaField):
                    
            return r
        
        def _is_target(node: BaseBlock) -> bool:
            return isinstance(node, BlockSchemaField)
        res = self.gather_trees(_is_target, _clone_target_node)
        if len(res) == 1:
            self.pure_schema = res[0]
            return self.pure_schema
        else:
            raise ValueError("Multiple target nodes found")      
        
    
    def new(self, target: SchemaTarget, format: SchemaFormat = "xml"): 
        return 
    
    
    def _get_field(self, field_name: str) -> ResponseBlock:
        if self.inst is None:
            raise ValueError("Schema is not initialized")
        field = self.inst.get(field_name)
        if field is None:
            raise ValueError(f"Field '{field_name}' not found in response schema")
        if not isinstance(field, ResponseBlock):
            raise ValueError(f"Field '{field_name}' is not a response block")
        return field
    
    def _get_field_schema(self, field_name: str) -> BlockSchemaField:
        if self.pure_schema is None:
            raise ValueError("Schema is not initialized")
        field = self.pure_schema.get(field_name)
        if field is None:
            raise ValueError(f"Field '{field_name}' not found in schema")
        if not isinstance(field, BlockSchemaField):
            raise ValueError(f"Field '{field_name}' is not a schema field")
        return field

    
    # def _inst_field(self, field_schema: BlockSchemaField) -> ResponseBlock:
    #     res = ResponseBlock(
    #         schema=field_schema,
    #         tags=field_schema.tags,
    #     )
    #     if self.inst is None:
    #         self.inst = res
    #     else:
    #         path = field_schema.path[1:-1]
    #         if path:
    #             self.inst.insert(path, res)
    #         else:
    #             self.inst.append(res)
    #     return res
    def _inst_field(self, field_schema: BlockSchemaField) -> ResponseBlock:
        # schema_path = [field_schema]
        # while (parent := schema_path[0].parent) is not None:
        #     schema_path.insert(0, parent)
        
        def build_response_block(schema: BlockSchemaField) -> ResponseBlock:
            res = ResponseBlock(
                schema=schema,
                tags=schema.tags,
            )
            return res
        
        def instatiate_field(curr: ResponseBlock, schema: BlockSchemaField):
            res = ResponseBlock(
                schema=schema,
                tags=schema.tags,
            )
            curr.append(res)
            return res
            
            
        if self.pure_schema is None:
            raise ValueError("Schema is not initialized")
        
        path_fields = [field_schema]
        while (parent := path_fields[0].parent) is not None:
            path_fields.insert(0, parent)
        
        curr = self.inst
        for fld in path_fields:
            if curr is None:
                curr = build_response_block(fld)
                
            
                
                
        
        # curr = self.inst
        # if curr is None:
        #     self.inst = build_response_block(self.pure_schema)
        #     curr = self.inst
        # else:
            
            
        return res
        
        

            
            
            
    
    def partial_init_field(
        self,
        field_name: str, 
        root: list[BlockChunk] | None = None, 
        children: list[BlockChunk] | None = None, 
        attrs: dict[str, str] | None = None, 
        tags: list[str] | None = None
    ):        
        field_schema = self._get_field_schema(field_name)
        field = self._inst_field(field_schema)
        if root:
            for block in root:
                field.root.append(block)
        if attrs:
            field.set_attributes(attrs)
        if children:
            for block in children:
                field.append(block)
        if tags:
            field.tags += tags
        c = field.copy_without_children()
        return c
    
    
    
    def partial_append_field(
        self,
        field_name: str,
        content: ContentType | BlockChunk,
        tags: list[str] | None = None,
    ):
        field = self._get_field(field_name)
        value = field.inline_append(content, sep="")        
        return value
    
    # def set_postfix(
    #     self, 
    #     field_name: str, 
    #     postfix: list[BlockChunk],
    #     tags: list[str] | None = None,
    #     commit: bool = False
    # ):
    #     field = self._get_field(field_name)
    #     end_sent = BlockSent(tags=tags or [], sep="")
        
    #     if commit:
    #         field.commit()
    #     return end_sent
        
    def commit_field(
        self,
        field_name: str,
        postfix: list[BlockChunk] | None = None,
        tags: list[str] | None = None,
    ):
        field = self._get_field(field_name)
        if postfix:
            end_sent = BlockSent(default_sep="")
            # end_sent = Block(tags=tags or [])
            for chunk in postfix:
                end_sent.append(chunk)                
            field.set_postfix(end_sent)
        field.commit()
        return field
    
    
    def copy(self):
        return copy.deepcopy(self)
        

        
        
    


    
    



class Blockable(Protocol):
    def block(self) -> BlockChunk:
        ...









def block(
    *content: ContentType | BlockChunk | list,
    role: str | None = None,
    tags: list[str] | None = None,
    style: str | None = None,
    attrs: dict | None = None,
):
    return BlockChunk(
        *content,
        role=role,
        tags=tags,
        style=style,
        attrs=attrs,
    )