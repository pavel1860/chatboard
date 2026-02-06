from collections import defaultdict
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import inspect
import json
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Concatenate, ContextManager, Generator, Generic, Iterator, List, Literal, ParamSpec, Protocol, SupportsIndex, Type, TypeVar, Union, overload
from pydantic_core import core_schema
from pydantic import BaseModel, GetCoreSchemaHandler
from .style import InlineStyle, BlockStyle, style_manager
from .util import BlockRole, LlmUsage, ToolCall
from ..utils.model_utils import schema_to_ts
if TYPE_CHECKING:
    from ..model import BlockModel
    from ..model.model import SelectQuerySet



ContentType = Union[str , dict , "Block", "ContextManager[Block]"]



    
class ContextStack:
    """
    A stack-based context for managing nested block structures.
    """
    _ctx_stack: "list[Block]"
    
    def __init__(self):
        self._ctx_stack = []
        
    def __getitem__(self, idx: int) -> "Block":
        return self._ctx_stack[idx]
    
    def __len__(self) -> int:
        return len(self._ctx_stack)
    
    def root(self) -> "Block":
        if not self._ctx_stack:
            raise ValueError("No context stack")
        return self._ctx_stack[0]
    
    def push(self, block: "Block"):
        self._ctx_stack.append(block)
        
    def pop(self):
        return self._ctx_stack.pop()
    
    def top(self):
        return self._ctx_stack[-1]




MAP_RET = TypeVar("MAP_RET")

class BlockList(list["Block"], Generic[MAP_RET]):
    """
    A list of blocks
    """
    


    
    def group(self, role: BlockRole | None = None, tags: list[str] | None = None, extra: "Block | None" = None) -> "Block":
        """
        Group the blocks by role and tags
        """
        block = Block(items=self, role=role, tags=tags)
        if extra:
            block.append(extra)
        return block
    
    def group_to_list(self, role: BlockRole | None = None, tags: list[str] | None = None, extra: "Block | None" = None) -> "BlockList":
        """
        Group the blocks by role and tags
        """
        
        
        block = Block(items=self, role=role, tags=tags)
        if extra:
            block.append(extra)
        if block.items:
            return BlockList([block])
        else:
            return BlockList([])
    
    def group_or_none(self, role: BlockRole | None = None, tags: list[str] | None = None) -> "Block | None":
        """
        Group the blocks by role and tags
        """
        if not self:
            return None
        return self.group(role, tags)
    
    def slice(self, start: int, end: int) -> "BlockList":
        """
        Slice the blocks
        """
        return BlockList(self[start:end])
    
    def find(self, tag: str | list[str], default: Any = None) -> "BlockList":
        """
        Get the blocks by key
        """
        if isinstance(tag, str):
            tag = [tag]
        find_results = []
        for item in self: 
            if any(t in item.tags or t == item.role for t in tag):
                find_results.append(item)
            else:
                find_results.extend(item.find(tag, default))
        return BlockList(find_results)
        # return BlockList([**item.find(tag) for item])
        # return BlockList([item for item in self if all(tag in item.tags for tag in tag)])
    
    def find_before(self, tag: str) -> "BlockList":
        """
        Get the blocks before the pivot tag
        """
        before_list = BlockList()
        for item in self:
            if tag in item.tags:
                break
            before_list.append(item)
        return before_list
    
    def find_after(self, tag: str) -> "BlockList":
        """
        Get the blocks after the pivot tag
        """
        before, pivot, after = self.split(tag)
        return after
    
    def filter(self, tag: str | list[str]) -> "BlockList":
        """
        Get the blocks by key
        """
        if isinstance(tag, str):
            tag = [tag]
        return BlockList([item for item in self if not any(tag in item.tags + [item.role] for tag in tag)])
    
    def split(self, pivot_tag: str) -> tuple["BlockList", "Block", "BlockList"]:
        """
        Split the blocks by pivot tag
        """
        pre_blocks = BlockList()
        pivot_block = None
        post_blocks = BlockList()
        current = pre_blocks
        for item in self:
            if pivot_block is None and pivot_tag in item.tags:
                pivot_block = item
                current = post_blocks
                continue
            else:
                current.append(item)
        if pivot_block is None:
            raise ValueError(f"Pivot tag {pivot_tag} not found")
        return pre_blocks, pivot_block, post_blocks
                
    
    def map(self, func: "Callable[[Block], MAP_RET]") -> "list[MAP_RET]":
        """
        Map the blocks by function
        """
        return [func(item) for item in self]
    
    def bmap(self, func: "Callable[[Block], Block]") -> "BlockList":
        """
        Map the blocks by function
        """
        return BlockList([func(item) for item in self])
    
    
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
                if not isinstance(item, Block):
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
    
    
        
class Block:
    """
    A block is a container for content and other blocks.
    """
    
    __slots__ = [
        "_content", 
        "tags", 
        "items", 
        "inline_style", 
        "computed_style", 
        "parent",
        "depth",
        "_ctx",
        "run_id",
        "role",
        "name",
        "model",
        "tool_calls",
        "usage",
        "id",
        "db_id",
        "attrs",
        "_children_block",
    ]
    # tags: list[str]
    # items: list["Block"]
    # inline_style: BlockStyle
    # computed_style: InlineStyle
    # content: Any | None
    # parent: "Block | None"
    # depth: int
    
    def __init__(
        self, 
        *content: str, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None, 
        attrs: dict | None = None,
        depth: int = 0, 
        parent: "Block | None" = None, 
        dedent: bool = True, 
        items: list["Block"] | None = None,
        ctx: ContextStack | None = None,
        run_id: str | None = None,
        role: BlockRole | None = None,
        name: str | None = None,
        model: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        usage: LlmUsage | None = None,
        id: str | None = None,
        db_id: str | None = None,
    ):
        if dedent and content:
            if isinstance(content, str):
                content = textwrap.dedent(content).strip()
            elif isinstance(content, tuple):                
                content = tuple(textwrap.dedent(item).strip() if isinstance(item, str) and item.startswith("\n") else item for item in content)
        self._content: tuple[str, ...] = content if isinstance(content, tuple) else (content,)
        self.tags = tags or []
        self.items = items or []
        self.depth = depth or 0
        self.inline_style = BlockStyle(style)
        self.parent = parent
        self._ctx = ctx
        self.run_id = run_id
        if role == "tool" and not id:
            raise ValueError("Tool blocks must have an id")
        self.role = role
        self.name = name
        self.model = model
        self.tool_calls = [ToolCall(**tool) if isinstance(tool, dict) else tool for tool in tool_calls or []]
        self.usage = usage
        self.id = id
        self.db_id = db_id
        self.attrs = attrs
        self._children_block = None
    # def __call__(
    #     self, 
    #     content: Any | None = None, 
    #     tags: list[str] | None = None, 
    #     style: InlineStyle | None = None, 
    #     depth: int = 0, 
    #     parent: "BaseBlock | None" = None, 
    #     dedent: bool = True, 
    #     items: list["BaseBlock"] | None = None
    # ):
    #     pass
    
    @property
    def content(self):
        return " ".join(str(item) for item in self._content if item is not None)
    
    def model_dump(self):
        return {
            "_type": self.__class__.__name__,
            "content": self.content,
            "tags": self.tags,
            "style": self.inline_style.style,
            "attrs": self.attrs,
            "items": [item.model_dump() for item in self.items],
            "role": self.role,
            "name": self.name,
            "model": self.model,
            "tool_calls": [tool.model_dump() for tool in self.tool_calls],
            "usage": self.usage.model_dump() if self.usage else None,
            "id": self.id,
            "db_id": self.db_id,
            "run_id": self.run_id,
            "depth": self.depth,            
        }
        
    def copy(self) -> "Block":
        return Block(**self.model_dump())
    
    @classmethod
    def model_validate(cls, data: dict):
        if "_type" not in data:
            raise ValueError("Missing _type, not a valid block")
        if data["_type"] != cls.__name__:
            raise ValueError(f"Invalid _type: {data['_type']}")
        _type = data.pop("_type")
        items = data.pop("items")
        content = data.pop("content")
        content = content if isinstance(content, tuple) else (content,)
        return cls(*content,**data, items=[cls.model_validate(item) for item in items])
    
    @property
    def ctx_items(self) -> list["Block"]:
        if self._ctx:
            return self._ctx[-1].items
        else:
            return self.items
    
    def __enter__(self):
        if self._ctx is None:
            self._ctx = ContextStack()
        # if self._children_block is not None:
        #     self._ctx.push(self._children_block)
        # else:
        self._ctx.push(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._ctx is None:
            raise ValueError("No context stack")        
        if len(self._ctx) > 1:
            self._ctx.pop()
        
        
    def children(self):
        if not self.ctx_items:
            raise ValueError("No context stack. you should set parent block first")
        self._children_block = self.ctx_items[-1]
        return self
        
    def find(self, tag: str | list[str], default: Any = None) -> "BlockList":
        if isinstance(tag, str):
            tag = [tag]
        sel_items = BlockList()
        for item in self.items:
            for k in tag:
                if k in item.tags or k == item.role:
                    sel_items.append(item)
                    break
            else:
                sel_items.extend(item.find(tag, default))
        return sel_items
    
        
    
    def filter(self, tag: str | list[str]) -> "BlockList":
        """
        Get the blocks by key
        """
        if isinstance(tag, str):
            tag = [tag]
        return BlockList([item for item in self if not any(tag in item.tags + [item.role] for tag in tag)])
    
    def _iter(self, max_depth: int, depth: int = 1) -> Iterator["Block"]:
        if depth > max_depth:
            return
        for item in self.items:
            yield item
            yield from item._iter(max_depth, depth + 1)
            
    def iter(self, max_depth: int = 10000) -> Iterator["Block"]:
        return self._iter(max_depth)
    
    
    def split(self, pivot_tag: str) -> tuple["BlockList", "Block", "BlockList"]:
        pre_blocks = BlockList()
        post_blocks = BlockList()
        pivot_block = None
        current = pre_blocks
        for item in self.iter(1):
            if pivot_tag in item.tags or pivot_tag == item.role:
                pivot_block = item
                current = post_blocks
                continue
            else:
                current.append(item)
        if pivot_block is None:
            raise ValueError(f"Pivot tag {pivot_tag} not found")
        return pre_blocks, pivot_block, post_blocks
        
    
    def first(self, key: str | list[str], default: Any = None, raise_error: bool = True) -> "Block":
        blocks = self.find(key, default)
        if len(blocks) == 0:
            if raise_error:
                raise ValueError(f"No block found for key {key}")
            else:
                return default
        return blocks[0]
        
        
    def __getitem__(self, idx: int) -> "Block":
        return self.items[idx]
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __str__(self) -> str:
        return self.render()
    
    
    def _validate_content_primitive(self, content: Any):
        if not content:
            return True
        if not isinstance(content, str):
            raise ValueError(f"Invalid content: {content}")
        return True
        
    def _validate_content_list(self, content: list[Any] | tuple[Any, ...]):
        for item in content:
            self._validate_content(item)
        return True
        
    def _validate_content(self, content: Any):
        if isinstance(content, list) or isinstance(content, tuple):
            self._validate_content_list(content)
        else:
            self._validate_content_primitive(content)
        return True
    
    def _build_instance(
        self, 
        *content: Any, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None,
        attrs: dict | None = None,
        items: list["Block"] | None = None,
        role: BlockRole | None = None,
    ):
        if isinstance(content, Block):
            inst = content
        else:
            # self._validate_content(content)
            inst = Block(
                    *content, 
                    tags=tags, 
                    style=style, 
                    parent=self._ctx[-1] if self._ctx else self,
                    depth=len(self._ctx) if self._ctx else 0,
                    items=items,
                    ctx=self._ctx,
                    attrs=attrs,
                )
        if role is not None:
            inst.role = role
        if tags is not None:
            inst.tags = tags
        # if style is not None:
            # inst.inline_style.style = style
        # if attrs is not None:
            # inst.attrs = attrs
        return inst
        
    def __call__(
        self, 
        *content: Any, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None, 
        attrs: dict | None = None,
        **kwargs
    ):
        if not content:
            self.append(None, tags=tags, style=style, attrs=attrs, **kwargs)
        else:
            for item in content:
                if isinstance(item, list):
                    for it in item:
                        self.append(it, tags=tags, style=style, attrs=attrs, **kwargs)
                else:
                    self.append(item, tags=tags, style=style, attrs=attrs, **kwargs)
        return self.ctx_items[-1]
    
    
    def append(
        self, 
        *content: Any, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None,
        attrs: dict | None = None,
        items: list["Block"] | None = None,
        role: BlockRole | None = None,
    ):
        
        
        inst = self._build_instance(
            *content, 
            tags=tags, 
            style=style, 
            items=items,
            attrs=attrs,
            role=role,
        )
        self.ctx_items.append(inst)
        # if self._ctx:
        #     self._ctx[-1].items.append(inst)
        # else:
        #     self.items.append(inst)
        return self
    
    def extend(self, *content: Any):
        if self.ctx_items:
            self.ctx_items[-1]._content += content
        else:
            self.append(*content)
    
    
    def merge(self, other: "Block"):
        """
        Merge another block into this one
        """
        self.ctx_items.extend(other.items)
        return self
    
    
    def __itruediv__(self, content: ContentType | tuple[ContentType, ...]):
        """
        Append a new item to the block
        """
        if isinstance(content, list):
            raise ValueError("Cannot use list as single line content")
        c = content if isinstance(content, tuple) else (content,)
        self.append(*c)
        return self    
    
    def __truediv__(self, content: ContentType):
        """
        Append a new item to the block
        """
        if isinstance(content, list):
            raise ValueError("Cannot use list as single line content")
        self.append(content)
        return self
    
    def __add__(self, other: "Block | Any"):
        """
        Append a new item to the block
        """
        # TODO: figure out how to merge block content
        if isinstance(other, Block):
            self.merge(other)
        else:
            o = other if isinstance(other, tuple) else (other,)
            self.extend(*o)
        return self
        
    # def append(self, item: "Block | Any"):
    #     self.items.append(item)
    #     return item
    
    @property
    def is_block(self) -> bool:
        return len(self.items) > 0
    
    @property
    def is_wrapper(self) -> bool:
        return self.content is None and len(self.items) > 0
    
    @property
    def is_inline(self) -> bool:
        return len(self.items) == 0
    
    def add_style(self, **style_props: Any) -> "Block":
        """
        Add inline style properties to this block
        """
        self.inline_style.update(style_props)
        return self
    
    def model_schema(self, model: Type[BaseModel], format: str = "ts"):    
        if format == "ts":
            content = schema_to_ts(model)
        else:
            content = model.model_json_schema()        
        return self.append(content)
    
    def find_tool(self, tool_name: str) -> ToolCall | None:
        for tool in self.tool_calls:
            if tool.name == tool_name:
                return tool
        return None

    
    def get_style(self, property_name: str, default: Any = None) -> Any:
        """
        Get a computed style property value
        """
        return self.inline_style.get(property_name, default)
    
    

    
    
    def render(self) -> str:
        from .block_renderer import BlockRenderer
        from .renderer import RendererMeta
        if self.items != self.ctx_items and not self._children_block:
            raise ValueError("Wrong block context was passed to render. probably you are using the same ctx name for child and parent blocks")
        rndr = BlockRenderer(style_manager, RendererMeta._renderers)
        return rndr.render(self)
    
    def print(self):
        print(self.render())
    
    
    def __repr__(self) -> str:                
        content = self.render()
        if len(content) > 30:
            content = content[0:30] + "..."
        prompt = f"""<{self.__class__.__name__} tags={self.tags} role={self.role} style={self.inline_style.style} depth={self.depth} content="{content}">"""
        return prompt      
      
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
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")
        
        
    def to_model(self) -> "BlockModel":
        from ..model import BlockModel
        return BlockModel.from_block(self)
    
    @classmethod
    def query(cls) -> "SelectQuerySet[BlockModel]":
        from ..model import BlockModel
        return BlockModel.query(parse=lambda x: x.to_block())


class BlockContext:
    """
    A context for managing nested block structures.
    """
    ctx: ContextStack
    
    def __init__(self, root: "Block"):
        self.ctx = ContextStack()
        self.ctx.push(root)
        
    def __call__(self, *content: Any, tags: list[str] | None = None, style: InlineStyle | None = None, attrs: dict | None = None, **kwargs):       
        self._append(*content, tags, style, attrs)
        return self
    
    @property
    def root(self):
        return self.ctx.root()
    
    
    @property
    def last(self):
        """
        The last block in the context stack
        """
        if not self.ctx:
            raise ValueError("No context stack")
        if not self.ctx[-1].items:
            return self.ctx[-1]
        return self.ctx[-1].items[-1]
    
    
    def _append(
        self, 
        *content: Any, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None,
        attrs: dict | None = None,
        items: list["Block"] | None = None
    ):
        inst = Block(
            *content, 
            tags=tags, 
            style=style, 
            parent=self.ctx[-1],
            depth=len(self.ctx) if self.ctx else 0,
            items=items,
            attrs=attrs,
        )        
        self.ctx[-1].append(inst)
        return inst
    
    
    def __itruediv__(self, *content: Any):
        self._append(*content)
        return self
    
    
    
    
    
    
    
    
class Blockable(Protocol):
    def block(self) -> Block:
        ...



P = ParamSpec("P")
R = TypeVar("R")


class FunctionBlock(Generic[P, R]):
        
    def __init__(self, func, blk: Block):
        self.func = func
        self.blk = blk
        wraps(func)(self)
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Block:
        print("[Calling]")
        return self.func(*args, **kwargs)

    def __enter__(self):
        print("[Entering context]")
        # return self.func  # or `self` if you want to preserve decorator logic
        self.blk.__enter__()
        return self.blk

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[Exiting context]")
        self.blk.__exit__(exc_type, exc_val, exc_tb)




# def block(
#     content: str | None = None,
#     *,
#     tags: list[str] | None = None,
#     style: "InlineStyle | None" = None,
#     attrs: dict | None = None,
#     depth: int = 0,
#     dedent: bool = True,
#     role: "BlockRole | None" = None,
#     name: str | None = None,
#     model: str | None = None,
#     tool_calls: list["ToolCall"] | None = None,
#     usage: "LlmUsage | None" = None,
# ) -> Callable[[Callable[Concatenate[Block, P], R]], Callable[P, R]]:
#     def decorator(func: Callable[Concatenate[Block, P], R]) -> Callable[P, R]:
#         if inspect.isgeneratorfunction(func):
#             @wraps(func)
#             @contextmanager
#             def generator_wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Block, None, None]:
#             # def generator_wrapper(*args: P.args, **kwargs: P.kwargs) -> ContextManager[Block]:
#                 blk = Block(
#                     content=content,
#                     tags=tags,
#                     style=style,
#                     attrs=attrs,
#                     depth=depth,
#                     dedent=dedent,
#                     role=role,
#                     name=name,
#                     model=model,
#                     tool_calls=tool_calls,
#                     usage=usage,
#                 )
#                 with blk:
#                     gen = func(blk, *args, **kwargs)
#                     yield next(gen)
#             return generator_wrapper  # type: ignore
#         else:
#             @wraps(func)
#             def normal_wrapper(*args: P.args, **kwargs: P.kwargs) -> Block:
#                 blk = Block(
#                     content=content,
#                     tags=tags,
#                     style=style,
#                     attrs=attrs,
#                     depth=depth,
#                     dedent=dedent,
#                     role=role,
#                     name=name,
#                     model=model,
#                     tool_calls=tool_calls,
#                     usage=usage,
#                 )
#                 res = func(blk, *args, **kwargs)
#                 return res if res is not None else blk  # type: ignore
#             return normal_wrapper
#     return decorator




from typing import Callable, TypeVar, ParamSpec, Concatenate, Generator, ContextManager, Union
from contextlib import contextmanager
from functools import wraps
import inspect

P = ParamSpec("P")

def block(
    *content: str,
    tags: list[str] | None = None,
    style: "InlineStyle | None" = None,
    attrs: dict | None = None,
    depth: int = 0,
    dedent: bool = True,
    role: "BlockRole | None" = None,
    name: str | None = None,
    model: str | None = None,
    tool_calls: list["ToolCall"] | None = None,
    usage: "LlmUsage | None" = None,
    ) -> Callable[[Callable[Concatenate[Block, P], Any]], Callable[P, Union[Block, ContextManager[Block]]]]:
    def decorator(func: Callable[Concatenate[Block, P], Any]) -> Callable[P, Union[Block, ContextManager[Block]]]:
        if inspect.isgeneratorfunction(func):
            @wraps(func)
            @contextmanager
            def generator_wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Block, None, None]:
                blk = Block(
                    *content,
                    tags=tags,
                    style=style,
                    attrs=attrs,
                    depth=depth,
                    dedent=dedent,
                    role=role,
                    name=name,
                    model=model,
                    tool_calls=tool_calls,
                    usage=usage,
                )  # Build your Block
                with blk:
                    gen = func(blk, *args, **kwargs)
                    yield next(gen)
            return generator_wrapper
        else:
            @wraps(func)
            def normal_wrapper(*args: P.args, **kwargs: P.kwargs) -> Block:
                blk = Block(
                    *content,
                    tags=tags,
                    style=style,
                    attrs=attrs,
                    depth=depth,
                    dedent=dedent,
                    role=role,
                    name=name,
                    model=model,
                    tool_calls=tool_calls,
                    usage=usage,
                )
                result = func(blk, *args, **kwargs)
                return result if isinstance(result, Block) else blk
            return normal_wrapper
    return decorator






