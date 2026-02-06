from typing import Any, Type
from chatboard.prompt.legacy.block4 import BaseBlock
from chatboard.block.block_renderer import BlockRenderer
from chatboard.prompt.llm_block import BlockRole, LLMBlock, ToolCall
from chatboard.block.renderer import RendererMeta
from chatboard.block.style import InlineStyle
from chatboard.block.style import style_manager




class ContextStack:
    """
    A stack-based context for managing nested block structures.
    """
    _ctx_stack: list[BaseBlock]
    
    def __init__(self):
        self._ctx_stack = []
        
    def __getitem__(self, idx: int) -> BaseBlock:
        return self._ctx_stack[idx]
    
    def __len__(self) -> int:
        return len(self._ctx_stack)
    
    def root(self) -> BaseBlock:
        if not self._ctx_stack:
            raise ValueError("No context stack")
        return self._ctx_stack[0]
    
    def push(self, block: BaseBlock):
        self._ctx_stack.append(block)
        
    def pop(self):
        return self._ctx_stack.pop()
    
    def top(self):
        return self._ctx_stack[-1]



class Block(object):
    """
    the main class for building blocks.
    Block is a builder class for creating and managing hierarchical block structures.
    It serves as a high-level interface for constructing blocks of content with associated
    tags, styles and nested relationships. The class maintains a context stack to track
    the current block hierarchy and provides methods for building and manipulating blocks.

    Key features:
    - Creates blocks with content, tags and styles
    - Maintains a context stack for nested block structures 
    - Supports different block types through a registry
    - Provides access to the root block and current context
    - Allows appending child blocks and retrieving blocks by tags
    """
    
    _block_type_registry: dict[Type, Type[BaseBlock]]
    _ctx: ContextStack | None
    
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_block_type_registry"):
            cls._block_type_registry = {}
        return super().__new__(cls)

        
    def __init__(
        self,
        content: Any | None = None,
        tags: list[str] | None = None,
        style: InlineStyle | None = None,
        dedent: bool = True,
        items: list[BaseBlock] | None = None,
        # ctx_stack: "List[Block] | None" = None, 
    ):
        self._ctx = None
        inst = self._build_instance(
            content=content, 
            tags=tags, 
            style=style, 
            dedent=dedent,            
            items=items,
        )
        self._main_inst = inst
    
    @property
    def root(self):
        if self._ctx is None:
            return self._main_inst
        return self._ctx[0]
    
    @property
    def ctx(self):
        if self._ctx is None:
            self._ctx =ContextStack()
        return self._ctx
    
    def get(self, key: str | list[str], default: Any = None) -> Any:
        return self.root.get(key, default)
    
    
    def __call__(self, content: Any, tags: list[str] | None = None, style: InlineStyle | None = None, **kwargs):       
        self._append(content, tags, style)
        return self
    
    def _append(self, value: Any, tags: list[str] | None = None, style: InlineStyle | None = None):
        inst = self._build_instance(value, tags, style)        
        self.ctx[-1].append(inst)
        return inst

    
    @property
    def last(self):
        """
        The last block in the context stack
        """
        if not self._ctx:
            raise ValueError("No context stack")
        if not self._ctx[-1].items:
            return self._ctx[-1]
        return self._ctx[-1].items[-1]
        
    def __enter__(self):        
        if not self._ctx:
            self.ctx.push(self._main_inst)
        else:
            self.ctx.push(self.last)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if len(self.ctx) > 1:
            self.ctx.pop()

    
    @classmethod
    def register(cls, typ: Type, block_type: Type[BaseBlock]):
        if not hasattr(cls, "_block_type_registry"):
            cls._block_type_registry = {}
        cls._block_type_registry[typ] = block_type    
    
        
    def _build_instance(self, content: Any, tags: list[str] | None = None, style: InlineStyle | None = None, parent: "BaseBlock | None" = None, dedent: bool = True, items: list[BaseBlock] | None = None):
        """
        Build an instance of a block based on the content type
        """
        if isinstance(content, BaseBlock):
            return content
        block_type = self.__class__._block_type_registry.get(type(content), BaseBlock)
        inst = block_type(
                content=content, 
                tags=tags, 
                style=style, 
                parent=parent,
                depth=len(self._ctx) if self._ctx else 0,
                items=items,
            )
        return inst
    
    def copy(self, extra_items: list[BaseBlock] | None = None):
        inst = self._build_instance(
            content=self.root.content,
            tags=self.root.tags,
            style=self.root.inline_style.style,
            parent=self.root.parent,
            items=self.root.items + (extra_items or []),
        )
        return Block(inst)
        
    def __add__(self, other: Any):
        # inst = self._append(content)
        if isinstance(other, Block):
            other = other.root
        return self.copy(extra_items=[other])
    
    def __iadd__(self, content: Any):
        self._append(content)
        return self
    
    def __truediv__(self, content: Any):
        self._append(content)
        return self
    
    def __itruediv__(self, content: Any):
        self._append(content)
        return self
    
    @classmethod
    def message(
        cls,
        content: Any | None = None, 
        role: BlockRole = "user", 
        tool_calls: list[ToolCall] | None = None,
        id: str | None = None, 
        name: str | None = None,
        model: str | None = None,
        run_id: str | None = None,
        tags: list[str] | None = None, 
        style: InlineStyle | None = None, 
        depth: int = 0, 
        parent: "BaseBlock | None" = None, 
        dedent: bool = True
    ):
        """
        Create a new LLM block with the given parameters
        """
        return cls(
            LLMBlock(
                content=content,
                role=role,
                tool_calls=tool_calls,
                id=id,
                name=name,
                model=model,
                run_id=run_id,
                tags=tags,
                style=style,
                depth=depth,
                parent=parent,
                dedent=dedent,
            )
        )
        
    
    
    def render(self):        
        rndr = BlockRenderer(style_manager, RendererMeta._renderers)
        return rndr.render(self.root)
    
    
    
    def __repr__(self) -> str:
        content = self.render()
        return f"{self.__class__.__name__}()\n{content}"

    


def print_block(b, depth=0):
    tags = f"{b.tags}" if b.tags else ""
    print(str(depth) + " " + tags + "  " * depth + str(b.content))
    for item in b.items:
        print_block(item, depth+1)
        
        
        
        
        
        
        
        
