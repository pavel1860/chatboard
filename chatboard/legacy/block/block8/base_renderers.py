import textwrap
from typing import TYPE_CHECKING, Any, List
from chatboard.block8.block8 import Block
from chatboard.block8.style_manager import StyleManager





class RenderContext:
    def __init__(
        self, 
        block: "Block | None",
        style: dict | None, 
        parent_ctx: "RenderContext | None" = None,
        verbose: bool = False,
    ):
        self.block = block
        self.style = style
        self.parent_ctx = parent_ctx
        if parent_ctx and parent_ctx.verbose:
            self.verbose = True
        else:
            self.verbose = verbose
        
    @property
    def is_list(self) -> bool:
        from chatboard.block.block7 import BlockList
        return isinstance(self.block, BlockList)
    
    def get_style(self, key: str) -> dict:
        if self.style is None:
            return None
        return self.style.get(key)
    
    def get_parent_style(self, key: str) -> dict:
        if self.parent_ctx is None:
            return None
        return self.parent_ctx.get_style(key)
    
    @property
    def is_context(self) -> bool:
        from chatboard.block.block7 import Block
        return isinstance(self.block, Block)
    
    @property
    def is_root(self) -> bool:
        return self.parent_ctx is None
    
    @property
    def is_wrapper(self) -> bool:
        from chatboard.block.block7 import Block
        if self.block and isinstance(self.block, Block):
            if not self.block.root:
                return True
        return False
    
    def log(self, name: str, target: Any):
        if self.verbose:
            print(f"{name}{self.block.path}:", target)
            
            
            






class RendererRegistry:
    
    
    def __init__(self):
        self._registry = {}
        self._default_registry = {}
    
    def register(self, name: str, renderer: "BaseRenderer"):
        self._registry[name] = renderer
        
    def register_default(self, name: str, renderer: "BaseRenderer"):
        self._default_registry[name] = renderer

    def get(self, name: str | None = None, default_name: str | None = None) -> "BaseRenderer":
        if name:
            renderer =  self._registry[name]
        else:
            renderer =  self._default_registry[default_name]
        return renderer()

    def list_renderers(self):
        return list(self._registry.keys());




style_manager = StyleManager()       
renderer_registry = RendererRegistry()


class MetaRenderer(type):
    
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)
        if name == "SentenceRenderer":
            renderer_registry.register_default("sentence-format", cls_obj)
        elif name == "ListRenderer":
            renderer_registry.register_default("list-format", cls_obj)
        elif name == "BlockRenderer":
            renderer_registry.register_default("block-format", cls_obj)
        elif bases:
            if "styles" not in dct or not dct["styles"]:
                raise ValueError(f"Renderer {name} must define styles")                
            if bases[0] == SentenceRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"sentence-format": name})
            elif bases[0] == ListRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"list-format": name})
            elif bases[0] == BlockRenderer:
                renderer_registry.register(name, cls_obj)
                style_manager.add_style(dct["styles"], {"block-format": name})
            else:
                raise ValueError(f"Renderer {name} must inherit from SentenceRenderer, ListRenderer, or BlockRenderer")            
        
        return cls_obj
    

            
            
class BaseRenderer(metaclass=MetaRenderer):
    styles = []
    
    def render(self, ctx: RenderContext, block: "Block", inner_content: str | None = None) -> str:
        """
        Render the current block itself.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement render method")    

     





class SentenceRenderer(BaseRenderer):
    styles = ["sentence"]

    def render(self, ctx: RenderContext, block: Block, inner_content: str | None = None) -> str:
        return inner_content or ""




class ListRenderer(BaseRenderer):
    styles = ["list"]

    def render(self, ctx: RenderContext, block: Block, inner_content: list[str]) -> str:
        return "\n".join(inner_content)



class BlockRenderer(BaseRenderer):    
    
    def render(self, ctx: RenderContext, block: Block, title_content: str, inner_content: str, postfix_content: str | None = None) -> str:
        content =  f"{title_content}\n{textwrap.indent(inner_content, '  ')}"
        if postfix_content:
            content += f"\n{postfix_content}"
        return content



class XmlRenderer(BlockRenderer):
    styles = ["xml"]

    def render(self, ctx: RenderContext, block: Block, inner_content: str) -> str:
        attrs = " ".join(f'{k}="{v}"' for k, v in block.attrs.items()) if block.attrs else ""
        name = block.name or "block"
        if not inner_content:
            return f"<{name}{(' ' + attrs) if attrs else ''} />"
        return f"<{name}{(' ' + attrs) if attrs else ''}>{inner_content}</{name}>"
