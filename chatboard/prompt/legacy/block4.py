from collections import defaultdict
from abc import abstractmethod
import textwrap
from typing import Any, List, Type
from chatboard.block.style import InlineStyle, BlockStyle, style_manager


    
    
    

        
class BaseBlock:
    
    tags: list[str]
    items: list["BaseBlock"]
    inline_style: BlockStyle
    computed_style: InlineStyle
    content: Any | None
    parent: "BaseBlock | None"
    
    def __init__(
        self, 
        content: Any | None = None, 
        tags: list[str] | None = None, 
        style: InlineStyle | None = None, 
        depth: int = 0, 
        parent: "BaseBlock | None" = None, 
        dedent: bool = True, 
        items: list["BaseBlock"] | None = None
    ):
        if dedent and isinstance(content, str):
            content = textwrap.dedent(content).strip()
        self.content = content
        self.tags = tags or []
        self.items = items or []
        self.depth = depth or 0
        self.inline_style = BlockStyle(style)
        self.parent = parent
        
    def get(self, key: str | list[str], default: Any = None) -> Any:
        if isinstance(key, str):
            key = [key]
        sel_items = []
        for item in self.items:
            for k in key:
                if k in item.tags:
                    sel_items.append(item)
                    break
            else:
                sel_items.extend(item.get(key, default))
        return sel_items
        
        
    def append(self, item: "BaseBlock | Any"):
        self.items.append(item)
        return item
    
    @property
    def is_block(self) -> bool:
        return len(self.items) > 0
    
    @property
    def is_wrapper(self) -> bool:
        return self.content is None and len(self.items) > 0
    
    @property
    def is_inline(self) -> bool:
        return len(self.items) == 0
    
    def add_style(self, **style_props: Any) -> "BaseBlock":
        """
        Add inline style properties to this block
        """
        self.inline_style.update(style_props)
        return self
    
    
    # def compute_styles(self) -> StyleDict:
    #     """
    #     Compute the final styles for this block by applying the style manager rules
    #     """
    #     self.computed_style = style_manager.apply_styles(self)
        
    #     # Recursively compute styles for all child blocks
    #     for item in self.items:
    #         if hasattr(item, 'compute_styles'):
    #             item.compute_styles()
                
    #     return self.computed_style
    
    def get_style(self, property_name: str, default: Any = None) -> Any:
        """
        Get a computed style property value
        """
        return self.inline_style.get(property_name, default)
    
    
    def render(self) -> str:
        from chatboard.block.block_renderer import BlockRenderer
        from chatboard.block.renderer import RendererMeta
        rndr = BlockRenderer(style_manager, RendererMeta._renderers)
        return rndr.render(self)
    
      
    def __repr__(self) -> str:
        content = self.render()
        return f"{self.__class__.__name__}()\n{content}"


