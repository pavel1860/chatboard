from .renderer import ItemsRenderer, MarkdownListRenderer, MarkdownParagraphRenderer, MarkdownTitleRenderer, ContentRenderer, Renderer, IndentRenderer
from .style import InlineStyle, BlockStyle, StyleManager, UndefinedTagError
from typing import TYPE_CHECKING, List, Literal, Type, TypedDict, Union
from .block7 import Block



def get_block_data(block, prefix: str = "", error: Exception | None = None):
    block_str = '  ' * block.depth + prefix + "|"
    if block.role:
        block_str += f" {block.role}"
    if block.tags:
        block_str += f" [{', '.join([t for t in block.tags])}]"
    if block.content:
        block_str += f"{block.content}"
    if error:
        block_str += f" err: {error}"
    return block_str
        
def print_block_tree(block, error: Exception | None = None):
    curr = block.parent
    block_stack = [get_block_data(block, "===>", error)]
    while curr is not None:    
        block_stack.append(get_block_data(curr))
        curr = curr.parent
        
    block_str = "\n".join(reversed(block_stack))
    return block_str




class RendererError(Exception):
    def __init__(self, block: Block, renderer: Type[Renderer], error: Exception):
        self.block = block
        self.renderer = renderer
        self.error = error
        super().__init__(f"Error rendering block content: {block.content}. with renderer {renderer.__name__}: {error}")


class RenderersClassDict(TypedDict):
    content: Type[ContentRenderer] | None
    items: Type[ItemsRenderer] | None

class RendererContext:
    _content_renderer: ContentRenderer
    _items_renderer: ItemsRenderer 
    is_wrapper: bool
    _depth: int
    
    def __init__(self, content_renderer: ContentRenderer, items_renderer: ItemsRenderer, is_wrapper: bool, depth: int = 0):
        self._content_renderer = content_renderer
        self._items_renderer = items_renderer
        self._depth = depth
        self.is_wrapper = is_wrapper
        
    def set_renderer(self, renderer: Type[Renderer]):
        if renderer.target == "content" and not isinstance(self._content_renderer, renderer):
            self._content_renderer = renderer() # type: ignore
        elif renderer.target == "items" and not isinstance(self._items_renderer, renderer):
            self._items_renderer = renderer() # type: ignore
    
    def render_content(self, block: Block, items_content: List[str]) -> str:
        try:
            return self._content_renderer(block, items_content, self._depth)
        except Exception as e:
            print(print_block_tree(block, e))
            raise e

    def render_items(self, block: Block, inner_content: List[str]) -> List[str]:
        try:
            return self._items_renderer(block, inner_content, self._depth)
        except Exception as e:
            print(print_block_tree(block, e))
            raise e
    
    
    def copy(self, replace: RenderersClassDict | None = None, inc_depth: bool = True) -> "RendererContext":
        content_renderer = self._content_renderer
        items_renderer = self._items_renderer
        if replace is not None:
            if replace["content"] is not None and not isinstance(content_renderer, replace["content"]):
                content_renderer = replace["content"]()
            if replace["items"] is not None and not isinstance(items_renderer, replace["items"]):
                items_renderer = replace["items"]()            
        depth = self._depth
        if inc_depth and not self.is_wrapper:
            depth += 1     
        return RendererContext(content_renderer, items_renderer, False, depth)
    
    
    def get_class_dict(self) -> dict[str, Type[Renderer]]:
        return {
            "content": self._content_renderer.__class__,
            "items": self._items_renderer.__class__
        }
        
        


class BlockRenderer:
    
    renderer_lookup: dict[str, Type[Renderer]]
    
    def __init__(self, style_manager: StyleManager, renderer_lookup: dict[str, Type[Renderer]]):
        self.style_manager = style_manager
        self.renderer_lookup = renderer_lookup
    
    
    
    def build_ctx(self, block: Block, parent_ctx: "RendererContext | None" = None) -> "RendererContext":

        ctx = RendererContext(
            # MarkdownTitleRenderer(),
            IndentRenderer(),
            MarkdownParagraphRenderer(),
            block.is_wrapper,
        )                
        try:
            for tag in block.inline_style.style:
                try:
                    renderer_cls = self.renderer_lookup[tag]
                except KeyError:
                    raise UndefinedTagError(f"Tag {tag} not found in renderer lookup. Implement a renderer for this tag.")
                else:
                    ctx.set_renderer(renderer_cls)
            return ctx
        except Exception as e:
            # logger.exception("Error building context")
            raise e        
        

    def render(self, block: Block, ctx: RendererContext | None = None) -> str:                
        ctx = self.build_ctx(block, ctx)
        items = [self.render(
                item, 
                ctx,
            ) for item in block.items]
        items_content = ctx.render_items(block, items)
        return ctx.render_content(block, items_content)