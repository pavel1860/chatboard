from dataclasses import dataclass, field
import textwrap
from typing import Generator, Literal
from promptview.block.block9.block import Block, BlockChunk, BlockSent, AttrBlock
import contextvars




# header.print()

style_registry_ctx = contextvars.ContextVar("style_registry_ctx", default={})

_target2type = {
    "block": Block,
    "text": BlockSent,
    "chunk": BlockChunk,
}

_type2target = {
    Block: "block",
    BlockSent: "text",
    BlockChunk: "chunk",
}


def _style_key(style: str, target: str, effects: str="all") -> str:
    return f"{style}_{target}_{effects}" 


TargetType = Literal["block", "children", "tree", "subtree"]



class StyleMeta(type):
    # _registry: dict[str, "StyleMeta"] = {}
    # _styles: dict[str, "BlockStyle"] = {}
    
    
    def __new__(cls, name, bases, attrs):        
        new_cls = super().__new__(cls, name, bases, attrs)
        # cls._registry[name] = new_cls
        if styles := attrs.get("styles"):
            for style in styles:
                # style_obj = new_cls()
                # style_registry_ctx.get()[style] = style_obj
                style_registry_ctx.get()[style] = new_cls
        return new_cls
    
    
    
    @classmethod
    def list_styles(cls) -> list[str]:
        current = style_registry_ctx.get()
        return list(current.keys())
    
    @classmethod
    def resolve(
        cls, 
        styles: list[str], 
        targets: set[TargetType],
        default: "BaseRenderer | None"=None
    ) -> "list[Type[BaseRenderer]]":
        current = style_registry_ctx.get()        
        renderers = []
        for style in styles:
            if style_cls := current.get(style): 
                if style_cls.target.issubset(targets):               
                    renderers.append(style_cls)
        if not renderers and default is not None:
            return [default]
        return renderers
    
    
@dataclass
class BlockFiber:
    _block: Block 
    line_start: str = ""
    line_end: str = ""
    # seperator: str = ""    
    content: str = ""
    prefix: str = ""
    postfix: str = ""
    tab_num: int = 0
    tab: str = ""    
    line_seperator: str = "\n"
    block_postfix: str = ""
    children: "list[BlockFiber]" = field(default_factory=list)
    
    
    
    @property
    def path(self) -> list[int]:
        return self._block.path
    
    
@dataclass
class RenderContext:
    max_path_len: int = 0
    max_path: list[int] = field(default_factory=list)
    num_blocks: int = 0
    index: int = 0
    depth: int = 0
    
    
class BaseRenderer(metaclass=StyleMeta):
    styles = []
    target = {"block"}
    effects = "all"
    
    def __init__(self, block: Block):
        self.block = block
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext) -> BlockFiber:
        raise NotImplementedError("Subclass must implement this method")
    


    
class BlockRenderer(BaseRenderer):
    styles = ["block"]
    target = {"block"}
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext) -> BlockFiber:        
        content = "".join([c.content for c in block.content])
        fiber.content = content
        return fiber
   


class NumerateListRenderer(BaseRenderer):
    styles = ["num-li"]
    target = {"children"}
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
        fiber.prefix = f"{fiber.path[-1] + 1}. "
        return fiber

  
class MarkdownRenderer(BaseRenderer):
    styles = ["md"]
    target = {"block"}
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
        fiber.prefix = "#" * len(fiber.path) + " "
        return fiber
    
    
class XMLRenderer(BaseRenderer):
    styles = ["xml"]
    target = {"block"}
    
    def render_attr(self, attr: AttrBlock):
        content = f"{attr.name}=\""
        instructions = ""
        if attr.type in (int, float):
            instructions = " wrap in quotes"
        content += f"(\"{attr.type.__name__}\"{instructions}) {attr.description}"
        
        if attr.description:
            content += f" {attr.description}"
        if attr.gt is not None:
            content += f" gt={attr.gt}"
        if attr.lt is not None:
            content += f" lt={attr.lt}"
        if attr.ge is not None:
            content += f" ge={attr.ge}"
        if attr.le is not None:
            content += f" le={attr.le}"
        return content + "\""
    
    def render_all_attrs(self, block: Block):
        attrs = ""
        for attr in block.attrs.values():
            attrs += " " + self.render_attr(attr)
        return attrs

    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
        attrs = self.render_all_attrs(block)
        # if attrs:
        #     attrs = " " + attrs
        if not block.children:
            fiber.prefix = f"<"
            fiber.postfix = attrs + f"/>"
        else:
            fiber.prefix = f"<"
            fiber.postfix = attrs + f">"
            fiber.block_postfix = f"</{fiber.content}>\n"
        return fiber
    
    
# class PathRenderer(BaseRenderer):
#     styles = ["path"]
#     target = {"tree"}
    
#     def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
#         path_str = ".".join([str(p) for p in block.path])
#         path_str = path_str + " " * (ctx.max_path_len + ctx.max_path_len - 1 - len(path_str))
#         path_str += ": "
#         fiber.line_start = path_str
#         return fiber


class PathContainerRenderer(BaseRenderer):
    styles = ["path"]
    target = {"subtree"}
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
        parent_path_len = len(self.block.path)
        path_str = ".".join([str(p) for p in block.path[parent_path_len:]])
        path_str = path_str + " " * (2 * ctx.max_path_len - 1 - 2 * parent_path_len - len(path_str))
        # path_str = " " * (ctx.max_path_len + ctx.max_path_len - 1 - len(path_str)) + path_str
        path_str += ": "
        fiber.line_start = path_str
        return fiber


class LineRenderer(BaseRenderer):
    styles = ["lines"]
    target = {"tree"}
    
    def __call__(self, fiber: BlockFiber, block: Block, ctx: RenderContext):
        index = str(ctx.index)
        max_index_str = str(ctx.num_blocks)
        index = " " * (len(max_index_str) - len(index)) + index
        fiber.line_start = index + ": "        
        return fiber
    
    
def apply_styles(fiber: BlockFiber, block: Block, targets: set[TargetType], ctx: RenderContext):
    for style in StyleMeta.resolve(block.styles, targets):
        fiber = style(block)(fiber, block, ctx)
    return fiber


def apply_ctx_styles(fiber: BlockFiber, block: Block, ctx_renderers: set[BaseRenderer], ctx: RenderContext):
    renderers = StyleMeta.resolve(block.styles, {"tree"})
    for renderer in renderers:
        ctx_renderers.add(renderer(block))
    for renderer in ctx_renderers:
        fiber = renderer(fiber, block, ctx)
    return fiber, ctx_renderers


def merge_ctx_renderers(block: Block, ctx_renderers: set[BaseRenderer], targets: set[TargetType]) -> set[BaseRenderer]:
    renderers = StyleMeta.resolve(block.styles, targets)
    for renderer in renderers:
        ctx_renderers.add(renderer(block))
    return ctx_renderers

# def apply_renderers(fiber: BlockFiber, block: Block, ctx_renderers: set[BaseRenderer], ctx: RenderContext):
#     for renderer in ctx_renderers:
#         fiber = renderer(fiber, block, ctx)
#     return fiber

def build_fiber(ctx: RenderContext, block: Block, parent: Block | None = None, ctx_renderers: set[BaseRenderer] = set()) -> BlockFiber:
    ctx.index += 1
    fiber = BlockFiber(_block=block)
    fiber, ctx_renderers = apply_ctx_styles(fiber, block, ctx_renderers, ctx)
    ctx_renderers = merge_ctx_renderers(block, ctx_renderers, {"subtree"})
    fiber = apply_styles(fiber, block, {"block"}, ctx)
    if parent is not None:
        fiber = apply_styles(fiber, parent, {"children"}, ctx)
    fiber.children = [build_fiber(ctx, child, block, ctx_renderers) for child in block.children]    
    return fiber


def render_fiber(fiber: BlockFiber) -> str:
    prompt = ""
    prompt += fiber.line_start + fiber.prefix + fiber.content + fiber.postfix + fiber.line_end + fiber.line_seperator
    for child in fiber.children:
        prompt += render_fiber(child)
    if fiber.block_postfix:
        tabs = ""
        if fiber.line_start:
            tabs = " " * (len(fiber.line_start))
        prompt += tabs + fiber.block_postfix
    return prompt


def build_render_context(block: Block) -> RenderContext:
    num_blocks = 0
    max_path_len = 0
    max_path = []
    for blk in block.traverse():
        max_path_len = max(max_path_len, len(blk.path))
        max_path = max(max_path, blk.path)
        num_blocks += 1
    return RenderContext(
        max_path_len=max_path_len,
        max_path=max_path,
        num_blocks=num_blocks,        
    )

def render(block: Block) -> str:
    stats = build_render_context(block)
    fiber = build_fiber(stats, block, ctx_renderers={BlockRenderer(block)})
    return render_fiber(fiber)
