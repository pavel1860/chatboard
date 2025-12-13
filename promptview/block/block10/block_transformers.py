from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import textwrap
from typing import Generator, Literal, Type
from promptview.block.block10.block import BlockBase, Block
import contextvars



style_registry_ctx = contextvars.ContextVar("style_registry_ctx", default={})

TargetType = Literal["content", "block", "children", "tree", "subtree"]



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
        default: "type[BaseTransformer] | None"=None
    ) -> "list[Type[BaseTransformer]]":
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
class RenderContext:
    max_path_len: int = 0
    max_path: list[int] = field(default_factory=list)
    num_blocks: int = 0
    index: int = 0
    depth: int = 0
    total_depth: int = 0
    renderers: "list[BaseTransformer]" = field(default_factory=list)
    ctx_renderers: "list[BaseTransformer]" = field(default_factory=list)
    ctx_block: BlockBase | None = None    
    parent_ctx: "RenderContext | None" = None
    is_wrapper: bool = False
   
    
    
    
    
class BaseTransformer(metaclass=StyleMeta):
    styles = []
    target = {"block"}
    effects = "all"
    
    def __init__(self, block: BlockBase):
        self.block = block
    
    def render(self, block: BlockBase) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")
    
    
    
    
class ContentTransformer(BaseTransformer):
    styles = ["content"]
    target = {"content"}
    
    def render(self, block: BlockBase) -> BlockBase:
        block.postfix_append("\n")
        return block




class MarkdownHeaderTransformer(BaseTransformer):
    styles = ["markdown", "md"]
    target = {"content"}
    
    def render(self, block: BlockBase) -> BlockBase:
        # block.prefix_prepend("#" * len(block.path) + " ")
        # block.prefix_prepend("#" + " ")
        # block.postfix_append("\n")     
        block = "#" + block + "\n"
        
        return block


class XmlTransformer(BaseTransformer):
    styles = ["xml"]
    target = {"content"}
    
    def render(self, block: BlockBase) -> BlockBase:
        if len(block) == 0:
            block = "<" & block & "/>\n"
            return block
        # elif len(block) == 1:
        #     block = "<" & block & ">"
        # else:
        #     block = "<" & block & ">\n"
        content = block.content
        with Block() as blk:
            blk /= "<" & block & ">"
            blk /= "</" & content & ">\n"
        return blk
    
    
def build_fiber_context(block: BlockBase) -> RenderContext:    
    # add default block renderer
    renderers = OrderedDict[str, BaseTransformer]({
        "ContentTransformer": ContentTransformer(block)
    })
    
    renderers.update({r.__class__.__name__: r for r in ctx.ctx_renderers if r.target.issubset({"children", "tree", "subtree"})})
    
    renderers.update({r.__class__.__name__: r(block) for r in StyleMeta.resolve(block.styles, {"content"})})
    renderers.update({r.__class__.__name__: r(block) for r in StyleMeta.resolve(block.styles, {"block"})})
    renderers.update({r.__class__.__name__: r(block) for r in StyleMeta.resolve(block.styles, {"tree"})})
        
    children_renderers = OrderedDict[str, BaseTransformer]({r.__class__.__name__: r for r in ctx.ctx_renderers if r.target.issubset({"tree", "subtree"})})
    children_renderers.update({r.__class__.__name__: r(block) for r in StyleMeta.resolve(block.styles, {"children"})})
    children_renderers.update({r.__class__.__name__: r(block) for r in StyleMeta.resolve(block.styles, {"subtree"})})

    
    new_ctx = RenderContext(
        # max_path_len=ctx.max_path_len,
        # max_path=ctx.max_path,
        # num_blocks=ctx.num_blocks,
        # index=ctx.index + 1,
        # depth=ctx.depth + (1 if not block.is_wrapper else 0),
        # total_depth=ctx.total_depth + 1,
        renderers=list(renderers.values()),
        ctx_renderers=list(children_renderers.values()),
        ctx_block=block,
        # parent_ctx=ctx,
        # is_wrapper=block.is_wrapper
    )
    
    return new_ctx
  
    
    
def transform_with_styles(block: BlockBase) -> BlockBase:    
    renderers = StyleMeta.resolve(block.styles, {"content"}, default=ContentTransformer)
    for renderer in renderers:
        block = renderer(block).render(block)
    return block


def transform(block: BlockBase) -> BlockBase:
    """
    Transform a block tree by applying style renderers.

    Uses post-order traversal (children first, then parent) so that
    parent transformers can wrap already-transformed children.
    """
    # Transform children first (recursive)
    for i, child in enumerate(block.children):
        transformed_child = transform(child)
        if transformed_child is not child:
            # Get insertion point - where the old child's chunks were
            old_start = child.start_chunk
            old_end = child.end_chunk

            # Remove old child's chunks from parent's BlockText
            block._block_text.replace(old_start, old_end, [])

            # Find where to insert - after previous child or after parent content
            if i > 0:
                insert_after = block.children[i - 1].end_chunk
            else:
                insert_after = block.end_postfix_chunk or block.content_end_chunk

            # Move transformed child's chunks into parent's BlockText
            block._block_text.extend_block_text(
                transformed_child._block_text,
                after=insert_after,
                copy=False
            )

            # Update BlockText references for transformed child subtree
            block._remap_block_text(transformed_child, block._block_text)

            # Replace in parent's children list
            block.children[i] = transformed_child
            transformed_child.parent = block

    # Then transform this block
    renderers = StyleMeta.resolve(
        block.styles,
        {"content"},
        # default=ContentTransformer
    )
    for renderer in renderers:
        block = renderer(block).render(block)

    return block
    
    
    
    
    
    
# def render(source_block: BlockBase) -> str:
    # block = source_block.copy()
    # transform(block)
    # return block.render()