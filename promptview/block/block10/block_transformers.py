from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import textwrap
from typing import Generator, Literal, Type
from .block import BlockBase, Block, BlockSchema, ContentType, BlockListSchema
from .path import Path
from .chunk import BlockChunk
import contextvars
from ...utils.function_utils import is_overridden
from ...utils.type_utils import UNSET, UnsetType




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
        renderers = {}
        for style in styles:
            if style_cls := current.get(style): 
                if style_cls.target in targets: 
                    renderers[style_cls.target] = style_cls              
        if not renderers and default is not None:
            return [default]
        return list(renderers.values())
    
    
    
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
    
    def __init__(self, block: BlockBase):
        self.block = block
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")
    
    
    def instantiate(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")
    
    def append(self, block: BlockBase, chunk: BlockChunk) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")
    
    def commit(self, block: BlockBase, content: ContentType, style: str | None = None, role: str | None = None, tags: list[str] | None = None):
        raise NotImplementedError("Subclass must implement this method")
    



class ContentTransformer(BaseTransformer):
    styles = ["content"]
    target = "content"
    
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        block.postfix_append("\n")
        return block




class MarkdownHeaderTransformer(ContentTransformer):
    styles = ["markdown", "md"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        # block.prefix_prepend("#" * len(block.path) + " ")
        # block.prefix_prepend("#" + " ")
        # block.postfix_append("\n")     
        block = "#" + block + "\n"
        
        return block


class XmlTransformer(ContentTransformer):
    styles = ["xml"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        if len(block) == 0:
            block = "<" & block & "/>\n"
            return block
        content = block.content
        with Block() as blk:
            blk /= "<" & block & ">"
            blk /= "</" & content & ">"
        return blk
    
    def instantiate(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None) -> BlockBase:
        with Block( style=style, role=role, tags=tags) as blk:
            blk /= content
        return blk
    
    def append(self, block: BlockBase, chunk: BlockChunk) -> BlockBase:
        # block.append(chunk, sep="")
        # print(repr(chunk.content), chunk.isspace())
        block.children[0].append(chunk, sep="")
        return block
    
    def commit(self, block: BlockBase, content: ContentType, style: str | None = None, role: str | None = None, tags: list[str] | None = None):
        block /= content
        return block
    
    
class XmlDefTransformer(ContentTransformer):
    styles = ["xml-def"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        print(path.tag_str())
        block.strip()
        block = " " * (path.depth - 1) + "<" & block & "> - "
        return block
            
            
            
            
    
class XmlListTransformer(BaseTransformer):
    styles = ["xml-list"]
    target = "content"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        # block /= "{... more items}"
        return block
    
    
    
class BlockTransformer:
    
    def __init__(self, block_schema: BlockSchema, transformers: list[BaseTransformer]):
        self.block_schema = block_schema
        self._block = None
        transformer_lookup = {}
        for t in transformers:
            transformer_lookup[t.target] = t
        self.transformer_lookup = transformer_lookup
        
    @classmethod
    def from_block_schema(cls, block: BlockSchema) -> "BlockTransformer":
        transformers = StyleMeta.resolve(
            block.styles,
            targets={"content"},
        )    
        return cls(block, [transformer(block) for transformer in transformers])
    
    @property
    def block(self) -> BlockBase:
        if self._block is None:
            raise RuntimeError("Block not instantiated")
        return self._block
    
    @property
    def is_list(self) -> bool:
        return isinstance(self.block_schema, BlockListSchema)
    
    @property
    def is_list_item(self) -> bool:
        return self.block_schema.is_list_item
        
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        render_order = ["content"]
        for trans_type in render_order:
            if transformer := self.transformer_lookup.get(trans_type):
                block = transformer.render(block, path)
            else:
                raise ValueError(f"Transformer for {trans_type} not found")
        return block
    
    def instantiate(
        self,
        content: ContentType | None = None,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
        force_schema: bool = False,
    ) -> BlockBase:
        content_transformer = self.transformer_lookup.get("content")
        if force_schema or content_transformer is None or not is_overridden(content_transformer.__class__, "instantiate", BaseTransformer):
            self._block = self.block_schema.instantiate(
                content=content, 
                style=style if style is not UNSET else self.block_schema.styles, 
                role=role if role is not UNSET else self.block_schema.role, 
                tags=tags if tags is not UNSET else self.block_schema.tags,
            )        
        else:
            self._block = content_transformer.instantiate(
                content=content, 
                style=style if style is not UNSET else self.block_schema.styles, 
                role=role if role is not UNSET else self.block_schema.role, 
                tags=tags if tags is not UNSET else self.block_schema.tags,
            )
        return self._block
    
    
    def append(self, chunk: BlockChunk, force_schema: bool = False):
        content_transformer = self.transformer_lookup.get("content")
        if force_schema or content_transformer is None or not is_overridden(content_transformer.__class__, "append", BaseTransformer):
            self.block.append(chunk, sep="")
        else:
            content_transformer.append(self.block, chunk)
            
            
    def append_child(self, child: "BlockTransformer"):
        block = self.block.append_child(child.block, copy=False)
        return child

    def commit(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None, force_schema: bool = False):
        content_transformer = self.transformer_lookup.get("content")
        if force_schema or content_transformer is None or not is_overridden(content_transformer.__class__, "commit", BaseTransformer):
            return 
        if content is not None:    
            content_transformer.commit(self.block, content=content, style=style, role=role, tags=tags)
        return content_transformer
    
    
  
    
    
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

    Builds the transformed tree incrementally - each block is copied
    individually and children are appended, avoiding upfront deep copy.
    """
    # Transform children first (recursive)
    transformed_children = []
    for child in block.children:
        transformed_children.append(transform(child))

    # Copy this block's metadata and content (not children)
    path = block.path
    new_block = block.copy_metadata()

    # Add transformed children to the new block
    for child in transformed_children:
        new_block.append_child(child)

    # Apply style renderers to the new block
    renderers = StyleMeta.resolve(
        new_block.styles,
        {"content"},
        # default=ContentTransformer
    )
    for renderer in renderers:
        new_block = renderer(new_block).render(new_block, path)

    return new_block
    
    
def gather_transformers(block: BlockSchema) -> dict[str, BlockTransformer]:
    """
    Gather all transformers for block tree.
    
    """
    transformers_lookup = {}
    for child in block.children:
        transformers_lookup.update(gather_transformers(child))

    # Copy this block's metadata and content (not children)
    new_block = block.copy_metadata()

    # Apply style renderers to the new block
    renderers = StyleMeta.resolve(
        new_block.styles,
        {"content"},
        # default=ContentTransformer
    )    
    transformers_lookup.update({block.path: BlockTransformer(new_block, [renderer(new_block) for renderer in renderers])})
    return transformers_lookup