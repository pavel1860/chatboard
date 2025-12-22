from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import textwrap
from typing import Generator, Literal, Type, TypedDict
from .block import BlockBase, Block, BlockSchema, ContentType, BlockListSchema
from .path import Path
from .chunk import BlockChunk
import contextvars
from ...utils.function_utils import is_overridden
from ...utils.type_utils import UNSET, UnsetType




style_registry_ctx = contextvars.ContextVar("style_registry_ctx", default={})

TargetType = Literal["content", "block", "children", "tree", "subtree"]



    
    
@dataclass
class TransformerConfig:
    list: "type[ListTransformer] | None" = None
    block: "Type[BlockTransformer] | None" = None
    content: "Type[ContentTransformer] | None" = None
    body: "Type[BodyTransformer] | None" = None
    prefix: "Type[PrefixTransformer] | None" = None
    hidden: "bool" = False
    is_wrapper: "bool" = False
    
    
    def get(self, key: str, default: "Type[BaseTransformer] | None" = None) -> "Type[BaseTransformer] | None":
        if self.block is not None and key in ["content"]:
            return default
        return getattr(self, key)
    
    def __getitem__(self, key: str) -> "Type[BaseTransformer] | None":
        return self.get(key)
    
    def __setitem__(self, key: str, value: "Type[BaseTransformer] | None"):            
        setattr(self, key, value)
        
    def iter_transformers(self):
        if self.is_wrapper:
            order = ["list", "body"]
        else:
            order = ["list", "block", "content", "body", "prefix"]
        for target in order:
            if transformer := self.get(target):
                yield (target, transformer)
        
    def get_body_transformers(self) -> "TransformerConfig":
        return TransformerConfig(
            body=self.body,
        )
        
        
    
        
        
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
    
    @staticmethod
    def filter_styles(styles: list[str]):
        config = TransformerConfig()
        current = style_registry_ctx.get()
        for style in styles:
            if style_cls := current.get(style):
                config[style_cls.target] = style_cls
        if config.get("block") is not None:
            config["content"] = None
        if config.get("content") is not None:
            config["body"] = None
        return config
    
    @classmethod
    def list_styles(cls) -> list[str]:
        current = style_registry_ctx.get()
        return list(current.keys())
    
    @classmethod
    def resolve(
        cls, 
        styles: list[str], 
        targets: set[TargetType] | None = None,
        default: "type[BaseTransformer] | None"=None
    ) -> "TransformerConfig":
        current = style_registry_ctx.get()
        transformer_cfg = TransformerConfig()
        for style in styles:
            if style == "hidden":
                transformer_cfg.hidden = True                
            elif style_cls := current.get(style): 
                if targets is None or style_cls.target in targets: 
                    transformer_cfg[style_cls.target] = style_cls              
        # if not renderers and default is not None:
            # return [default]
        if transformer_cfg.get("block") is not None:
            transformer_cfg["content"] = None
            
        return transformer_cfg
    
    
    
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
    """
    Base class for block transformers.

    Acts as a wrapper around the original block, providing:
    - Read-only access to original block's properties (no copy needed)
    - Lazy copy-on-write for mutations via self.copy property

    Subclasses implement render() to transform blocks.
    """
    styles = []

    def __init__(self, block: BlockBase):
        self._original = block
        self._copy: BlockBase | None = None

    # -------------------------------------------------------------------------
    # Read-only access to original block (no copy needed)
    # -------------------------------------------------------------------------

    @property
    def original(self) -> BlockBase:
        """Access the original block (read-only)."""
        return self._original

    @property
    def content_str(self) -> str:
        """Get content string from original block."""
        return self._original.content_str

    @property
    def children(self) -> list[BlockBase]:
        """Get children from original block."""
        return self._original.children

    @property
    def tags(self) -> list[str]:
        """Get tags from original block."""
        return self._original.tags

    @property
    def styles_list(self) -> list[str]:
        """Get styles from original block."""
        return self._original.styles

    @property
    def role(self) -> str | None:
        """Get role from original block."""
        return self._original.role

    @property
    def is_wrapper(self) -> bool:
        """Check if original block is a wrapper."""
        return self._original.is_wrapper

    @property
    def body(self):
        """Get body from original block."""
        return self._original.body

    def get(self, tag: str) -> BlockBase | None:
        """Get first block with tag from original."""
        return self._original.get(tag)

    def get_one(self, tag: str) -> BlockBase:
        """Get first block with tag from original (raises if not found)."""
        return self._original.get_one(tag)

    def get_one_or_none(self, tag: str) -> BlockBase | None:
        """Get first block with tag from original, or None."""
        return self._original.get_one_or_none(tag)

    def get_all(self, tags: str | list[str]) -> list[BlockBase]:
        """Get all blocks matching tag path from original."""
        return self._original.get_all(tags)

    def traverse(self):
        """Traverse original block tree."""
        return self._original.traverse()

    # -------------------------------------------------------------------------
    # Copy-on-write for mutations
    # -------------------------------------------------------------------------

    @property
    def copy(self) -> BlockBase:
        """
        Get a mutable copy of the block's content.

        Creates copy on first access (lazy copy-on-write).
        Use this for any modifications.
        """
        if self._copy is None:
            self._copy = self._original.copy_metadata()
        return self._copy

    def set_copy(self, block: BlockBase):
        """Set the copy directly (used when copy is created externally)."""
        self._copy = block

    @property
    def has_copy(self) -> bool:
        """Check if a copy has been created."""
        return self._copy is not None

    # -------------------------------------------------------------------------
    # Transform methods (to be implemented by subclasses)
    # -------------------------------------------------------------------------

    def render(self, block: BlockBase, path: Path) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")

    def instantiate(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")

    def append(self, block: BlockBase, chunk: BlockChunk, as_child: bool = False, start_offset: int | None = None, end_offset: int | None = None) -> BlockBase:
        raise NotImplementedError("Subclass must implement this method")

    def commit(self, block: BlockBase, content: ContentType, style: str | None = None, role: str | None = None, tags: list[str] | None = None):
        raise NotImplementedError("Subclass must implement this method")
    


class PrefixTransformer(BaseTransformer):
    styles = ["prefix"]
    target = "prefix"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        return block


class ListTransformer(BaseTransformer):
    styles = ["list"]
    target = "list"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        return block


class ContentTransformer(BaseTransformer):
    styles = ["content"]
    target = "content"
    
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        block.postfix_append("\n")
        return block


class BodyTransformer(BaseTransformer):
    styles = ["body"]
    target = "body"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        return block


class BlockTransformer(BaseTransformer):
    styles = ["block"]
    target = "block"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        return block



class PathTransformer(PrefixTransformer):
    styles = ["path"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        max_depth = block.max_depth()
        for child in block.traverse(body_only=True):
            cp = child.path - path
            padding = " " * (2 * (max_depth - cp.depth) - 1 )
            prefix = padding + str(cp) + "> "
            child.prefix_prepend(prefix)
            # child.indent_body(len(prefix))
        return block

class NumberedListTransformer(ListTransformer):
    styles = ["numbered-list", "num-li"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        
        for child in block.children:
            prefix = f"{child.path.indices[-1] + 1}. "
            child.prefix_prepend(prefix)
            child.indent_body(len(prefix))
        
        return block
    
    
class AlphaListTransformer(ListTransformer):
    styles = ["alpha-list", "alpha-li"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        for child in block.children:
            prefix = f"{chr(96 + child.path.indices[-1])}. "
            child.prefix_prepend(prefix)
            child.indent_body(len(prefix))
        return block
    
class RomanListTransformer(ListTransformer):
    styles = ["roman-list", "roman-li"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        from ...utils.string_utils import int_to_roman
        for child in block.children:
            prefix = f"{int_to_roman(child.path.indices[-1])}. "
            child.prefix_prepend(prefix)
            child.indent_body(len(prefix))
        return block
    
class DashListTransformer(ListTransformer):
    styles = ["dash-list", "dash-li"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        for child in block.children:
            prefix = "- "
            child.prefix_prepend(prefix)
            child.indent_body(len(prefix))
        return block
    
    
class AsteriskListTransformer(ListTransformer):
    styles = ["asterisk-list", "asterisk-li"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        for child in block.children:
            prefix = "* "
            child.prefix_prepend(prefix)
            child.indent_body(len(prefix))
        return block
    
    






class MarkdownHeaderTransformer(ContentTransformer):
    styles = ["markdown", "md"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        # block.prefix_prepend("#" * len(block.path) + " ")
        # block.prefix_prepend("#" + " ")
        # block.postfix_append("\n")
        print(path)     
        block = ("#" * path.depth) + block        
        return block
    
    
class BannerTransformer(ContentTransformer):
    styles = ["banner"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        with Block() as blk:
            with blk() as header:
                header /= "=" * 51
                header /= block.content_str
                header /= "=" * 51
            for child in block.children:
                blk /= child
        return blk

class InlineTransformer(BodyTransformer):
    styles = ["inline"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        block.remove_new_line()
        return block


class AsteriskTransformer(ContentTransformer):
    styles = ["astrix"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        block = "*" & block & "*"
        return block

class XmlTransformer(ContentTransformer):
    styles = ["xml"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        if len(block) == 0:
            block = "<" & block & "/>"
            return block
        # elif len(block) == 1:
        #     content = block.content
        #     blk = Block()
        #     blk.append_child("<" & block & ">", add_new_line=False)
        #     blk.append_child("</" & content & ">", add_new_line=False)
        #     return blk                
        else:
            content = block.content
            with Block() as blk:
                blk /= "<" & block.indent_body(2) & ">"
                blk /= "</" & content & ">"
            return blk
    
    def instantiate(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None) -> BlockBase:
        # print("inst>", repr(content))
        with Block(style=style, role=role, tags=tags) as blk:
            blk /= content
        return blk
    
    def append(self, block: BlockBase, chunk: BlockChunk, as_child: bool = False, start_offset: int | None = None, end_offset: int | None = None) -> BlockBase:
        if len(block) == 2:
            block.children[1].append(chunk, sep="", as_child=as_child, start_offset=start_offset, end_offset=end_offset)
        else:
            if as_child or len(block.children[0]) > 0:    
                block.children[0].append(chunk, sep="", as_child=as_child, start_offset=start_offset, end_offset=end_offset)
            else:
                block.append(chunk, sep="", as_child=as_child, start_offset=start_offset, end_offset=end_offset)
        return block
  
    
    def commit(self, block: BlockBase, content: ContentType, style: str | None = None, role: str | None = None, tags: list[str] | None = None):
        block.append_child(content, add_new_line=False)
        # block /= content
        return block
    
    
class XmlDefTransformer(ContentTransformer):
    styles = ["xml-def"]
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        print(path.tag_str())
        block.strip()
        block = " " * (path.depth - 1) + "<" & block & "> - "
        return block
            
            
class ToolDescriptionTransformer(BlockTransformer):
    styles = ["tool-desc"]

    def render(self, block: BlockBase, path: Path) -> BlockBase:
        # Read from original via self (no copy needed for reading)
        key_field = self.get_one("key-field")
        description = self.get_one("description")
        parameters = self.get_one("parameters")

        # Build new output
        with Block("# Name: " + key_field.body[0].content_str) as blk:
            with blk("## Purpose") as purpose:
                purpose /= description.body[0].content_str
            with blk("## Parameters") as params:
                for param in parameters.children:
                    with params(param.content_str) as param_blk:
                        param_blk /= param.body
                        if hasattr(param, 'type_str') and param.type_str is not None:
                            param_blk /= "Type:", param.type_str
                        if hasattr(param, 'is_required'):
                            param_blk /= "Required:", param.is_required                        
        return blk
            
    
class XmlListTransformer(BaseTransformer):
    styles = ["xml-list"]
    target = "content"
    
    def render(self, block: BlockBase, path: Path) -> BlockBase:
        # block /= "{... more items}"
        return block
    
    
    
class BlockSchemaTransformer:
    
    def __init__(self, block_schema: BlockSchema, transformers: list[BaseTransformer]):
        self.block_schema = block_schema
        self._block = None
        transformer_lookup = {}
        for t in transformers:
            transformer_lookup[t.target] = t
        self.transformer_lookup = transformer_lookup
        self._did_init = False
        self._did_commit = False
        
    @classmethod
    def from_block_schema(cls, block: BlockSchema) -> "BlockSchemaTransformer":
        transformer_cfg = StyleMeta.resolve(
            block.styles,
            # targets={"content"},
        )    
        return cls(block, [transformer(block) for _, transformer in transformer_cfg.iter_transformers()])
    
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
            self._block_transformer = content_transformer
        self._did_init = True
        return self._block
    
    
    def append(self, chunk: BlockChunk, force_schema: bool = False, as_child: bool = False, start_offset: int | None = None, end_offset: int | None = None):
        content_transformer = self.transformer_lookup.get("content")
        if force_schema or content_transformer is None or not is_overridden(content_transformer.__class__, "append", BaseTransformer):
            self.block.append(chunk, sep="", as_child=as_child, start_offset=start_offset, end_offset=end_offset)
        else:
            content_transformer.append(self.block, chunk, as_child=as_child, start_offset=start_offset, end_offset=end_offset)
            
            
    def append_child(self, child: "BlockSchemaTransformer"):
        block = self.block.append_child(child.block, copy=False)
        return child

    def commit(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None, force_schema: bool = False):
        content_transformer = self.transformer_lookup.get("content")
        if force_schema or content_transformer is None or not is_overridden(content_transformer.__class__, "commit", BaseTransformer):
            pass
        elif content is not None:    
            content_transformer.commit(self.block, content=content, style=style, role=role, tags=tags)
        self._did_commit = True
        return content_transformer
    
    
  
    
    
def transform_with_styles(block: BlockBase) -> BlockBase:    
    renderers = StyleMeta.resolve(block.styles, {"content"}, default=ContentTransformer)
    for renderer in renderers:
        block = renderer(block).render(block)
    return block


def transform(block: BlockBase, depth: int = 0) -> BlockBase:
    """
    Transform a block tree by applying style renderers.

    Uses post-order traversal (children first, then parent) so that
    parent transformers can wrap already-transformed children.

    Builds the transformed tree incrementally - each block is copied
    individually and children are appended, avoiding upfront deep copy.
    """
    # Transform children first (recursive)
    transformed_children = []
    render_cfg = StyleMeta.resolve(
        block.styles     
    )
    render_cfg.is_wrapper = block.is_wrapper
    # should not render children if block renderer is present
    if render_cfg.block is None:
        for child in block.children:
            transformed_children.append(transform(child, depth + 1))

    # Copy this block's metadata and content (not children)
    path = block.path
    
    if render_cfg.hidden:
        new_block = Block(tags=block.tags)
    else:
        new_block = block.copy_metadata()

    # Add transformed children to the new block
    # Use copy=False since children are already fresh copies from recursive transform
    for child in transformed_children:
        new_block.append_child(child, copy=False, add_new_line=False)        

    # Apply style renderers to the new block
    for target, transformer in render_cfg.iter_transformers():
        new_block = transformer(block).render(new_block, path)
        
        
    # for child in transformed_children:
    #     new_block.append_child(child)
    if depth == 0:
        new_block.last_descendant.remove_new_line()

    return new_block
    
    
def gather_transformers(block: BlockSchema) -> dict[str, BlockSchemaTransformer]:
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
    transformers_lookup.update({block.path: BlockSchemaTransformer(new_block, [renderer(new_block) for renderer in renderers])})
    return transformers_lookup