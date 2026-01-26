"""
Mutator - Strategy for style-aware block operations.

Mutators intercept block operations (append, prepend, append_child) and
modify behavior based on the block's style. The base Mutator provides
direct pass-through. Subclasses like XmlMutator add prefix/postfix handling.

MutatorMeta is a metaclass that registers mutators by their style names.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Type


from .block import Block, BlockChildren, ContentType
from .chunk import BlockChunk
# if TYPE_CHECKING:
#     from .block import Block
#     from .chunk import ChunkMeta


class RenderError(Exception):
    pass


@dataclass
class MutatorConfig:
    mutator: "Type[Mutator]"
    stylizers: list["Type[Stylizer]"] = field(default_factory=list)
    hidden: "bool" = False
    is_wrapper: "bool" = False
    
    
    def __post_init__(self):
        if self.mutator is None:
            self.mutator = Mutator
    
    def get(self, key: str, default: "Type[MutatorConfig] | None" = None) -> "Type[MutatorConfig] | None":
        return getattr(self, key)
    
    def __getitem__(self, key: str) -> "Type[MutatorConfig] | None":
        return self.get(key)
    
    def __setitem__(self, key: str, value: "Type[MutatorConfig] | None"):            
        setattr(self, key, value)
        
    def iter_transformers(self):
        if self.is_wrapper:
            order = ["list", "body"]
        else:
            order = ["list", "block", "content", "body", "prefix"]
        for target in order:
            if transformer := self.get(target):
                yield (target, transformer)
                
                
    def create_block(self, content: ContentType, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None, is_streaming: bool = False, type: Type | None = None) -> Block:
        from .transform import transform_context
        with transform_context(True):
            block = Block(content, tags=tags, role=role, style=style, attrs=attrs, type=type)
            block = self.mutator.init(block)
            block = _apply_metadata(block, tags, role, style, attrs, type=type)
            block.mutator = self.mutator(block, is_streaming=is_streaming)
            block.stylizers = [stylizer() for stylizer in self.stylizers]
        return block
    
    
    def _get_reused_blocks(self, block: Block, orig_block: Block):
        new_ids = set()
        for b in block.iter_depth_first():
            new_ids.add(id(b))
        reused_blocks = []
        for b in orig_block.iter_depth_first():
            if id(b) in new_ids:
                reused_blocks.append(b)
        return reused_blocks
    
    def build_block(self, orig_block: Block, is_streaming: bool = False) -> Block:
        from .transform import transform_context        
        with transform_context(True):
            block = self.mutator.init(orig_block)
            if reused_blocks := self._get_reused_blocks(block, orig_block):
                raise RenderError(f"You cannot reuse original blocks in a Mutator. Please use the block content instead.\n Reused blocks: {reused_blocks}")
            block = _apply_metadata(block, orig_block.tags, orig_block.role, orig_block.style, orig_block.attrs)
            block.mutator = self.mutator(block, is_streaming=is_streaming)
            block.stylizers = [stylizer() for stylizer in self.stylizers]
        return block




# Global registry of style -> mutator class
_style_registry: dict[str, type[Mutator]] = {}


class MutatorMeta(type):
    """
    Metaclass that registers Mutator subclasses by their styles.

    When a class with `styles = ["xml", ...]` is defined, this metaclass
    registers it in the global style registry for lookup.
    """

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        new_cls = super().__new__(mcs, name, bases, attrs)

        # Register by styles if defined
        if styles := attrs.get("styles"):
            for style in styles:
                _style_registry[style] = new_cls

        return new_cls

    @classmethod
    def get_mutator(mcs, style: str) -> type[Mutator]:
        """
        Get the Mutator class for a given style.

        Returns the base Mutator if no specific mutator is registered.
        """
        return _style_registry.get(style, Mutator)

    @classmethod
    def list_styles(mcs) -> list[str]:
        """List all registered styles."""
        return list(_style_registry.keys())
    
    
    
    @classmethod
    def resolve(
        cls, 
        styles: list[str] | None = None,
        default: "Type[Mutator] | None" = None,
    ) -> "MutatorConfig":
        # mutator_cfg = MutatorConfig(mutator=Mutator)
        mutator_cfg = MutatorConfig(mutator=default or Mutator)
        if styles is None:
            return mutator_cfg
        for style in styles:
            if style == "hidden":
                mutator_cfg.hidden = True                
            elif style_cls := _style_registry.get(style):
                if issubclass(style_cls, Stylizer):
                    mutator_cfg.stylizers.append(style_cls)
                else:
                    mutator_cfg["mutator"] = style_cls                          
        return mutator_cfg


def _apply_metadata(
    to_block: Block, 
    tags: list[str] | None = None, 
    role: str | None = None, 
    style: str | list[str] | None = None, 
    attrs: dict[str, Any] | None = None, 
    type: Type | None = None
    ) -> Block:
    to_block.tags = tags or to_block.tags
    to_block.role = role or to_block.role
    to_block.style = style or to_block.style
    to_block.attrs = attrs or to_block.attrs
    to_block._type = type or to_block._type
    return to_block



def _transfer_metadata(from_block: Block, to_block: Block):        
    to_block.tags = from_block.tags.copy()
    to_block.role = from_block.role
    to_block.style = from_block.style.copy()
    to_block.attrs = from_block.attrs.copy()
    return from_block, to_block


class Stylizer(metaclass=MutatorMeta):
    styles = ()
    
    def on_append(self, chunk):
        raise NotImplementedError("Stylizer.append is not implemented")
        
        
    def on_append_child(self, child: Block):
        raise NotImplementedError("Stylizer.append_child is not implemented")
        
     

class Mutator(metaclass=MutatorMeta):
    """
    Base mutator providing direct pass-through to block operations.

    Mutators wrap a block and intercept operations like append, prepend,
    and append_child. The base Mutator simply delegates to the block's
    raw operations without modification.

    Subclasses override methods to provide style-specific behavior:
    - XmlMutator adds prefix/content/postfix region handling
    - Other mutators can implement different transformation logic

    Attributes:
        block: The block this mutator operates on
        styles: Class attribute listing styles this mutator handles
    """

    styles: tuple[str, ...] = ()
    target: str = "content"
    _initialized: bool = False
    _committed: bool = False


    def __init__(self, block: Block, is_streaming: bool = False):
        self.block = block
        self.is_streaming = is_streaming

    # =========================================================================
    # Structure Access Properties
    # =========================================================================

    @property
    def head(self) -> Block:
        """
        Get the head block (e.g., opening tag in structured mutators).

        Base implementation returns the block itself.
        Override in subclasses that create wrapper structures.
        """
        return self.block

    @property
    def body(self) -> BlockChildren:
        """
        Get the body blocks (content children).

        Base implementation returns all children.
        Override in subclasses that separate head/body/tail.
        """
        return self.block.children

    @property
    def tail(self) -> Block:
        """
        Get the tail block (e.g., closing tag in structured mutators).
        Override in subclasses that create closing structures.
        """
        if self.body:
            return self.body[-1].tail
        return self.head

    # @property
    # def content(self) -> str:
    #     """
    #     Get the content text (unstyled chunks only).

    #     Delegates to block.content property.
    #     """
    #     return self.block.content
    @property
    def content(self) -> str:
        """
        Get the content text (unstyled chunks only).

        Delegates to block.content property.
        """
        block = self.block
        if not block.chunks:
            return block._text

        content_parts = []
        for chunk in block.chunks:
            if chunk.style is None:
                content_parts.append(block._text[chunk.start:chunk.end])
        return "".join(content_parts)
    
    
    def get_style(self):
        if not self.__class__.styles:
            return None
        return self.__class__.styles[0]

    def content_chunks(self) -> list[BlockChunk]:
        """
        Get content chunks only (chunks without a style).

        Returns:
            List of BlockChunk objects containing only unstyled content.
        """
        block = self.block.head
        if not block.chunks:
            if block._text:
                return [BlockChunk(block._text)]
            return []

        result = []
        for chunk in block.chunks:
            if chunk.style is None:
                content = block._text[chunk.start:chunk.end]
                if content:
                    result.append(BlockChunk(content, logprob=chunk.logprob))
        return result

    @classmethod
    def create_block(cls, content: ContentType, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None, is_streaming: bool = False, type: Type | None = None) -> Block:
        block = Block(content, tags=tags, role=role, style=style, attrs=attrs, type=type)
        block = cls.init(block)
        block = _apply_metadata(block, tags, role, style, attrs)
        block.mutator = cls(block, is_streaming=is_streaming)
        return block
    
    @classmethod
    def init(cls, block: Block) -> Block:
        return block.copy_head()
    
    
    def extract(self) -> Block:
        return Block(self.block.content_chunks(), tags=self.block.tags, role=self.block.role, style=self.block.style, attrs=self.block.attrs, type=self.block._type)
    
    
    def on_append(self, content: Block) -> Generator[Block | BlockChunk, Any, Any]:
        raise NotImplementedError("Mutator.on_append is not implemented")
    
    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        raise NotImplementedError("Mutator.on_append_child is not implemented")
    

    def call_commit(self, postfix: Block | None = None) -> Block | None:
        if self._committed:
            raise ValueError(f"Mutator {self.__class__.__name__} is already committed for block {self.block}")
        res = self.commit(self.block, postfix)
        self._committed = True
        return res
    
    def commit(self, block: Block, postfix: Block | None = None) -> Block | None:
        return None
    
    
    def _transfer_metadata(self, from_block: Block, to_block: Block):        
        to_block.tags = from_block.tags.copy()
        to_block.role = from_block.role
        to_block.style = from_block.style.copy()
        to_block.attrs = from_block.attrs.copy()
        return from_block, to_block
    



        
        
            
        
        
    
    
class BlockMutator(Mutator):
    styles = ("block",)
    
        
    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        prev = child.prev()
        if not prev.is_wrapper and not prev.has_newline():
            yield child.prev().add_newline(style=self.styles[0])
        
        
        
        
        
class XmlMutator(Mutator):
    """
    Mutator for XML-style block structures.

    Creates blocks with:
    - head: Opening tag block (<tag>)
    - body: Content children between tags
    - tail: Closing tag block (</tag>) after commit

    Structure after init:
        block
        └── children[0]: opening tag (head)

    Structure after commit:
        block
        ├── children[0]: opening tag (head)
        ├── children[1:-1]: body content
        └── children[-1]: closing tag (tail)
    """
    styles = ("xml",)

    # =========================================================================
    # Structure Access Properties
    # =========================================================================

    @property
    def head(self) -> Block:
        """Opening tag block."""
        return self.block.children[0]

    
    @property
    def body(self) -> BlockChildren:
        return self.block.children[0].children

    @property
    def tail(self) -> Block:
        """Closing tag block (only after commit)."""
        if self._committed:
            return self.block.children[-1]
        return super().tail

    @property
    def content(self) -> str:
        """Get the tag name (unstyled content of head block)."""
        if self.head:
            return self.head.content
        return ""
    
    @classmethod
    def render_attrs(cls, block: Block) -> str:
        attrs = ""
        for k, v in block.attrs.items():
            attrs += f"{k}=\"{v}\""
            
        if attrs:
            attrs = " " + attrs
        return attrs


    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @classmethod
    def init(cls, block: Block) -> Block:
        prefix, suffix = block.split_prefix("<", create_on_empty=True)
        content, postfix = suffix.split_postfix("/>")
        if not postfix:
            content, postfix = suffix.split_postfix(">", create_on_empty=True)
        content,_, _ = content.split(" ")
        content.snake_case()
        
        with Block(tags=["xml-container"]) as blk:
            with blk(content, tags=["xml-opening-tag"]) as opening_tag:
                opening_tag.append(cls.render_attrs(block), style="xml")
                opening_tag.prepend(prefix, style="xml")
                opening_tag.append(postfix, style="xml")
                # with opening_tag() as body:
                #     pass
        return blk
    

    def commit(self, block: Block, postfix: Block | None = None) -> Block:
        if not postfix and self.is_streaming:
            return None
        if not self.is_streaming and not block.tail.has_newline():       
            block.tail.add_newline(style="xml")
        if postfix is None:
            postfix = Block("</" + block.content + ">", tags=["xml-closing-tag"])
        # block._raw_append_child(postfix, to_body=False)
        block.append_child(postfix, to_body=False)
        return postfix
    
    # def on_append(self, content: Block):
    #     if not self.body and (self.head.has_newline() or content != "\n"):
    #         yield self.block.children[0].append_child("")
            
        

    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        prev = child.prev()
        if not self.is_streaming and not prev.has_newline():
            yield prev.add_newline(style="xml")
        yield child.indent(style="xml")
        
        
        
        
        
        


class MarkdownMutator(BlockMutator):
    styles = ("md",)
    
    
    @classmethod
    def init(cls, block: Block) -> Block:
        prefix, content = block.split_prefix(r"#+\s+", regex=True)        
        if not prefix:
            prefix = "#" * (block.path.depth + 1) + " "
        
        content.prepend(prefix, style="md")
        return content
        # if content:
        #     content.prepend(prefix, style="md")
        #     return content
        # else:
        #     return prefix.apply_style("md")            
        
        
        
    # @classmethod
    # def init(cls, block: Block) -> Block:
    #     block = block.copy_head()        
    #     block.prepend("\n" + "#" * (block.path.depth + 1) + " ", style="md")
    #     return block
    
    # def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
    #     prev = child.prev()
    #     if not self.is_streaming and not prev.has_newline():
    #         yield prev.add_newline(style="markdown")
        # yield child.indent(style="markdown")
        
        
        
class BannerMutator(BlockMutator):
    styles = ("banner",)
    
    @property
    def head(self) -> Block:
        return self.block.children[0].children[1]
    
    @property
    def body(self) -> BlockChildren:
        return self.block.children[1].children
    
    @classmethod
    def init(cls, block: Block) -> Block:
        with Block() as blk:
            block.prev().append("\n", style="banner")
            with blk() as header:
                header /= "=" * 51
                header /= block.content
                header /= "=" * 51
            with blk() as body:
                pass
        return blk
        
        
        
        
class MarkdownListStylizer(Stylizer):
    styles = ["li"]
    
    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        yield child.prepend("- ", style="li")
        
class MarkdownAstrixListStylizer(Stylizer):
    styles = ["li-ast"]
    
    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        yield child.prepend("* ", style="li-ast")
        
        
        
        
class MarkdownNumberListStylizer(Stylizer):
    styles = ["li-num"]
    
    def on_append_child(self, child: Block) -> Generator[Block | BlockChunk, Any, Any]:
        
        yield child.prepend(f"{child.path[-1] + 1}. ", style="li")
        # yield child.prepend(child.path[-1], style="li-num")
        
        
        
class XmlDefMutator(Mutator):
    styles = ("xml-def",)
    
    @classmethod
    def init(cls, block: Block) -> Block:
        block = " " * (block.path.depth - 1) + "<" + block + "> - "
        return block
    
    
class ToolDescriptionMutator(Mutator):
    styles = ("tool-desc",)
    target = "block"
    
    @classmethod
    def init(cls, block: Block) -> Block:
        description = block.get("description")
        parameters = block.get("parameters")

        # Build new output
        with Block("# Name: " + block.attrs.get("name", "")) as blk:
            with blk("## Purpose") as purpose:
                purpose /= description.body[0].content
            with blk("## Parameters") as params:
                for i, param in enumerate(parameters.body):
                    with params(f"{i + 1}.", tags=["param", param.content]) as param_blk: 
                        param_blk /= "  name: " + param.content
                        if hasattr(param, 'type_str') and param.type_str is not None:
                            param_blk /= "  Type: ", param.type_str
                        if hasattr(param, 'is_required'):
                            param_blk /= "  Required: ", param.is_required                        
                    # with params("name: " + param.content) as param_blk:                        
                    #     if hasattr(param, 'type_str') and param.type_str is not None:
                    #         param_blk /= "  Type:", param.type_str
                    #     if hasattr(param, 'is_required'):
                    #         param_blk /= "  Required:", param.is_required                        
        return blk

    
    
    