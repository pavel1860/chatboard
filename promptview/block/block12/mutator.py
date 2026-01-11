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


from .block import Block, BlockChildren
from .chunk import Chunk
# if TYPE_CHECKING:
#     from .block import Block
#     from .chunk import ChunkMeta



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


def _apply_metadata(to_block: Block, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None):
    to_block.tags = tags or to_block.tags
    to_block.role = role or to_block.role
    to_block.style = style or to_block.style
    to_block.attrs = attrs or to_block.attrs
    return to_block


class Stylizer(metaclass=MutatorMeta):
    styles = ()
    
    def on_append(self, chunk):
        raise NotImplementedError("Stylizer.append is not implemented")
        
        
    def on_child(self, child: Block):
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
    _committed: bool = False


    def __init__(self, block: Block):
        self.block = block

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

        Base implementation returns None (no separate tail).
        Override in subclasses that create closing structures.
        """
        if self.body:
            return self.body[-1].tail
        return self.head

    @property
    def content(self) -> str:
        """
        Get the content text (unstyled chunks only).

        Delegates to block.content property.
        """
        return self.block.content
        

    
    @classmethod
    def create_block(cls, content: str, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None) -> Block:        
        block = Block(content)
        block = cls.init(block)
        block = _apply_metadata(block, tags, role, style, attrs)
        block.mutator = cls(block)
        return block
    
    @classmethod
    def init(cls, block: Block) -> Block:
        return block
    
    
    def on_append(self, child: Block) -> Generator[Block | Chunk, Any, Any]:
        raise NotImplementedError("Mutator.on_append is not implemented")
    
    def on_append_child(self, child: Block) -> Generator[Block | Chunk, Any, Any]:
        raise NotImplementedError("Mutator.on_append_child is not implemented")

    
    def commit(self, block: Block) -> Block | None:
        return None
    
    
    def _transfer_metadata(self, from_block: Block, to_block: Block):        
        to_block.tags = from_block.tags.copy()
        to_block.role = from_block.role
        to_block.style = from_block.style.copy()
        to_block.attrs = from_block.attrs.copy()
        return from_block, to_block
    



        
        
            
        
        
    
    
class BlockMutator(Mutator):
    styles = ("block",)
    
        
    def on_append_child(self, child: Block) -> Generator[Block | Chunk, Any, Any]:
        if not child.prev().has_newline():
            yield child.prev().add_newline(style="block")
        
        
        
        
        
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

    # @property
    # def body(self) -> BlockChildren:
    #     """Content blocks between opening and closing tags."""
    #     # if not self.block.children:
    #     #     return []
    #     if self._committed and len(self.block.children) > 1:
    #         # Exclude head (first) and tail (last)
    #         return self.block.children[1:-1]
    #     # Before commit: everything after head is body
    #     return self.block.children[1:]
    
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

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @classmethod
    def init(cls, block: Block) -> Block:
        with Block() as blk:
            blk /= block.text
            blk[0].prepend("<", style="xml")
            blk[0].append(">", style="xml")
        return blk

    def commit(self, block: Block) -> Block:
        if not self.tail.has_newline():       
            self.tail.add_newline(style="xml")
        post_fix = Block("</" + self.content + ">")
        block._raw_append_child(post_fix, to_body=False)
        return post_fix

    def on_append_child(self, child: Block) -> Generator[Block | Chunk, Any, Any]:
        if not child.prev().has_newline():
            yield child.prev().add_newline(style="xml")
        yield child.indent(style="xml")
        
        
        