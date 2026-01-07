from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Callable, Type
from collections import UserList
from .span import BlockChunkList, Span, BlockChunk, SpanEvent
from .mutator_meta import MutatorMeta
from .path import Path, compute_path
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from promptview.utils.type_utils import UnsetType, UNSET
if TYPE_CHECKING:
    from .block_text import BlockText
    from .schema import BlockSchema, BlockListSchema, BlockList




def parse_style(style: str | list[str] | None) -> list[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []

BaseContentTypes = str | bool | int | float
ContentType = BaseContentTypes | list[str] | list[BlockChunk] | BlockChunkList | BlockChunk



class BlockChildren(UserList):
    
    def __init__(self, parent: Block, items: list[Block] | None = None):
        self.parent: Block = parent
        UserList.__init__(self, items)



@dataclass
class Mutator(metaclass=MutatorMeta):
    """
    Strategy for accessing and mutating block fields.

    Mutators provide indirection between blocks and their data,
    allowing transformations (like XML wrapping) to redirect
    field access to virtual locations.

    Base Mutator provides direct access (no transformation).
    Subclasses override accessors to redirect to different locations.
    """
    styles = ()
    
    _block: Block | None = None
    did_init: bool = False
    did_commit: bool = False

    # def __init__(self, block: Block | None = None, did_instantiate: bool = True, did_commit: bool = True):
    #     self._block: Block | None = block
    #     self.did_init: bool = did_instantiate
    #     self.did_commit: bool = False
        # self.is_rendered: bool = False

    @property
    def block(self) -> Block:
        if self._block is None:
            raise ValueError("Block is not set")
        return self._block
    
    @property
    def is_rendered(self) -> bool:
        return not self.__class__ is Mutator

    @block.setter
    def block(self, value: Block | None) -> None:
        self._block = value
        
    @property
    def head(self) -> Block:
        return self.block
    
    @property
    def tail(self) -> Block:
        if not self.body:
            return self.head
        return self.body[-1].tail

    
    @property
    def body(self) -> BlockChildren:
        return self.block.children
    
    @property
    def content(self) -> str:
        return self.block.span.extract_content().text
    
    
    # @property
    # def current_head(self) -> Block:
    #     if not self.body:
    #         return self.head
    #     # return self.body[-1].mutator.current_head
    #     return self.body[-1].mutator.tail
    
    # @property
    # def block_end(self) -> Span:
    #     if not self.body:
    #         return self.head
    #     return self.body[-1].mutator.block_end
    
    
    @property
    def block_postfix(self) -> Span | None:
        return None
    
    

    def is_head_closed(self) -> bool:
        return self.head.has_newline()
    
    def is_head_open(self, chunks: list[BlockChunk]) -> bool:
        return not self.head.has_newline()
    
    def is_last_block_open(self,  chunks: list[BlockChunk]) -> bool:
        if not self.body:
            return False
        elif self.body[-1].has_newline():
            return False
        else:
            return True
        
        
    def promote(self, content: ContentType, style: str | None = None) -> BlockChunkList:
        """
        Promote content of various types to list[Chunk].

        Override in subclasses for context-specific transformations
        (e.g., XML escaping, Markdown escaping).

        Handles:
            - str: wraps in single Chunk
            - bool: converts to lowercase str ("true"/"false")
            - int, float: converts to str
            - Chunk: wraps in list
            - list[str]: converts each to Chunk
            - list[Chunk]: returns as-is

        Returns:
            list[Chunk]
        """
        if isinstance(content, BlockChunkList):
            return content.apply_style(style)
        elif isinstance(content, str):
            return BlockChunkList(chunks=[BlockChunk(content=content, style=style)])
        elif isinstance(content, bool):
            return BlockChunkList(chunks=[BlockChunk(content=str(content).lower(), style=style)])
        elif isinstance(content, (int, float)):
            return BlockChunkList(chunks=[BlockChunk(content=str(content), style=style)])
        elif isinstance(content, BlockChunk):
            if style:
                content.style = style
            return BlockChunkList(chunks=[content])
        elif isinstance(content, list):
            if len(content) == 0:
                return BlockChunkList(chunks=[])
            if isinstance(content[0], str):
                return BlockChunkList(chunks=[BlockChunk(content=s, style=style) for s in content])
            elif isinstance(content[0], BlockChunk):
                if style:
                    for c in content:
                        c.style = style
                return BlockChunkList(chunks=content)
        return BlockChunkList(chunks=[])
    

   
   
    # def current_span(self) -> Span:        
    #     return self.body[-1].mutator.current_span() if self.body else self.head.span
        

        
            
            
        # self.body[-1].mutator.current_span()
    
    def _spawn_block(self, chunks: list[BlockChunk] | None = None) -> Block:
        block = Block(chunks, _auto_handle=self.block._auto_handle)        
        return block
        
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None):
        block = Block(chunks)        
        return block
    
    def call_init(self, chunks: BlockChunkList, path: Path, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, _auto_handle: bool = True, attrs: dict[str, Any] | None = None):
        block = self.init(chunks, path, attrs=attrs)
        block = self._apply_metadata(block, tags=tags, role=role, style=style)
        block._auto_handle = _auto_handle
        block.mutator = self
        self.block = block
        self.did_init = True
        return block
    
    def commit(self, chunks: BlockChunkList | None = None)-> Block | None:
        return None
    
    
    def call_commit(self, chunks: BlockChunkList | None = None) -> Block | None:
        """
        Call the commit method of the block.
        """
        block = self.commit(chunks)
        self.did_commit = True
        self.did_init = True
        return block
    
    def on_newline(self, chunk: BlockChunk):
        span = self.tail.span
        return span.append([chunk])


    def on_space(self, chunk: BlockChunk):
        span = self.tail.span
        if not span.chunks:
            return span.append([chunk])
        return span.append([chunk])        
    
    def on_symbol(self, chunk: BlockChunk):
        return self.on_text([chunk])
    
    
    def on_text(self, chunks: list[BlockChunk]):
        span = self.tail.span
        return span.append(chunks)
        
        
    def on_child(self, child: Block):
        return child
    
    
        

    def _attach_child(self, child: Block, insert_after_span: Span | None = None) -> None:
        """Attach a child to this block (set parent and inherit block_text).

        Args:
            child: The child block to attach
            insert_after_span: The span after which to insert the child's spans.
                              If None, appends to the end of the BlockText.
        """
        child.parent = self._block
        # Copy child's spans into parent's BlockText
        if self._block is not None:
            parent_bt = self._block.block_text
            if child.block_text is not parent_bt:
                # Copy spans from the child and its descendants to parent's BlockText
                # and update the block's span reference to the new copy
                # Insert after the specified span (or append if None)
                current_insert_point = insert_after_span
                for block in child.iter_depth_first(all_blocks=True):
                    if block.span is not None:
                        new_span = block.span.copy()
                        if current_insert_point is not None:
                            parent_bt.insert_after(current_insert_point, new_span)
                        else:
                            parent_bt.append(new_span)
                        block.span = new_span
                        current_insert_point = new_span
                # Recursively update block_text on child and all descendants
                self._set_block_text_recursive(child, parent_bt)
            else:
                raise ValueError("Child block already has a block_text")
                child.block_text = parent_bt

    def _set_block_text_recursive(self, block: Block, bt: "BlockText") -> None:
        """Recursively set block_text on a block and all its descendants."""
        block.block_text = bt
        for child in block.children:
            self._set_block_text_recursive(child, bt)
                   


    def append_child(self, child: Block, to_body: bool = True) -> Block:
        """Append a child block to the body."""
        # Find the span to insert after: the last span in the current subtree
        child = self.on_child(child)
        insert_after = self.tail.span
        child = self._append_child_after(child, insert_after, to_body)
        return child
    
    def _append_child_after(self, child: Block, insert_after: Span, to_body: bool = True) -> Block:
        self._attach_child(child, insert_after_span=insert_after)
        if to_body:
            self.body.append(child)
        else:
            self.block.children.append(child)
        return child


    def prepend_child(self, child: Block) -> Mutator:
        """Prepend a child block to the body."""
        # Insert after the parent's own span (before any existing children)
        insert_after = self.block.span
        self._attach_child(child, insert_after_span=insert_after)
        self.body.insert(0, child)
        return self

    def insert_child(self, index: int, child: Block) -> Mutator:
        """Insert a child block at the given index."""
        # Find the span to insert after
        if index == 0:
            # Insert after the parent's own span
            insert_after = self.block.span
        else:
            # Insert after the last span of the preceding sibling
            preceding_sibling = self.body[index - 1]
            insert_after = preceding_sibling.mutator.get_last_span()
        self._attach_child(child, insert_after_span=insert_after)
        self.body.insert(index, child)
        return self

    def remove_child(self, child: Block) -> Mutator:
        """Remove a child block from the body."""
        if child in self.body:
            child.parent = None
            # Note: we don't clear block_text - the span is still owned by it
            self.body.remove(child)
        return self
    
    
    def join(self, sep: BlockChunk): 
        for span in self.block.iter_delimiters():
            span.append([sep])
 
    
    # def get_last_span(self) -> Span:
    #     body = self.body
    #     if not body:
    #         return body.parent.span
    #     return body[-1].mutator.get_last_span()
    
    
    # def get_last_block(self) -> Block:
    #     body = self.body
    #     if not body:
    #         return body.parent
    #     return body[-1].mutator.get_last_block()
    
    # def get_last_appendable_block(self) -> Block:
    #     if not self.body:
    #         if self.head.has_newline():
    #             block = self._spawn_block()
    #             self.append_child(block)
    #             return self.body[-1]
    #         else:
    #             return self.block
    #     if last := self.body[-1]:
    #         if last.has_newline():
    #             block = self._spawn_block()
    #             self.append_child(block)
    #             return self.body[-1]
    #         else:
    #             return last
    
    # def auto_handle_newline(self) -> Block:
    #     """Auto handle newline for the block."""
    #     if last := self.get_last_block():
    #         if not last.has_newline() and not last.is_wrapper:
    #             last.add_newline()
    #     return last
    
    # -------------------------------------------------------------------------
    # Text Operations
    # -------------------------------------------------------------------------

    
    def iter_delimiters(self) -> Iterator[Block]:
        yield self.block
        length = len(self.body)
        for i in range(length):
            if i == length - 1:
                return
            yield self.body[i].mutator.tail

    # -------------------------------------------------------------------------
    # Traversal Operations
    # -------------------------------------------------------------------------



    # -------------------------------------------------------------------------
    # Render (transformation point)
    # -------------------------------------------------------------------------
    
    
    def apply_style(self, style: str, only_views: bool = False, copy: bool = True, recursive: bool = True):
        from .schema import BlockSchema
        if self.block is None:
            raise ValueError("Block is not set")
        block_copy = self.block.copy(copy) if copy else self.block
        styles = parse_style(style)
        
        if recursive:
            for block in block_copy.traverse():
                if only_views and not isinstance(block, BlockSchema):
                    continue
                if block.is_leaf():
                    continue
                block.style.extend(styles)
        else:
            block_copy.style.extend(styles)
        return block_copy
    
    # def call_render(self, block: Block, path: Path) -> Block:
    #     """Render the block and set the mutator."""
        
    #     new_block = self.render(block, path)
    #     block, new_block = self._transfer_metadata(block, new_block)
    #     self.block = new_block
    #     new_block.mutator = self
        
    #     return new_block

    # def render(self, block: Block, path: Path) -> Block:
    #     """
    #     Transform the block structure.

    #     Base implementation returns the block unchanged.
    #     Subclasses override to create wrapper structures.

    #     Returns:
    #         The (possibly transformed) block with this mutator attached.
    #     """
    #     # if not block.head.has_newline():
    #         # block.head.append_postfix([Chunk(content="\n")])
    #     return block
    
    
    def call_render(self, block: Block, path: Path) -> Block:
        """Render the block and set the mutator."""
        
        new_block = self.call_init(block.chunks(), path, tags=block.tags, role=block.role, style=block.style, attrs=block.attrs)
        # block, new_block = self._transfer_metadata(block, new_block)
        for child in block.body:
            new_block.append_child(child)
        new_block.mutator.call_commit()
        self.block = new_block
        new_block.mutator = self
        
        return new_block

    # def render(self, block: Block, path: Path) -> Block:
    #     """
    #     Transform the block structure.

    #     Base implementation returns the block unchanged.
    #     Subclasses override to create wrapper structures.

    #     Returns:
    #         The (possibly transformed) block with this mutator attached.
    #     """
    #     # if not block.head.has_newline():
    #         # block.head.append_postfix([Chunk(content="\n")])
    #     block = self.call_init(block.chunks())
    #     block.mutator.call_commit()
    #     return block
        
    
    def call_extract(self) -> Block:
        # from .mutators import RootMutator
        # if isinstance(self, RootMutator):
        #     return self.block[1].extract()
        # if not self.is_rendered:
            # raise ValueError(f"Block is not rendered. can't extract: {self.block}")
        ex_block = self.extract()
        
        block, ex_block = self._transfer_metadata(self.block, ex_block)
        for child in self.body:
            ex_child = child.mutator.call_extract()
            ex_block.append_child(ex_child)
        return ex_block
    
    def extract(self) -> Block:
        return Block(self.head.span.content, tags=self.block.tags, role=self.block.role, style=self.block.style, attrs=self.block.attrs)

    # -------------------------------------------------------------------------
    # Schema Operations (for BlockSchema support)
    # -------------------------------------------------------------------------
    
    def _transfer_metadata(self, from_block: Block, to_block: Block):
        # ms = set(self.styles)
        # bs = set(self.block.style)
        # if not ms & bs and self.styles:
        #     self.block.style.append(self.styles[0])
        to_block.tags = from_block.tags.copy()
        to_block.role = from_block.role
        to_block.style = from_block.style.copy()
        to_block.attrs = from_block.attrs.copy()
        return from_block, to_block
    
    def _apply_metadata(self, to_block: Block, tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None):
        to_block.tags = tags or to_block.tags
        to_block.role = role or to_block.role
        to_block.style = style or to_block.style
        to_block.attrs = attrs or to_block.attrs
        return to_block
            
    def call_instantiate(self, content: "ContentType | None" = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None) -> Block:
        """
        Call the instantiate method of the block.
        """
        block = self.instantiate(content, role, tags, style)
        # block.mutator = self.__class__(block, did_instantiate=True, did_commit=False)
        # self._transfer_metadata()
        block.mutator = self
        self.block = block
        self.did_init = True
        self.did_commit = False
        
        return block

    def instantiate(self, content: "ContentType | None" = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None, attrs: dict[str, Any] | None = None) -> Block:
        """
        Create a Block instance from a schema.

        Base implementation creates a simple block with the content.
        Subclasses override for style-specific instantiation (e.g., XML structure).

        Args:
            schema: The schema to instantiate from
            content: Content for the new block
            **kwargs: Additional arguments

        Returns:
            A new Block instance
        """
        # Default: create a block with the content (or schema name)
        block = Block(
            content=content,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
            _auto_handle=self.block._auto_handle,
        )

        # Recursively instantiate child schemas
        # for child in schema.children:
        #     if hasattr(child, 'instantiate'):
        #         # Child is a schema - instantiate it
        #         block.append_child(child.instantiate())
        #     else:
        #         # Child is a regular block - copy it
        #         block.append_child(child.copy())

        return block


    
    



class Block:
    """
    Tree node with a head span and children.

    Block = Span (head) + Children (body)

    Blocks reference Spans but don't own them - BlockText owns Spans.
    The Mutator provides indirection for accessing/mutating fields.
    """

    __slots__ = ["span", "children", "parent", "block_text", "role", "tags", "mutator", "_style", "_auto_handle", "attrs", "_schema"]

    def __init__(
        self,
        content: ContentType | None = None,
        *,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
        # Internal: for factory methods
        _span: Span | None = None,
        _children: list["Block"] | None = None,
        _auto_handle: bool = True,
        _schema: "BlockSchema | None" = None,
    ):
        """
        Create a block.

        Args:
            content: Initial content (str, int, float, bool, Chunk, list)
            role: Role identifier for the block
            tags: List of tags for categorization
            style: Style string (space-separated classes)
            mutator: Custom mutator for field access
            block_text: BlockText that owns spans (created if not provided)
        """
        # Import here to avoid circular import
        from .block_text import BlockText

        # Simple attributes
        self.children: BlockChildren = BlockChildren(items=_children, parent=self) if _children is not None else BlockChildren(parent=self)
        self.parent: Block | None = None
        self.role = role
        self.tags = tags or []
        self._style = parse_style(style)
        self._auto_handle = _auto_handle
        self.attrs = attrs or {}
        self._schema = _schema
        # Set up mutator first (needed for promote)
        if mutator is None:
            self.mutator = Mutator(self)
        else:
            self.mutator = mutator
            mutator.block = self

        # Set up BlockText - root blocks create their own, children inherit
        if block_text is None:
            self.block_text = BlockText()
        else:
            self.block_text = block_text

        # Handle span initialization - create via BlockText for proper ownership
        if _span is not None:
            self.span = _span
        else:
            # Create span via BlockText so it's properly owned
            if content is not None:
                # chunks = self.mutator.promote(content)
                self.span = self.block_text.create_span()
                self.append(content)
                # self.span.content = chunks
            else:
                self.span = self.block_text.create_span()

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_span(
        cls,
        span: Span,
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
    ) -> Block:
        """Create a block from an existing span."""
        return cls(_span=span, mutator=mutator, block_text=block_text)

    @classmethod
    def empty(
        cls,
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
    ) -> Block:
        """Create an empty block (with empty span)."""
        return cls(mutator=mutator, block_text=block_text)
    
    
    def commit(self, content: ContentType):
        """
        Commit the block.
        """        
        chunks = self.mutator.promote(content) if content is not None else BlockChunkList(chunks=[])
        # block = self.mutator.commit(chunks)
        # self.mutator.did_commit = True
        block = self.mutator.call_commit(chunks)
        return block

    # -------------------------------------------------------------------------
    # Properties (only where logic is needed)
    # -------------------------------------------------------------------------

    # @property
    # def mutator(self) -> Mutator:
    #     """Get the mutator for this block."""
    #     if self._mutator is None:
    #         self._mutator = Mutator(self)
    #     return self._mutator

    # @mutator.setter
    # def mutator(self, value: Mutator) -> None:
    #     """Set a new mutator for this block."""
    #     self._mutator = value
    #     value.block = self

    @property
    def style(self) -> list[str]:
        """Get style classes."""
        return self._style

    @style.setter
    def style(self, value: str | list[str]) -> None:
        """Set style classes (parses string into list)."""
        self._style = parse_style(value)

    # -------------------------------------------------------------------------
    # Mutated Access (goes through mutator)
    # -------------------------------------------------------------------------

    @property
    def head(self) -> Block:
        """Get head span via mutator."""
        return self.mutator.head

    @property
    def tail(self) -> Block:
        """Get tail block via mutator."""
        return self.mutator.tail
    
    @property
    def body(self) -> BlockChildren:
        """Get children via mutator."""
        return self.mutator.body

    @property
    def content(self) -> str:
        """Get content text via mutator."""
        return self.mutator.content
    
    @property
    def text(self) -> str:
        return self.render()
    
    @property
    def type(self) -> Type | None:
        """Get type of the block."""
        return self._schema._type if self._schema is not None else str
    
    
    @property
    def value(self) -> Any:
        """Get value of the block."""
        return self.get_value()
    
    def chunks(self) -> BlockChunkList:
        return BlockChunkList(chunks=list(self.span.chunks))
        
    def get_value(self):
        from .object_helpers import block_to_object, parse_content
        if self._schema is not None:
            _type = self._schema._type
        else:
            _type = str
        if _type == str:
            return "".join([b.content for b in self.body])
        elif issubclass(_type, BaseModel):
            return block_to_object(self, _type)
        else:
            return parse_content("".join([b.content for b in self.body]), _type)
    
    # -------------------------------------------------------------------------
    # context building
    # -------------------------------------------------------------------------
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __call__(
        self,
        content: ContentType | Block | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "Block":
        if isinstance(content, Block):
            block = content
        else:
            block = Block(content, role=role, tags=tags, style=style, attrs=attrs, _auto_handle=self._auto_handle)
        
        # if self._auto_handle:
        #     curr_head = self.mutator.current_head
        #     if not curr_head.has_newline() and not curr_head.is_empty:
        #         self.mutator.current_head.add_newline()
        
        self.mutator.append_child(block)
        
        return block
    
    
    @classmethod
    def schema_view(cls, name: str | None = None, type: Type | None = None, tags: list[str] | None = None, style: str | None = None, attrs: dict[str, Any] | None = None) -> "BlockSchema":
        from .schema import BlockSchema
        schema_block = BlockSchema(
            name,
            type=type,
            tags=tags,
            attrs=attrs,
            # style=["xml"] if style is None and name is not None else parse_style(style),
            style=style,            
        )
        return schema_block

    
    
    def view(
        self,
        name: str | None = None,
        type: Type | None = None,        
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        is_required: bool = True,
    ) -> "BlockSchema":
        from .schema import BlockSchema
        schema_block = BlockSchema(
            name,
            type=type,
            tags=tags,
            attrs=attrs,
            # style="xml" if style is None and name is not None else style,
            style=style,
            is_required=is_required,
        )
        # if self._auto_handle:
        #     curr_head = self.mutator.current_head
        #     if not curr_head.has_newline() and not curr_head.is_empty:
        #         self.mutator.current_head.add_newline()
        self.mutator.append_child(schema_block)
        return schema_block
    
    
    
    def view_list(
        self,
        item_name: str,
        key: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,        
        style: str | None = None,
        attrs: dict[str, Any] | None = None,
        is_required: bool = True,
    ) -> "BlockListSchema":
        from .schema import BlockListSchema
        schema_block = BlockListSchema(
            item_name=item_name,
            key=key,
            name=name,
            tags=tags,
            attrs=attrs,
            # style=["xml-list"] if style is None else parse_style(style),
            style=style,
            is_required=is_required,
        )
        # if not self.mutator.current_head.has_newline():
            # self.mutator.current_head.add_newline()
        # if self._auto_handle:
        #     curr_head = self.mutator.tail
        #     if not curr_head.has_newline() and not curr_head.is_empty():
        #         self.mutator.tail.add_newline()
        self.mutator.append_child(schema_block)
        return schema_block


    # -------------------------------------------------------------------------
    # Mutation (via mutator)
    # -------------------------------------------------------------------------
    
    # def append(self, content: ContentType) -> Block:
    #     """Append content to the last block"""
    #     chunks = self.mutator.promote(content)

    #     last_idx = 0  # Start at 0, tracks position after last processed separator
    #     for i in range(len(chunks)):
    #         if not chunks[i].is_text:
    #             # Get text chunks before this separator
    #             text_chunks = chunks[last_idx:i]
    #             sep = chunks[i]
    #             last_idx = i + 1  # Move past the separator

    #             if text_chunks:
    #                 self.mutator.on_text(text_chunks)

    #             if sep.is_line_end:
    #                 self.mutator.on_newline(sep)
    #             elif sep.isspace():
    #                 self.mutator.on_space(sep)
    #             else:
    #                 self.mutator.on_symbol(sep)

    #     # Handle remaining text chunks after the last separator
    #     if last_idx < len(chunks):
    #         text_chunks = chunks[last_idx:]
    #         self.mutator.on_text(text_chunks)


    def append(self, content: ContentType, style: str | None = None) -> list[BlockChunkList | Block]:
        """Append content to the last block"""
        chunks = self.mutator.promote(content, style=style)

        last_idx = 0  # Start at 0, tracks position after last processed separator
        events = []
        for i in range(len(chunks)):
            if not chunks[i].is_text:
                # Get text chunks before this separator
                text_chunks = chunks[last_idx:i]
                sep = chunks[i]
                last_idx = i + 1  # Move past the separator

                if text_chunks:
                    res = self.mutator.on_text(text_chunks)
                    events.append(res)
                if sep.is_line_end:
                    res = self.mutator.on_newline(sep)
                    events.append(res)
                elif sep.isspace():
                    res = self.mutator.on_space(sep)
                    events.append(res)
                else:
                    res = self.mutator.on_symbol(sep)
                    events.append(res)

        # Handle remaining text chunks after the last separator
        if last_idx < len(chunks):
            text_chunks = chunks[last_idx:]
            res = self.mutator.on_text(text_chunks)
            events.append(res)
        return events



    # def append(self, content: ContentType, style: str | None = None) -> Block:
    #     """Append to head content."""
    #     chunks = self.mutator.promote(content, style=style)
    #     self.mutator.head.span.append(chunks, style=style)
    #     # self.mutator.append_content(self.mutator.promote(content))
    #     return self

    def prepend(self, content: ContentType, style: str | None = None) -> Block:
        """Prepend to head content."""
        chunks = self.mutator.promote(content, style=style)
        self.mutator.head.span.prepend(chunks, style=style)
        # self.mutator.prepend_content(self.mutator.promote(content))
        return self


    def add_newline(self) -> Block:
        """Add newline to head postfix."""
        self.mutator.head.span.append([BlockChunk(content="\n", style="newline")])
        return self
    
    def has_newline(self) -> bool:
        """Check if the block has end of line."""
        # if self.span is not None:
        #     return self.span.has_end_of_line()
        return self.mutator.head.span.has_newline()
    
    def is_empty(self) -> bool:
        """Check if the block is empty."""
        return self.mutator.head.span.is_empty
    
    def indent(self, spaces: int = 2):        
        if not self.is_wrapper:            
            spaces_chunk = BlockChunk(content=" " * spaces, style="tab")            
            self.mutator.head.prepend([spaces_chunk])
        for child in self.children:
            child.indent(spaces)
        return self
    
    def indent_body(self, spaces: int = 2):
        for child in self.children:
            child.indent(spaces)
        return self
    
    
    # def append_child(self, child: ContentType) -> Block:
    #     """Append child to body."""
    #     block = Block(content=child)
    #     self.mutator.append_child(block)
    #     return self
    
    def append_child(self, child: Block | ContentType) -> Block:
        """Append child to body."""
        if not isinstance(child, Block):
            child = Block(content=child)
        self.mutator.append_child(child)
        return self
    
    # def append_child(self, child: Block | ContentType) -> Block:
    #     """Append child to body."""
    #     self.mutator.auto_handle_newline()
    #     if isinstance(child, Block):
    #         self.mutator.append_child(child)
    #     else:
    #         self.mutator.append_child(Block(content=child))
    #     # self.mutator.auto_handle_newline()
    #     return self
    
    
    
    # def append_child(self, child: Block | ContentType) -> Block:
    #     """Append child to body.""" 
    #     # curr_head = self.mutator.current_head
    #     if self._auto_handle:
    #         curr_head = self.mutator.block_end
    #         if not curr_head.has_newline() and not curr_head.is_empty:
    #             curr_head.add_newline()    
    #     if isinstance(child, Block):
    #         self.mutator.append_child(child)
    #     else:
    #         chunks = self.mutator.promote(child)
    #         block = self.mutator._spawn_block(chunks)
    #         self.mutator.append_child(block)
    #     # self.mutator.auto_handle_newline()
    #     return self


    def prepend_child(self, child: Block) -> Block:
        """Prepend child to body."""
        self.mutator.prepend_child(child)
        return self

    def insert_child(self, index: int, child: Block) -> Block:
        """Insert child at index."""
        self.mutator.insert_child(index, child)
        return self

    def remove_child(self, child: Block) -> Block:
        """Remove child from body."""
        self.mutator.remove_child(child)
        return self
    
    @property
    def is_wrapper(self) -> bool:
        """Check if the block is a wrapper."""
        return self.span.is_empty
    
    # -------------------------------------------------------------------------
    # Text Operations
    # -------------------------------------------------------------------------
    
    
    def join(self, sep: BlockChunk | str = "\n"):
        sep = BlockChunk(content=sep) if isinstance(sep, str) else sep
        self.mutator.join(sep)
        return self
    
    # -------------------------------------------------------------------------
    # Operator Overloading
    # -------------------------------------------------------------------------
    def __itruediv__(self, other: Block | ContentType | tuple):
        # curr_head = self.mutator.current_head        
        # if self._auto_handle and not curr_head.has_newline():
        #     self.append("\n")
        # if self._auto_handle:
            # self.append("\n")
        if isinstance(other, tuple):
            if len(other) == 0:
                return self
            # Join tuple elements with spaces into a single child
            first = other[0]
            if isinstance(first, Block):
                child = first.copy()
            else:
                child = Block(content=first)
            for item in other[1:]:
                if isinstance(item, Block):
                    child.append_child(item.copy())
                else:
                    child.append(item)
            self.append_child(child)
        elif isinstance(other, Block):
            self.append_child(other)
        else:
            # Wrap ContentType in a Block
            self.append_child(other)        
        
        return self
    
    
    def __and__(self, other: ContentType):
        # self.append(other, sep="")
        self_copy = self.copy()
        self_copy.append(self.mutator.promote(other))
        return self_copy
    
    def __rand__(self, other: ContentType):
        self.prepend(other)
        return self
    
    def __iand__(self, other: ContentType):
        self.append(other)
        return self
    
    
    def __len__(self) -> int:
        return len(self.mutator.body)
    
    
    # -------------------------------------------------------------------------
    # Tree Navigation
    # -------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """
        Compute the path for this block in the logical tree.

        Uses mutator.body for navigation, making style wrappers
        (like XML tags) transparent to the path.

        Returns:
            Path representing this block's position
        """
        return compute_path(self)

    def is_leaf(self) -> bool:
        """True if block has no children."""
        return len(self.body) == 0

    def is_root(self) -> bool:
        """True if block has no parent."""
        return self.parent is None

    def depth(self) -> int:
        """Depth in tree (root = 0)."""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d

    def root(self) -> Block:
        """Get root of tree."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node
    
    def count_block_tree(self, all_blocks: bool = False) -> int:
        """Count all blocks in the tree."""
        return 1 + sum(child.count_block_tree(all_blocks=all_blocks) for child in (self.children if all_blocks else self.body))
    
    def max_depth(self, all_blocks: bool = False) -> int:
        """Get the maximum depth of the tree."""
        if not (self.children if all_blocks else self.body):
            return 1
        return 1 + max(child.max_depth(all_blocks=all_blocks) for child in (self.children if all_blocks else self.body))
    
    
    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    
    def get_all(self, tags: str | list[str], children_only: bool = False) -> list["Block"]:
        """
        Get all blocks matching a tag path.

        Supports dot-notation for nested tag searches:
        - "response" - find all blocks with tag "response" in direct children
        - "response.thinking" - find "thinking" blocks that are descendants of "response" children

        Args:
            tags: Single tag, dot-separated path, or list of tags
            children_only: If True, only search direct children for ALL tags (not just first)

        Returns:
            List of matching blocks
        """
        if isinstance(tags, str):
            tags = tags.split(".")

        if not tags:
            return []

        # Find all blocks matching first tag in direct children only
        candidates = [child for child in self.body if tags[0] in child.tags]

        # Filter through remaining tags - search descendants within each candidate
        for tag in tags[1:]:
            next_candidates = []
            for blk in candidates:
                if children_only:
                    # Only search direct children
                    for child in blk.body:
                        if tag in child.tags:
                            next_candidates.append(child)
                else:
                    # Search all descendants
                    for child in blk.traverse():
                        if child is not blk and tag in child.tags:
                            next_candidates.append(child)
            candidates = next_candidates

        return candidates


    def get_one(self, tags: str | list[str]) -> "Block":
        """
        Get the first block matching a tag path.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block

        Raises:
            ValueError: If no matching block found
        """
        result = self.get_all(tags, children_only=True)
        if not result:
            raise ValueError(f'Tag path "{tags}" does not exist')
        return result[0]
    
    
    def get_one_schema(self, tags: str | list[str]) -> "BlockSchema | None":
        """
        Get the first schema block matching a tag path.
        """
        from .schema import BlockSchema
        result = self.get_all(tags, children_only=True)
        if not result:
            raise ValueError(f'Tag path "{tags}" does not exist')
        if not isinstance(result[0], BlockSchema):
            raise ValueError(f"Block {result[0]} is not a BlockSchema")
        return result[0]

    def get_one_or_none(self, tags: str | list[str]) -> "Block | None":
        """
        Get the first block matching a tag path, or None if not found.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block, or None
        """
        result = self.get_all(tags, children_only=True)
        return result[0] if result else None
    
    
    def get_one_schema_or_none(self, tags: str | list[str]) -> "BlockSchema | None":
        """
        Get the first schema block matching a tag path, or None if not found.
        """
        from .schema import BlockSchema
        result = self.get_all(tags, children_only=True)
        if not result:
            return None
        if not isinstance(result[0], BlockSchema):
            return None
        return result[0]
    
    
    
    def get_list(self, tags: str | list[str]) -> "BlockList":
        """
        Get a list of blocks matching a tag path.
        """
        from .schema import BlockList
        result = self.get_all(tags, children_only=True)                
        if not result:
            return BlockList()
        target = result[0]
        if not isinstance(target, BlockList):
            raise ValueError(f"Block {target} is not a BlockList")
        return target
    
    
    def __getitem__(self, index: int) -> Block:
        return self.body[index]


    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------
    
    def __iter__(self):
        return iter(self.body)
    
    
    
    def traverse(self) -> Iterator[Block]:
        yield from self.iter_depth_first()
        
    def iter_delimiters(self) -> Iterator[Block]:
        return self.mutator.iter_delimiters()
            
    

    def iter_depth_first(self, all_blocks: bool = False) -> Iterator[Block]:
        """Iterate blocks depth-first (pre-order)."""
        yield self
        target = self.children if all_blocks else self.body
        for child in target:
            yield from child.iter_depth_first(all_blocks)

    def iter_breadth_first(self, all_blocks: bool = False) -> Iterator[Block]:
        """Iterate blocks breadth-first."""
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            target = node.children if all_blocks else node.body
            queue.extend(target)

    def iter_leaves(self) -> Iterator[Block]:
        """Iterate leaf blocks only."""
        for block in self.iter_depth_first():
            if block.is_leaf():
                yield block

    # -------------------------------------------------------------------------
    # Span Collection
    # -------------------------------------------------------------------------

    def iter_spans(self) -> Iterator[Span]:
        """Iterate all spans in depth-first order."""
        for block in self.iter_depth_first():
            if block.span is not None:
                yield block.span

    def collect_spans(self) -> list[Span]:
        """Collect all spans in depth-first order."""
        return list(self.iter_spans())

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    
    def apply_style(self, style: str, only_views: bool = False, copy: bool = True, recursive: bool = True):
        return self.mutator.apply_style(style, only_views, copy, recursive)
    
    def print_text(self):
        """Print the text of the block."""
        print(self.block_text.text())
    
    
    def transform(self) -> Block:
        """Apply mutator transformation."""
        from .rendering import render
        return render(self)
    
    def extract(self) -> Block:
        """Extract the block."""
        return self.mutator.call_extract()
    
    def render(self) -> str:
        """Render the block."""
        output = self.transform()
        return output.block_text.text_between(output.mutator.head.span, output.mutator.tail.span)
        # return output.block_text.text()
    
    def print(self):
        """Print the transformed block."""
        print(self.render())

    def render_text(self) -> str:
        """Render tree to text (depth-first span concatenation)."""
        parts = []
        for span in self.iter_spans():
            parts.append(span.text)
        return "".join(parts)

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy_head(self) -> Block:
        """
        Copy only the block's span (head) without children.

        Creates a new Block with a copied span but no children.
        The new block gets its own BlockText.
        """
        from .block_text import BlockText

        new_block_text = BlockText()
        new_span = self.span.copy() if self.span else None
        if new_span:
            new_block_text.append(new_span)

        return Block(
            _span=new_span,
            _children=[],
            role=self.role,
            tags=self.tags.copy() if self.tags else [],
            style=self._style.copy() if self._style else [],
            attrs=self.attrs.copy() if self.attrs else {},
            block_text=new_block_text,
        )

    def copy(self, deep: bool = True) -> Block:
        """
        Copy this block.

        Args:
            deep: If True, recursively copy children and spans (new BlockText).
                  If False, shallow copy (shares BlockText and children list).
        """
        from .block_text import BlockText

        if deep:
            # Create new BlockText to own the copied spans
            new_block_text = BlockText()
            new_span = self.span.copy() if self.span else None
            if new_span:
                new_block_text.append(new_span)

            new_block = Block(
                _span=new_span,
                _children=[],
                role=self.role,
                tags=self.tags.copy() if self.tags else [],
                style=self._style.copy() if self._style else [],
                block_text=new_block_text,
            )
            # Deep copy children - they inherit the new block_text
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_block.append_child(child_copy)
            return new_block
        else:
            return Block(
                _span=self.span,
                _children=self.children.copy(),
                role=self.role,
                tags=self.tags,
                style=self._style,
                block_text=self.block_text,
            )
            
    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Schema Extraction
    # -------------------------------------------------------------------------

    def extract_schema(self, style: str | list[str] | None = None) -> "BlockSchema | None":
        """
        Extract a new BlockSchema tree containing only BlockSchema nodes.

        Traverses this block's subtree and creates a new BlockSchema with
        only BlockSchema children, preserving the schema hierarchy while
        filtering out regular Block nodes.

        Returns:
            A new BlockSchema tree with only schema nodes, or None if no schemas found
        """
        from .schema import BlockSchema, BlockListSchema

        # Create a schema from this block
        if isinstance(self, BlockSchema):
            # Already a schema - copy without children
            new_schema = self.copy_head()
            # Ensure xml style is present for rendering
            if style:
                styles = parse_style(style)
                for style in styles:
                    if style not in new_schema._style:
                        new_schema._style.append(style)
            # if "xml" not in new_schema._style:
            #     new_schema._style.append("xml")
        else:
            # Regular block - don't create a schema, just collect children
            new_schema = None

        # Collect schema children
        schema_children = []
        for child in self.children:
            if isinstance(child, (BlockSchema, BlockListSchema)):
                child_schema = child.extract_schema(style=style)
                if child_schema is not None:
                    schema_children.append(child_schema)
            else:
                # For regular blocks, search their children for nested schemas
                self._collect_nested_schemas(child, schema_children, style=style)

        if new_schema is not None:
            # Add collected children to the schema
            for child_schema in schema_children:
                new_schema.append_child(child_schema)
            return new_schema
        else:
            # No parent schema - return based on number of children found
            if len(schema_children) == 0:
                return None
            elif len(schema_children) == 1:
                return schema_children[0]
            else:
                # Multiple schemas at root level - create a virtual wrapper
                wrapper_schema = BlockSchema(
                    style=["root"],  # No style - won't render tags
                )
                # Clear the span to make it a true wrapper (no content)
                wrapper_schema.span = self.block_text.create_span()
                for child_schema in schema_children:
                    wrapper_schema.append_child(child_schema)
                return wrapper_schema

    def _collect_nested_schemas(self, block: "Block", result: list, style: str | list[str] | None = None) -> None:
        """
        Recursively search a Block's children for BlockSchema nodes.

        Collects found BlockSchema nodes into the result list.
        """
        from .schema import BlockSchema, BlockListSchema

        for child in block.children:
            if isinstance(child, (BlockSchema, BlockListSchema)):
                child_schema = child.extract_schema(style=style)
                if child_schema is not None:
                    result.append(child_schema)
            else:
                # Keep searching deeper
                self._collect_nested_schemas(child, result, style=style)
                
    # -------------------------------------------------------------------------
    # Pydantic model support
    # -------------------------------------------------------------------------
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, Block):
            return v
        elif isinstance(v, dict):
            # if "_type" in v and v["_type"] == "Block":
            #     return Block.model_validate(v)
            return Block.model_validate(v)
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")


    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        if self.role:
            parts.append(f"role={self.role!r}")
        if self.span:
            text = self.span.content.text[:20]
            if len(self.span.content.text) > 20:
                text += "..."
            if text:
                parts.append(f"content={text!r}")
            postfix_text = self.span.postfix.text[:20]
            if len(self.span.postfix.text) > 20:
                postfix_text += "..."
            if postfix_text:
                parts.append(f"postfix={postfix_text!r}")
            prefix_text = self.span.prefix.text[:20]
            if len(self.span.prefix.text) > 20:
                prefix_text += "..."
            if prefix_text:
                parts.append(f"prefix={prefix_text!r}")
            if self.tags:
                parts.append(f"tags={self.tags!r}")
            if self.style:
                parts.append(f"style={self.style!r}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        return f"Block({', '.join(parts)})"

    # def debug(self, indent: int = 0) -> str:
    #     """Debug tree representation showing prefix, content, postfix."""
    #     ind = "  " * indent
    #     parts = []  
        
    #     parts.append(f"path={self.path.indices_str()}")      

    #     if self.role:
    #         parts.append(f"role={self.role!r}")
            


    #     if self.span:
    #         prefix_text = self.span.prefix_text
    #         content_text = self.span.content_text
    #         postfix_text = self.span.postfix_text

    #         if prefix_text:
    #             parts.append(f"prefix={prefix_text!r}")
    #         if content_text:
    #             # Truncate long content
    #             if len(content_text) > 30:
    #                 content_text = content_text[:30] + "..."
    #             parts.append(f"content={content_text!r}")
    #         if postfix_text:
    #             parts.append(f"postfix={postfix_text!r}")
        
        
                
    #     if self.tags:
    #         parts.append(f"tags={self.tags!r}")
                
    #     if self.style:
    #         parts.append(f"style={self.style!r}")

    #     if self.children:
    #         parts.append(f"children={len(self.children)}")
            
    #     if self.mutator.is_rendered:
    #         parts.append(f"rendered")
            
    #     cls_name = self.__class__.__name__
    #     lines = [f"{ind}{cls_name}({', '.join(parts)})"]
    #     for child in self.children:
    #         lines.append(child.debug(indent + 1))
    #     return "\n".join(lines)
    def debug(self, indent: int = 0, spans: bool = False) -> str:
        """Debug tree representation showing prefix, content, postfix."""
        ind = "  " * indent
        parts = []  
        path = self.path.indices_str()
        if path:
            parts.append(f"{self.path.indices_str()}")      

        if self.role:
            parts.append(f"role={self.role!r}")
            


        if self.span:
            prefix_text = self.span.prefix.text
            content_text = self.span.content.text
            postfix_text = self.span.postfix.text
            span_parts = []

            if prefix_text:
                span_parts.append(f"{prefix_text}|")
            if content_text:
                # Truncate long content
                if len(content_text) > 30:
                    content_text = content_text[:30] + "..."
                span_parts.append(f"{content_text}")
            if postfix_text:
                span_parts.append(f"|{postfix_text}")
                
            parts.append(f"{''.join(span_parts)!r}")
        
        
        
                
        if self.tags:
            parts.append(f"{self.tags!r}")
                
        if self.style:
            parts.append(f"style={' '.join(self.style)!r}")

        if self.children:
            parts.append(f"children={len(self.children)}")
        
        
        if spans:  
            parts.append(f"span={self.span!r}")
            # if self.span.prev is not None:
            #     parts.append(f"prev={self.span.prev.id}")
            # if self.span.next is not None:
            #     parts.append(f"next={self.span.next.id}")
                
        if self.mutator.is_rendered:
            parts.append(f"rendered")
            
        cls_name = self.__class__.__name__
        lines = [f"{ind}{cls_name}({', '.join(parts)})"]
        for child in self.children:
            lines.append(child.debug(indent + 1, spans=spans))
        return "\n".join(lines)
    
    
    def print_debug(self, spliter: str | None = None, spans: bool = False):
        print(self.debug(spans=spans))
        if spliter:
            print(spliter * 100)
    
    def get_block_texts_ids(self) -> set[int]:
        s = set()
        for block in self.iter_depth_first():
            s.add(id(block.block_text))
        return s

    def print_block_texts_ids(self, all_blocks: bool = False):
        for block in self.iter_depth_first(all_blocks):
            print(id(block.block_text))

    # -------------------------------------------------------------------------
    # Serialization (model_dump / model_validate)
    # -------------------------------------------------------------------------

    def model_dump(
        self,
        *,
        mode: str | None = None,
        include_chunks: bool = True,
        include_span: bool = True,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """
        Serialize block tree to a JSON-compatible dict.

        Args:
            include_chunks: Include chunk-level detail with logprobs
            include_span: Include full span structure (prefix/content/postfix)
            exclude_none: Exclude None values from output

        Returns:
            Dict representation suitable for JSON serialization

        Example:
            block.model_dump()
            # {"content": "hello", "role": "user", "children": [...]}

            block.model_dump(include_chunks=True)
            # {"content": "hello", "chunks": [{"content": "hello", "logprob": -0.1}], ...}
        """
        result: dict[str, Any] = {}

        # Content (always include as simple string)
        # result["content"] = self.content

        # Core fields
        if self.role is not None or not exclude_none:
            result["role"] = self.role
        if self.tags:
            result["tags"] = self.tags
        if self._style:
            result["style"] = self._style
        if self.attrs:
            result["attrs"] = self.attrs

        # Span detail (optional)
        # if include_span or include_chunks:
            # span_data: dict[str, Any] = {}
            # if self.span:
            # span_data["prefix"] = self.span.prefix_text
            # span_data["content"] = self.span.content_text
            # span_data["postfix"] = self.span.postfix_text

            # if include_chunks:
            #     span_data["prefix_chunks"] = [
            #         {"content": c.content, "logprob": c.logprob}
            #         for c in self.span.prefix
            #     ]
            #     span_data["content_chunks"] = [
            #         {"content": c.content, "logprob": c.logprob}
            #         for c in self.span.content
            #     ]
            #     span_data["postfix_chunks"] = [
            #         {"content": c.content, "logprob": c.logprob}
            #         for c in self.span.postfix
            #     ]
            # result["span"] = span_data
        result["span"] = {}
        result["span"]["prefix"] = [
            {"content": c.content, "logprob": c.logprob}
            for c in self.span.prefix
        ]
        result["span"]["content"] = [
            {"content": c.content, "logprob": c.logprob}
            for c in self.span.chunks
        ]
        result["span"]["postfix"] = [
            {"content": c.content, "logprob": c.logprob}
            for c in self.span.postfix
        ]


        # Children (recursive)
        if self.children:
            result["children"] = [
                child.model_dump(
                    include_chunks=include_chunks,
                    include_span=include_span,
                    exclude_none=exclude_none,
                )
                for child in self.children
            ]
        else:
            result["children"] = []

        return result

    @classmethod
    def model_validate(
        cls,
        data: dict[str, Any],
        *,
        block_text: "BlockText | None" = None,
    ) -> "Block":
        """
        Deserialize a dict to a Block tree.

        Args:
            data: Dict from model_dump() or JSON
            block_text: Shared BlockText for all blocks (created if None)

        Returns:
            Reconstructed Block tree

        Example:
            data = {"content": "hello", "role": "user", "children": []}
            block = Block.model_validate(data)
        """
        from .block_text import BlockText
        from .schema import BlockSchema

        # Create shared BlockText if not provided
        if block_text is None:
            block_text = BlockText()

        # Check if this is a BlockSchema (has 'name' field)
        if data.get("name") is not None:
            return BlockSchema.model_validate(data, block_text=block_text)

        # Extract fields
        content = data.get("content")
        role = data.get("role")
        tags = data.get("tags")
        style = data.get("style")
        attrs = data.get("attrs")

        # Handle span with chunks if present
        span = None
        span_data = data.get("span")
        if span_data and (span_data.get("content_chunks") or span_data.get("prefix_chunks")):
            # Reconstruct span from chunks
            from .span import BlockChunk
            prefix_chunks = [
                BlockChunk(content=c["content"], logprob=c.get("logprob"))
                for c in span_data.get("prefix_chunks", [])
            ]
            content_chunks = [
                BlockChunk(content=c["content"], logprob=c.get("logprob"))
                for c in span_data.get("content_chunks", [])
            ]
            postfix_chunks = [
                BlockChunk(content=c["content"], logprob=c.get("logprob"))
                for c in span_data.get("postfix_chunks", [])
            ]
            span = Span(prefix=prefix_chunks, chunks=content_chunks, postfix=postfix_chunks)
            block_text.append(span)
            content = None  # Content comes from span

        # Create block
        block = cls(
            content=content,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
            block_text=block_text,
            _span=span,
        )

        # If we reconstructed span, set it
        if span is not None:
            block.span = span

        # Recursively load children
        for child_data in data.get("children", []):
            child = cls.model_validate(child_data, block_text=block_text)
            child.parent = block
            block.children.append(child)

        return block
