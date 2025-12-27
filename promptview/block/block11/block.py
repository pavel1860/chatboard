from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Callable, Type

from .span import Span, Chunk
from .mutator_meta import MutatorMeta
from .path import Path, compute_path
if TYPE_CHECKING:
    from .block_text import BlockText
    from .schema import BlockSchema, BlockListSchema




def parse_style(style: str | list[str] | None) -> list[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []

BaseContentTypes = str | bool | int | float
ContentType = BaseContentTypes | list[str] | list[Chunk] | Chunk





class Mutator(metaclass=MutatorMeta):
    """
    Strategy for accessing and mutating block fields.

    Mutators provide indirection between blocks and their data,
    allowing transformations (like XML wrapping) to redirect
    field access to virtual locations.

    Base Mutator provides direct access (no transformation).
    Subclasses override accessors to redirect to different locations.
    """
    styles = []

    def __init__(self, block: Block | None = None, did_instantiate: bool = True, did_commit: bool = True):
        self._block: Block | None = block
        self._did_instantiate: bool = did_instantiate
        self._did_commit: bool = False
        self._is_rendered: bool = False

    @property
    def block(self) -> Block:
        if self._block is None:
            raise ValueError("Block is not set")
        return self._block

    @block.setter
    def block(self, value: Block | None) -> None:
        self._block = value
        
    
    
    @property
    def head(self) -> Span:
        return self.block.span
    
    @property
    def body(self) -> list[Block]:
        return self.block.children
    
    @property
    def content(self) -> str:
        return self.block.span.content_text
    
    
    @property
    def current_head(self) -> Span:
        if not self.body:
            return self.head
        # return self.body[-1].mutator.current_head
        return self.body[-1].mutator.block_end
    
    @property
    def block_end(self) -> Span:
        if not self.body:
            return self.head
        return self.body[-1].mutator.block_end
    
    

    def is_head_closed(self) -> bool:
        return self.head.has_newline()
    
    def is_head_open(self, chunks: list[Chunk]) -> bool:
        return not self.head.has_newline()
    
    def is_last_block_open(self,  chunks: list[Chunk]) -> bool:
        if not self.body:
            return False
        elif self.body[-1].has_newline():
            return False
        else:
            return True
   
   
    def get_appendable_target(self) -> Block | None:
        if not self.body:
            if self.head.has_newline():
                return None
            else:
                return self.block
        if last := self.body[-1]:
            if last.has_newline():
                return None
            else:
                return last
        return None
        
    
    def _spawn_block(self, chunks: list[Chunk] | None = None) -> Block:
        block = Block(chunks, block_text=self.block.block_text)        
        return block
        
    def init(self, chunks: list[Chunk], tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None):
        block = Block(chunks, tags=tags, role=role, style=style)        
        return block
    
    def call_init(self, chunks: list[Chunk], tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None):
        block = self.init(chunks, tags, role, style)
        block.mutator = self
        self.block = block
        return block
    
    def commit(self, chunks: list[Chunk]):
        return self.on_text(chunks)
        
    # def on_newline(self, chunk: Chunk):
    #     if self.body:
    #         # if not self.body[-1].head.has_end_of_line():
    #         self.body[-1].head.append_postfix([chunk])
    #     elif not self.head.has_newline():
    #         self.head.append_postfix([chunk])
    #     else:
    #         block = self._spawn_block([chunk])
    #         self.append_child(block)
    
    def on_newline(self, chunk: Chunk):
        if self.is_head_open([chunk]):
            self.head.append_postfix([chunk])
        elif self.is_last_block_open([chunk]):
            self.body[-1].head.append_postfix([chunk])
        else:
            block = self._spawn_block([chunk])
            self.append_child(block)
    
    def on_space(self, chunk: Chunk):
        return self.on_text([chunk])
    
    def on_symbol(self, chunk: Chunk):
        return self.on_text([chunk])
    
    
    # def on_text2(self, chunks: list[Chunk]):
    #     if len(self.body) > 0:
    #         if self.body[-1].has_newline():
    #             block = self._spawn_block()
    #             self.append_child(block)
    #         self.body[-1].head.append_content(chunks)
    #         # elif self.block.dfs(post=True, stop=lambda x: x.has_end_of_line and x.is_leaf())
            
    #         return
    #     if self.head.has_newline():
    #         block = self._spawn_block(chunks)
    #         self.append_child(block)
    #         return
    #     else:
    #         self.head.append_content(chunks)
        
            
            
    def on_text(self, chunks: list[Chunk]):
        if self.is_head_open(chunks):
            self.head.append_content(chunks)
        elif self.is_last_block_open(chunks):
            self.body[-1].head.append_content(chunks)
        else:
            block = self._spawn_block(chunks)
            self.append_child(block)
        
    
    
        
    
    

    # -------------------------------------------------------------------------
    # Accessor Methods (can be overridden for transformation)
    # -------------------------------------------------------------------------

  
    # def get_content(self) -> str:
    #     """Get the content text of the head span."""
    #     span = self.get_head()
    #     if span is None:
    #         return ""
    #     return span.content_text

    # -------------------------------------------------------------------------
    # Mutation Methods
    # -------------------------------------------------------------------------

    # def append_content(self, chunks: list[Chunk]) -> Mutator:
    #     """Append chunks to the head span's content."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.append_content(chunks)
    #     return self

    # def prepend_content(self, chunks: list[Chunk]) -> Mutator:
    #     """Prepend chunks to the head span's content."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.prepend_content(chunks)
    #     return self

    # def append_prefix(self, chunks: list[Chunk]) -> Mutator:
    #     """Append chunks to the head span's prefix."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.append_prefix(chunks)
    #     return self

    # def prepend_prefix(self, chunks: list[Chunk]) -> Mutator:
    #     """Prepend chunks to the head span's prefix."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.prepend_prefix(chunks)
    #     return self

    # def append_postfix(self, chunks: list[Chunk]) -> Mutator:
    #     """Append chunks to the head span's postfix."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.append_postfix(chunks)
    #     return self

    # def prepend_postfix(self, chunks: list[Chunk]) -> Mutator:
    #     """Prepend chunks to the head span's postfix."""
    #     span = self.get_head()
    #     if span is not None:
    #         span.prepend_postfix(chunks)
    #     return self

    # -------------------------------------------------------------------------
    # Content Promotion
    # -------------------------------------------------------------------------

    def promote(self, content: ContentType) -> list[Chunk]:
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
        if isinstance(content, str):
            return [Chunk(content=content)]
        elif isinstance(content, bool):
            return [Chunk(content=str(content).lower())]
        elif isinstance(content, (int, float)):
            return [Chunk(content=str(content))]
        elif isinstance(content, Chunk):
            return [content]
        elif isinstance(content, list):
            if len(content) == 0:
                return []
            if isinstance(content[0], str):
                return [Chunk(content=s) for s in content]
            elif isinstance(content[0], Chunk):
                return content
        return []

    def _attach_child(self, child: Block) -> None:
        """Attach a child to this block (set parent and inherit block_text)."""
        child.parent = self._block
        # Copy child's spans into parent's BlockText
        if self._block is not None:
            parent_bt = self._block.block_text
            if child.block_text is not parent_bt:
                # Copy spans from the child and its descendants to parent's BlockText
                # and update the block's span reference to the new copy
                for block in child.iter_depth_first(all_blocks=True):
                    if block.span is not None:
                        new_span = block.span.copy()
                        parent_bt.append(new_span)
                        block.span = new_span
                # Recursively update block_text on child and all descendants
                self._set_block_text_recursive(child, parent_bt)
            else:
                child.block_text = parent_bt

    def _set_block_text_recursive(self, block: Block, bt: "BlockText") -> None:
        """Recursively set block_text on a block and all its descendants."""
        block.block_text = bt
        for child in block.children:
            self._set_block_text_recursive(child, bt)
            
            
    
            
            
    # def append(self, content: ContentType) -> Mutator:
    #     """Append content to the last block"""
    #     target_chunks = []
    #     target_block = self.get_last_appendable_block()
    #     chunks = self.promote(content)
    #     # def append_chunks(target_block: "Block | None", chunks: list[Chunk]) -> "Block | None":
    #     #     if target_block is None:
    #     #         target_block = self.append_child(chunks, add_new_line=False)
    #     #     else:
    #     #         target_block._inline_append(chunks)
    #     #     return target_block
        
    #     for i, chunk in enumerate(chunks):
    #         if chunk.is_line_end:
    #             if target_chunks:
    #                 target_block = target_block.mutator.append_content(target_chunks)
    #                 target_block.mutator.append_postfix([chunk])
    #             else:
    #                 target_block.mutator.append_postfix([chunk])
    #             target_chunks = []
    #             if i < len(chunks) - 1:
    #                 target_block = self.get_last_appendable_block()
    #         else:
    #             target_chunks.append(chunk)
    #     if target_chunks:
    #         target_block.mutator.append_content(target_chunks)
    #     return self

    def append_child(self, child: Block) -> Mutator:
        """Append a child block to the body."""        
        self._attach_child(child)
        self.body.append(child)
        return self

    def prepend_child(self, child: Block) -> Mutator:
        """Prepend a child block to the body."""
        self._attach_child(child)
        self.body.insert(0, child)
        return self

    def insert_child(self, index: int, child: Block) -> Mutator:
        """Insert a child block at the given index."""
        self._attach_child(child)
        self.body.insert(index, child)
        return self

    def remove_child(self, child: Block) -> Mutator:
        """Remove a child block from the body."""
        if child in self.body:
            child.parent = None
            # Note: we don't clear block_text - the span is still owned by it
            self.body.remove(child)
        return self
    
    
    def get_last_span(self) -> Span:
        if not self.body:
            return self.block.span
        return self.body[-1].mutator.get_last_span()
    
    
    def get_last_block(self) -> Block:
        if not self.body:
            return self.block
        return self.body[-1].mutator.get_last_block()
    
    def get_last_appendable_block(self) -> Block:
        if not self.body:
            if self.head.has_newline():
                block = self._spawn_block()
                self.append_child(block)
                return self.body[-1]
            else:
                return self.block
        if last := self.body[-1]:
            if last.has_newline():
                block = self._spawn_block()
                self.append_child(block)
                return self.body[-1]
            else:
                return last
    
    def auto_handle_newline(self) -> Block:
        """Auto handle newline for the block."""
        if last := self.get_last_block():
            if not last.has_newline() and not last.is_wrapper:
                last.add_newline()
        return last


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
    
    def render_and_set(self, block: Block, path: Path) -> Block:
        """Render the block and set the mutator."""
        self._is_rendered = True
        block = self.render(block, path)
        self.block = block
        block.mutator = self
        return block

    def render(self, block: Block, path: Path) -> Block:
        """
        Transform the block structure.

        Base implementation returns the block unchanged.
        Subclasses override to create wrapper structures.

        Returns:
            The (possibly transformed) block with this mutator attached.
        """
        if not block.head.has_newline():
            block.head.append_postfix([Chunk(content="\n")])
        return block

    # -------------------------------------------------------------------------
    # Schema Operations (for BlockSchema support)
    # -------------------------------------------------------------------------
    
    def call_instantiate(self, content: "ContentType | None" = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None) -> Block:
        """
        Call the instantiate method of the block.
        """
        block = self.instantiate(content, role, tags, style)
        # block.mutator = self.__class__(block, did_instantiate=True, did_commit=False)
        block.mutator = self
        self.block = block
        self._did_instantiate = True
        self._did_commit = False
        return block

    def instantiate(self, content: "ContentType | None" = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None) -> Block:
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

    # def commit(self, content: ContentType) -> Block:
    #     """
    #     Validate and finalize a block against a schema.

    #     Base implementation returns the block unchanged.
    #     Subclasses override for validation logic.

    #     Args:
    #         block: The block to validate
    #         schema: The schema to validate against

    #     Returns:
    #         The validated block
    #     """
    #     return self.block
    
    
    def call_commit(self, block: Block) -> Block:
        """
        Call the commit method of the block.
        """
        block = self.commit(block)
        block.mutator._did_commit = True
        block.mutator._did_instantiate = True
        return block


class Block:
    """
    Tree node with a head span and children.

    Block = Span (head) + Children (body)

    Blocks reference Spans but don't own them - BlockText owns Spans.
    The Mutator provides indirection for accessing/mutating fields.
    """

    __slots__ = ["span", "children", "parent", "block_text", "role", "tags", "mutator", "_style"]

    def __init__(
        self,
        content: ContentType | None = None,
        *,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
        # Internal: for factory methods
        _span: Span | None = None,
        _children: list["Block"] | None = None,
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
        self.children: list[Block] = _children if _children is not None else []
        self.parent: Block | None = None
        self.role = role
        self.tags = tags or []
        self._style = parse_style(style)

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
                chunks = self.mutator.promote(content)
                self.span = self.block_text.create_span()
                self.span.content = chunks
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
        chunks = self.mutator.promote(content) if content is not None else []
        self.mutator.commit(chunks)
        return self

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
    def head(self) -> Span:
        """Get head span via mutator."""
        return self.mutator.head

    @property
    def body(self) -> list[Block]:
        """Get children via mutator."""
        return self.mutator.body

    @property
    def content(self) -> str:
        """Get content text via mutator."""
        return self.mutator.content
    
    
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
    ) -> "Block":
        if isinstance(content, Block):
            block = content
        else:
            block = Block(content, role=role, tags=tags, style=style)        
        # self.mutator.auto_handle_newline()
        curr_head = self.mutator.current_head
        if not curr_head.has_newline() and not curr_head.is_empty:
            self.mutator.current_head.add_newline()
        self.head.has_newline()
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
        curr_head = self.mutator.current_head
        if not curr_head.has_newline() and not curr_head.is_empty:
            self.mutator.current_head.add_newline()
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
        curr_head = self.mutator.current_head
        if not curr_head.has_newline() and not curr_head.is_empty:
            self.mutator.current_head.add_newline()
        self.mutator.append_child(schema_block)
        return schema_block


    # -------------------------------------------------------------------------
    # Mutation (via mutator)
    # -------------------------------------------------------------------------
    
    def append(self, content: ContentType) -> Block:
        """Append content to the last block"""        
        chunks = self.mutator.promote(content)
        
        # prefix, item, postfix = next(((chunks[:i], chunks[i], chunks[i+1:]) for i in range(len(chunks)) if not chunks[i].is_text))
        text_chunks = []
        # last_idx = 0
        last_idx = -1
        for i in range(len(chunks)):
            if not chunks[i].is_text:
                text_chunks = chunks[last_idx:i]
                sep = chunks[i]
                last_idx = i
                if text_chunks:
                    self.mutator.on_text(text_chunks)
                    
                if sep.is_line_end:
                    self.mutator.on_newline(sep)
                elif sep.isspace():
                    self.mutator.on_space(sep)
                else:
                    self.mutator.on_symbol(sep)
        else:
            # if last_idx < len(chunks):
            if last_idx < len(chunks) - 1:
                last_idx = max(last_idx, 0)
                text_chunks = chunks[last_idx:]
                self.mutator.on_text(text_chunks)



    def append_content(self, content: ContentType) -> Block:
        """Append to head content."""
        chunks = self.mutator.promote(content)
        self.mutator.head.append_content(chunks)
        # self.mutator.append_content(self.mutator.promote(content))
        return self

    def prepend_content(self, content: ContentType) -> Block:
        """Prepend to head content."""
        chunks = self.mutator.promote(content)
        self.mutator.head.prepend_content(chunks)
        # self.mutator.prepend_content(self.mutator.promote(content))
        return self

    def append_prefix(self, content: ContentType) -> Block:
        """Append to head prefix."""
        chunks = self.mutator.promote(content)
        self.mutator.head.append_prefix(chunks)
        # self.mutator.append_prefix(self.mutator.promote(content))
        return self

    def prepend_prefix(self, content: ContentType) -> Block:
        """Prepend to head prefix."""
        chunks = self.mutator.promote(content)
        self.mutator.head.prepend_prefix(chunks)        
        # self.mutator.prepend_prefix(self.mutator.promote(content))
        return self

    def append_postfix(self, content: ContentType) -> Block:
        """Append to head postfix."""
        chunks = self.mutator.promote(content)
        self.mutator.head.append_postfix(chunks)
        # self.mutator.append_postfix(self.mutator.promote(content))
        return self

    def prepend_postfix(self, content: ContentType) -> Block:
        """Prepend to head postfix."""
        chunks = self.mutator.promote(content)
        self.mutator.head.prepend_postfix(chunks)
        # self.mutator.prepend_postfix(self.mutator.promote(content))
        return self

    def add_newline(self) -> Block:
        """Add newline to head postfix."""
        self.head.append_postfix([Chunk(content="\n")])
        return self
    
    def has_newline(self) -> bool:
        """Check if the block has end of line."""
        # if self.span is not None:
        #     return self.span.has_end_of_line()
        return self.head.has_newline()
    
    
    
    def indent(self, spaces: int = 2):        
        if not self.is_wrapper:            
            spaces_chunk = Chunk(content=" " * spaces)            
            self.mutator.head.prepend_prefix([spaces_chunk])
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
    
    # def append_child(self, child: Block) -> Block:
    #     """Append child to body."""
    #     self.mutator.append_child(child)
    #     return self
    
    # def append_child(self, child: Block | ContentType) -> Block:
    #     """Append child to body."""
    #     self.mutator.auto_handle_newline()
    #     if isinstance(child, Block):
    #         self.mutator.append_child(child)
    #     else:
    #         self.mutator.append_child(Block(content=child))
    #     # self.mutator.auto_handle_newline()
    #     return self
    def append_child(self, child: Block | ContentType) -> Block:
        """Append child to body.""" 
        # curr_head = self.mutator.current_head
        curr_head = self.mutator.block_end
        if not curr_head.has_newline() and not curr_head.is_empty:
            curr_head.add_newline()    
        if isinstance(child, Block):
            self.mutator.append_child(child)
        else:
            chunks = self.mutator.promote(child)
            block = self.mutator._spawn_block(chunks)
            self.mutator.append_child(block)
        # self.mutator.auto_handle_newline()
        return self


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
    # Operator Overloading
    # -------------------------------------------------------------------------
    def __itruediv__(self, other: Block | ContentType | tuple):
        curr_head = self.mutator.current_head        
        if not curr_head.has_newline():
            self.append("\n")
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
                    child.append_content(item)
            self.append_child(child)
        elif isinstance(other, Block):
            self.append_child(other)
        else:
            # Wrap ContentType in a Block
            self.append(other)        
        
        return self
    
    
    def __and__(self, other: ContentType):
        # self.append(other, sep="")
        self_copy = self.copy()
        self_copy.append_content(self.mutator.promote(other))
        return self_copy
    
    def __rand__(self, other: ContentType):
        self.prepend_content(other)
        return self
    
    def __iand__(self, other: ContentType):
        self.append_content(other)
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
    
    
    def __getitem__(self, index: int) -> Block:
        return self.body[index]


    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------
    
    def __iter__(self):
        return iter(self.body)
    
    
    
    def traverse(self) -> Iterator[Block]:
        yield from self.iter_depth_first()

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
    
    
    def render(self) -> Block:
        """Render the block."""
        output = self.transform()
        return output.block_text.text()
    
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
                    style=[],  # No style - won't render tags
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
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        if self.role:
            parts.append(f"role={self.role!r}")
        if self.span:
            text = self.span.content_text[:20]
            if len(self.span.content_text) > 20:
                text += "..."
            if text:
                parts.append(f"content={text!r}")
            postfix_text = self.span.postfix_text[:20]
            if len(self.span.postfix_text) > 20:
                postfix_text += "..."
            if postfix_text:
                parts.append(f"postfix={postfix_text!r}")
            prefix_text = self.span.prefix_text[:20]
            if len(self.span.prefix_text) > 20:
                prefix_text += "..."
            if prefix_text:
                parts.append(f"prefix={prefix_text!r}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        return f"Block({', '.join(parts)})"

    def debug(self, indent: int = 0) -> str:
        """Debug tree representation showing prefix, content, postfix."""
        ind = "  " * indent
        parts = []

        if self.role:
            parts.append(f"role={self.role!r}")

        if self.span:
            prefix_text = self.span.prefix_text
            content_text = self.span.content_text
            postfix_text = self.span.postfix_text

            if prefix_text:
                parts.append(f"prefix={prefix_text!r}")
            if content_text:
                # Truncate long content
                if len(content_text) > 30:
                    content_text = content_text[:30] + "..."
                parts.append(f"content={content_text!r}")
            if postfix_text:
                parts.append(f"postfix={postfix_text!r}")
                
        if self.style:
            parts.append(f"style={self.style!r}")

        if self.children:
            parts.append(f"children={len(self.children)}")
        cls_name = self.__class__.__name__
        lines = [f"{ind}{cls_name}({', '.join(parts)})"]
        for child in self.children:
            lines.append(child.debug(indent + 1))
        return "\n".join(lines)
    
    
    def print_debug(self, spliter: str | None = None):
        print(self.debug())
        if spliter:
            print(spliter * 100)
    
    def get_block_texts_ids(self) -> set[int]:
        s = set()
        for block in self.iter_depth_first():
            s.add(id(block.block_text))
        return s
            
    def print_block_texts(self):
        for block in self.iter_depth_first():
            print(id(block.block_text))  
