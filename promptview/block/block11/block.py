from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Callable, Type

from .span import Span, Chunk
from .mutator_meta import MutatorMeta
if TYPE_CHECKING:
    from .block_text import BlockText
    from .schema import BlockSchema




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

    @property
    def block(self) -> Block | None:
        return self._block

    @block.setter
    def block(self, value: Block | None) -> None:
        self._block = value

    # -------------------------------------------------------------------------
    # Accessor Methods (can be overridden for transformation)
    # -------------------------------------------------------------------------

    def get_head(self) -> Span | None:
        """Get the span representing this block's head."""
        if self._block is None:
            return None
        return self._block.span

    def get_body(self) -> list[Block]:
        """Get the children (body) of this block."""
        if self._block is None:
            return []
        return self._block.children

    def get_content(self) -> str:
        """Get the content text of the head span."""
        span = self.get_head()
        if span is None:
            return ""
        return span.content_text

    # -------------------------------------------------------------------------
    # Mutation Methods
    # -------------------------------------------------------------------------

    def append_content(self, chunks: list[Chunk]) -> Mutator:
        """Append chunks to the head span's content."""
        span = self.get_head()
        if span is not None:
            span.append_content(chunks)
        return self

    def prepend_content(self, chunks: list[Chunk]) -> Mutator:
        """Prepend chunks to the head span's content."""
        span = self.get_head()
        if span is not None:
            span.prepend_content(chunks)
        return self

    def append_prefix(self, chunks: list[Chunk]) -> Mutator:
        """Append chunks to the head span's prefix."""
        span = self.get_head()
        if span is not None:
            span.append_prefix(chunks)
        return self

    def prepend_prefix(self, chunks: list[Chunk]) -> Mutator:
        """Prepend chunks to the head span's prefix."""
        span = self.get_head()
        if span is not None:
            span.prepend_prefix(chunks)
        return self

    def append_postfix(self, chunks: list[Chunk]) -> Mutator:
        """Append chunks to the head span's postfix."""
        span = self.get_head()
        if span is not None:
            span.append_postfix(chunks)
        return self

    def prepend_postfix(self, chunks: list[Chunk]) -> Mutator:
        """Prepend chunks to the head span's postfix."""
        span = self.get_head()
        if span is not None:
            span.prepend_postfix(chunks)
        return self

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
                for block in child.iter_depth_first():
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
            
            
    
            
            
    def append(self, content: ContentType) -> Mutator:
        """Append content to the last block"""
        target_chunks = []
        target_block = self.get_last_appendable_block()
        chunks = self.promote(content)
        # def append_chunks(target_block: "Block | None", chunks: list[Chunk]) -> "Block | None":
        #     if target_block is None:
        #         target_block = self.append_child(chunks, add_new_line=False)
        #     else:
        #         target_block._inline_append(chunks)
        #     return target_block
        
        for i, chunk in enumerate(chunks):
            if chunk.is_line_end:
                if target_chunks:
                    target_block = target_block.mutator.append_content(target_chunks)
                    target_block.mutator.append_postfix([chunk])
                else:
                    target_block.mutator.append_postfix([chunk])
                target_chunks = []
                if i < len(chunks) - 1:
                    target_block = self.get_last_appendable_block()
            else:
                target_chunks.append(chunk)
        if target_chunks:
            target_block.mutator.append_content(target_chunks)
        return self

    def append_child(self, child: Block) -> Mutator:
        """Append a child block to the body."""
        body = self.get_body()
        self._attach_child(child)
        body.append(child)
        return self

    def prepend_child(self, child: Block) -> Mutator:
        """Prepend a child block to the body."""
        body = self.get_body()
        self._attach_child(child)
        body.insert(0, child)
        return self

    def insert_child(self, index: int, child: Block) -> Mutator:
        """Insert a child block at the given index."""
        body = self.get_body()
        self._attach_child(child)
        body.insert(index, child)
        return self

    def remove_child(self, child: Block) -> Mutator:
        """Remove a child block from the body."""
        body = self.get_body()
        if child in body:
            child.parent = None
            # Note: we don't clear block_text - the span is still owned by it
            body.remove(child)
        return self
    
    
    def get_last_span(self) -> Span:
        if len(self.get_body()) == 0:
            return self.block.span
        return self.get_body()[-1].mutator.get_last_span()
    
    
    def get_last_block(self) -> Block:
        if len(self.get_body()) == 0:
            return self.block
        return self.get_body()[-1].mutator.get_last_block()
    
    def get_last_appendable_block(self) -> Block:
        if len(self.get_body()) == 0:
            if self.get_head().has_end_of_line():
                self.append_child(Block())
                return self.get_body()[-1]
            else:
                return self.block
        if last := self.get_body()[-1]:
            if last.has_end_of_line():
                self.append_child(Block())
                return self.get_body()[-1]
            else:
                return last
    
    def _auto_handle_newline(self) -> Block:
        """Auto handle newline for the block."""
        if last := self.get_last_block():
            if not last.has_end_of_line() and not last.is_wrapper:
                last.add_newline()
        return last


    # -------------------------------------------------------------------------
    # Render (transformation point)
    # -------------------------------------------------------------------------
    
    def render_and_set(self, block: Block) -> Block:
        """Render the block and set the mutator."""
        block = self.render(block)
        self.block = block
        return block

    def render(self, block: Block) -> Block:
        """
        Transform the block structure.

        Base implementation returns the block unchanged.
        Subclasses override to create wrapper structures.

        Returns:
            The (possibly transformed) block with this mutator attached.
        """
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

    def commit(self, content: ContentType) -> Block:
        """
        Validate and finalize a block against a schema.

        Base implementation returns the block unchanged.
        Subclasses override for validation logic.

        Args:
            block: The block to validate
            schema: The schema to validate against

        Returns:
            The validated block
        """
        return self.block
    
    
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
    def head(self) -> Span | None:
        """Get head span via mutator."""
        return self.mutator.get_head()

    @property
    def body(self) -> list[Block]:
        """Get children via mutator."""
        return self.mutator.get_body()

    @property
    def content(self) -> str:
        """Get content text via mutator."""
        return self.mutator.get_content()
    
    
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
        self.mutator._auto_handle_newline()
        self.append_child(block)
        
        return block
    
    
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
            style="xml" if style is None and name is not None else style,
            is_required=is_required,
        )
        self.append_child(schema_block)
        return schema_block


    # -------------------------------------------------------------------------
    # Mutation (via mutator)
    # -------------------------------------------------------------------------
    
    def append(self, content: ContentType) -> Block:
        """Append content to the last block"""
        self.mutator.append(content)
        return self

    def append_content(self, content: ContentType) -> Block:
        """Append to head content."""
        self.mutator.append_content(self.mutator.promote(content))
        return self

    def prepend_content(self, content: ContentType) -> Block:
        """Prepend to head content."""
        self.mutator.prepend_content(self.mutator.promote(content))
        return self

    def append_prefix(self, content: ContentType) -> Block:
        """Append to head prefix."""
        self.mutator.append_prefix(self.mutator.promote(content))
        return self

    def prepend_prefix(self, content: ContentType) -> Block:
        """Prepend to head prefix."""
        self.mutator.prepend_prefix(self.mutator.promote(content))
        return self

    def append_postfix(self, content: ContentType) -> Block:
        """Append to head postfix."""
        self.mutator.append_postfix(self.mutator.promote(content))
        return self

    def prepend_postfix(self, content: ContentType) -> Block:
        """Prepend to head postfix."""
        self.mutator.prepend_postfix(self.mutator.promote(content))
        return self

    def add_newline(self) -> Block:
        """Add newline to head postfix."""
        self.mutator.append_postfix([Chunk(content="\n")])
        return self
    
    def has_end_of_line(self) -> bool:
        """Check if the block has end of line."""
        if self.span is not None:
            return self.span.has_end_of_line()
        return False
    
    
    # def append_child(self, child: ContentType) -> Block:
    #     """Append child to body."""
    #     block = Block(content=child)
    #     self.mutator.append_child(block)
    #     return self
    
    # def append_child(self, child: Block) -> Block:
    #     """Append child to body."""
    #     self.mutator.append_child(child)
    #     return self
    def append_child(self, child: Block | ContentType) -> Block:
        """Append child to body."""
        if isinstance(child, Block):
            self.mutator.append_child(child)
        else:
            self.mutator.append_child(Block(content=child))
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
        self.mutator._auto_handle_newline()
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
            self.append_child(Block(content=other))        
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
    # -------------------------------------------------------------------------
    # Tree Navigation
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------
    
    def traverse(self) -> Iterator[Block]:
        yield from self.iter_depth_first()

    def iter_depth_first(self) -> Iterator[Block]:
        """Iterate blocks depth-first (pre-order)."""
        yield self
        for child in self.body:
            yield from child.iter_depth_first()

    def iter_breadth_first(self) -> Iterator[Block]:
        """Iterate blocks breadth-first."""
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.body)

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
            
            
    def print_block_texts(self):
        for block in self.iter_depth_first():
            print(id(block.block_text))  
