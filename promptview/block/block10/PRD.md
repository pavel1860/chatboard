# Block10 - Unified Block System PRD

## Overview

Block10 is a redesigned block system that unifies user-defined prompts and parsed LLM responses through a clean separation of storage (BlockText) from structure (Block). The key innovation is using a linked-list chunk storage with span-based views, enabling robust editing without span invalidation.

## Problem Statement

The current block9 system has fundamental architectural issues:

1. **Chunk Boundary Mismatch**: LLM returns arbitrary chunk boundaries (e.g., `">I"` containing both closing tag and content), but the tree structure assumes clean boundaries.

2. **Style Identity Crisis**: For user prompts, styles are metadata applied at render time. For parsed LLM output, styles (XML tags) ARE the content. The system conflates these.

3. **Editing Fragility**: Tree-based storage makes editing complex - inserting content requires updating spans across the tree.

4. **Multiple Representations**: BlockBuilderContext, StreamingBlockBuilder, and various renderers (renderers.py, renderers2.py, renderers3.py) create confusion.

## Goals

1. **Unified Model**: Single architecture that handles both user-defined prompts and parsed LLM responses
2. **Robust Editing**: Insert, delete, append operations that don't break span references
3. **Preserve LLM Metadata**: Original chunks with logprobs remain intact
4. **Clean Separation**: Storage (chunks) separate from structure (blocks) separate from presentation (styles)
5. **Parser-Renderer Unification**: Schema drives both parsing and rendering through unified Style definitions

## Non-Goals

1. Collaborative editing / CRDT support (single-user for now)
2. Full undo/redo system (can be added later)
3. Backward compatibility with block9 API (clean break)

## Architecture

### Core Data Flow

```
Chunk (atomic text unit)
    ↓
BlockText (linked list storage, owns all chunks)
    ↓
VirtualBlockText (span-based view into BlockText)
    ↓
Block (tree structure, styles, tags)
    ↓
Style (unified parse + render logic)
```

### Key Design Decisions

1. **Linked List for Chunks**: O(1) insertion, no index-based span invalidation
2. **Direct Chunk References in Spans**: Spans hold chunk references, not indices
3. **Chunk-Aligned Parsing**: Parser emits chunk boundaries aligned with block boundaries
4. **Style as Parse+Render**: Each style knows both how to parse and render its format

## Data Structures

### Chunk

```python
@dataclass
class Chunk:
    """Atomic unit of text, immutable content"""
    id: str                          # Stable UUID
    content: str                     # Text content
    logprob: float | None = None     # LLM log probability
    prev: "Chunk | None" = None      # Linked list prev
    next: "Chunk | None" = None      # Linked list next
```

### BlockText

```python
class BlockText:
    """Flat, linked-list storage for all chunks"""
    head: Chunk | None
    tail: Chunk | None
    _by_id: dict[str, Chunk]         # Fast lookup by ID

    # Operations
    def append(chunk: Chunk) -> None
    def insert_after(after: Chunk, new: Chunk) -> None
    def split_chunk(chunk: Chunk, offset: int) -> tuple[Chunk, Chunk]
    def __iter__() -> Iterator[Chunk]
    def fork() -> "BlockText"        # Copy for editing
```

### SpanAnchor & Span

```python
@dataclass
class SpanAnchor:
    """Position within a chunk"""
    chunk: Chunk                     # Direct reference
    offset: int                      # Byte offset within chunk

@dataclass
class Span:
    """A contiguous region of text"""
    start: SpanAnchor
    end: SpanAnchor

    def text() -> str                # Materialize content
    def chunks() -> Iterator[Chunk]  # Yield covered chunks
```

### VirtualBlockText

```python
class VirtualBlockText:
    """View into BlockText via one or more spans"""
    source: BlockText
    spans: list[Span]                # Can be discontiguous

    def render() -> str              # Concatenate span contents
    def append(content: str) -> None # Append chunk, extend spans
    def insert(offset: int, content: str) -> None
    def find_position(offset: int) -> tuple[Chunk, int]
```

### Block

```python
@dataclass
class Block:
    """Tree node with structure and style"""
    # Content
    text: VirtualBlockText | None = None

    # For parsed blocks: spans to actual tag chunks
    prefix_span: Span | None = None
    postfix_span: Span | None = None

    # Metadata
    tags: list[str]
    attrs: dict[str, Any]
    styles: list[str]                # Style names for rendering

    # Tree structure
    children: list["Block"]
    parent: "Block | None" = None

    # Source reference
    _source: BlockText | None = None

    # Properties
    @property
    def is_parsed(self) -> bool      # Has prefix/postfix spans

    # Operations
    def append_text(content: str) -> None
    def prepend_text(content: str) -> None
    def insert_text(offset: int, content: str) -> None
    def render() -> str
```

### BlockSchema

```python
@dataclass
class BlockSchema:
    """Template for block structure"""
    name: str
    tag: str
    style: str                       # "xml", "markdown", "code", etc.
    type: Type | None = None         # For value parsing
    attrs: dict[str, AttrSchema]
    children: list["BlockSchema"]

    def instantiate() -> Block       # Create empty block from schema
    def get_style() -> "Style"       # Get unified style handler
```

### Style (Unified Parser + Renderer)

```python
class Style(ABC):
    """Unified definition for parsing and rendering a format"""
    name: str
    split_on_newlines: bool = True

    # Parsing
    @abstractmethod
    def parse_prefix(self, cursor: "ChunkCursor") -> tuple[Span, dict[str, str]]:
        """Find opening boundary, return span and attributes"""

    @abstractmethod
    def parse_postfix(self, cursor: "ChunkCursor", tag: str) -> Span:
        """Find closing boundary"""

    @abstractmethod
    def is_content_boundary(self, cursor: "ChunkCursor") -> bool:
        """Check if cursor is at a content boundary (e.g., newline)"""

    # Rendering
    @abstractmethod
    def render_prefix(self, block: Block) -> str:
        """Generate opening markup"""

    @abstractmethod
    def render_postfix(self, block: Block) -> str:
        """Generate closing markup"""
```

## Styles

### XMLStyle

```python
class XMLStyle(Style):
    name = "xml"
    split_on_newlines = True

    # Parses: <tagname attr="value">content</tagname>
    # Renders: <{tag} {attrs}>{content}</{tag}>
```

### MarkdownStyle

```python
class MarkdownStyle(Style):
    name = "markdown"
    split_on_newlines = True

    # Parses: # Heading or ## Subheading
    # Renders: {"#" * depth} {content}\n
```

### CodeStyle

```python
class CodeStyle(Style):
    name = "code"
    split_on_newlines = False        # Preserve newlines in code

    # Parses: ```language\ncode\n```
    # Renders: ```{language}\n{content}\n```
```

### PlainStyle

```python
class PlainStyle(Style):
    name = "plain"
    split_on_newlines = True

    # No prefix/postfix, just content
```

## Parsing

### ChunkCursor

```python
class ChunkCursor:
    """Tracks position during parsing"""
    text: BlockText
    current_chunk: Chunk | None
    offset: int                      # Within current chunk

    def advance(n: int = 1) -> None
    def peek(n: int = 1) -> str
    def consume_until(pattern: str) -> Span
    def position() -> SpanAnchor
    def at_end() -> bool
```

### SchemaParser

```python
class SchemaParser:
    """Parses BlockText into Block tree using schema"""

    def __init__(self, schema: BlockSchema, text: BlockText):
        self.schema = schema
        self.text = text
        self.cursor = ChunkCursor(text)

    def parse(self) -> Block:
        """Parse and return root block"""
        style = self.schema.get_style()

        # Parse prefix (e.g., <response>)
        prefix_span, attrs = style.parse_prefix(self.cursor)

        # Create block
        block = Block(
            tags=[self.schema.tag],
            styles=[self.schema.style],
            attrs=attrs,
            prefix_span=prefix_span,
            _source=self.text,
        )

        # Parse content and children
        self._parse_content(block, style)

        # Parse postfix (e.g., </response>)
        block.postfix_span = style.parse_postfix(self.cursor, self.schema.tag)

        return block

    def _parse_content(self, block: Block, style: Style):
        """Parse content, creating children for nested tags or newlines"""
        ...
```

## Rendering

### Block.render()

```python
def render(self) -> str:
    result = []

    # Prefix: from span (parsed) or computed (user-defined)
    if self.prefix_span:
        result.append(self.prefix_span.text())
    elif self.styles:
        style = get_style(self.styles[0])
        result.append(style.render_prefix(self))

    # Content
    if self.text:
        result.append(self.text.render())

    # Children
    for child in self.children:
        result.append(child.render())

    # Postfix
    if self.postfix_span:
        result.append(self.postfix_span.text())
    elif self.styles:
        style = get_style(self.styles[0])
        result.append(style.render_postfix(self))

    return "".join(result)
```

## Editing Operations

### Append Text

```python
def append_text(self, content: str):
    """Append text to end of this block's content"""
    new_chunk = Chunk(content=content)

    if self.text and self.text.spans:
        # Insert after last chunk of last span
        last_span = self.text.spans[-1]
        self._source.insert_after(last_span.end.chunk, new_chunk)
    else:
        # No existing content, append to source
        self._source.append(new_chunk)

    # Add new span
    new_span = Span(
        start=SpanAnchor(new_chunk, 0),
        end=SpanAnchor(new_chunk, len(content))
    )

    if self.text is None:
        self.text = VirtualBlockText(self._source, [new_span])
    else:
        self.text.spans.append(new_span)
```

### Insert Text

```python
def insert_text(self, offset: int, content: str):
    """Insert text at offset within this block's content"""
    chunk, chunk_offset = self.text.find_position(offset)

    if chunk_offset == 0:
        # Insert before chunk
        new_chunk = Chunk(content=content)
        self._source.insert_before(chunk, new_chunk)
        # Update spans...
    elif chunk_offset == len(chunk.content):
        # Insert after chunk (same as append at this position)
        new_chunk = Chunk(content=content)
        self._source.insert_after(chunk, new_chunk)
        # Update spans...
    else:
        # Split chunk, insert in middle
        left, right = self._source.split_chunk(chunk, chunk_offset)
        new_chunk = Chunk(content=content)
        self._source.insert_after(left, new_chunk)
        self._update_spans_after_split(chunk, chunk_offset, left, right)
```

## Implementation Phases

### Phase 1: Core Data Structures (Foundation)
- [ ] Chunk class with linked list pointers
- [ ] BlockText with append, insert_after, iteration
- [ ] SpanAnchor and Span classes
- [ ] VirtualBlockText with render()
- [ ] Unit tests for all above

### Phase 2: Block Basics
- [ ] Block class with text, tags, styles, attrs
- [ ] Block tree structure (children, parent)
- [ ] Block.render() basic implementation
- [ ] BlockSchema class
- [ ] Block instantiation from schema
- [ ] Unit tests

### Phase 3: Editing Operations
- [ ] Block.append_text()
- [ ] Block.prepend_text()
- [ ] BlockText.split_chunk()
- [ ] Block.insert_text()
- [ ] Span update logic after splits
- [ ] Unit tests for editing scenarios

### Phase 4: Style System
- [ ] Style abstract base class
- [ ] XMLStyle (parse + render)
- [ ] PlainStyle
- [ ] Style registry
- [ ] Unit tests

### Phase 5: Parsing
- [ ] ChunkCursor class
- [ ] SchemaParser basic structure
- [ ] XML tag parsing (prefix, postfix)
- [ ] Content parsing with newline splitting
- [ ] Nested block parsing
- [ ] Integration tests with LLM-like chunks

### Phase 6: Advanced Styles
- [ ] MarkdownStyle
- [ ] CodeStyle (no newline split)
- [ ] Attribute parsing in XMLStyle
- [ ] Style composition

### Phase 7: Integration
- [ ] Parser integration with FBP (fbp_process.py)
- [ ] Block serialization (model_dump, model_validate)
- [ ] BlockText fork() for copy-on-write
- [ ] End-to-end tests

### Phase 8: Migration Helpers
- [ ] block9 to block10 conversion utilities
- [ ] Deprecation warnings in block9
- [ ] Documentation

## File Structure

```
promptview/block/block10/
├── __init__.py           # Public exports
├── PRD.md                # This document
├── chunk.py              # Chunk, BlockText
├── span.py               # SpanAnchor, Span, VirtualBlockText
├── block.py              # Block, BlockSchema
├── styles/
│   ├── __init__.py       # Style base, registry
│   ├── xml.py            # XMLStyle
│   ├── markdown.py       # MarkdownStyle
│   ├── code.py           # CodeStyle
│   └── plain.py          # PlainStyle
├── parser.py             # ChunkCursor, SchemaParser
└── tests/
    ├── test_chunk.py
    ├── test_span.py
    ├── test_block.py
    ├── test_styles.py
    ├── test_parser.py
    └── test_integration.py
```

## Success Criteria

1. **Parsing Robustness**: Can parse LLM chunks like `">I"` without corruption
2. **Edit Stability**: 100 random edits don't break any spans
3. **Logprob Preservation**: Original chunk logprobs accessible after parsing
4. **Round-Trip**: Parse then render produces semantically equivalent output
5. **Performance**: Parsing 10KB of XML in <10ms
6. **API Clarity**: Single way to do each operation

## Open Questions

1. **Whitespace Handling**: Should leading/trailing whitespace in content be trimmed or preserved?
2. **Error Recovery**: How to handle malformed XML from LLM?
3. **Streaming**: Should parser support incremental chunk arrival?
4. **Memory**: For very large texts, should we support chunk eviction?

## Appendix: Migration from block9

### Key Differences

| block9 | block10 |
|--------|---------|
| Tree stores chunks | Flat BlockText stores chunks |
| Spans use indices | Spans use chunk references |
| Multiple renderers | Unified Style class |
| Separate parse/render | Style handles both |
| BlockSent intermediate | No intermediate, direct Chunk→Block |

### Migration Path

1. New code uses block10
2. block9 remains for existing code
3. Conversion utilities for interop
4. Gradual migration of dependent code
5. Deprecate block9 after migration complete
