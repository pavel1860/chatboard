# Block12 Architecture PRD

## Overview

Block12 is a simplified block system that eliminates redundant abstractions while maintaining full functionality for LLM streaming, parsing, and block tree manipulation.

### Key Simplifications

| Block11 | Block12 |
|---------|---------|
| Block + Span + BlockChunkList + BlockChunk + BlockText | Block + ChunkMeta |
| 5 classes | 2 classes |
| Span chain (linked list) | Block tree only |
| Absolute chunk positions | Relative chunk positions |
| BlockText owns string | Root block owns string |

### Design Principles

1. **Single string storage** - The entire document is one string, owned by the root block
2. **Block tree is the structure** - No separate span chain; tree ordering is sufficient
3. **Relative chunk positions** - Chunk metadata uses positions relative to block start (no shifting on insert)
4. **Native string operations** - Find, replace, regex work directly on the string

---

## Architecture

### Core Classes

```
Block (tree node + positions + chunk metadata)
  │
  ├── Tree: parent, children
  ├── Position: start, end (absolute in root._text)
  ├── Chunks: list[ChunkMeta] (relative positions)
  ├── Metadata: role, tags, style, attrs
  └── Text: _text (only on root)

ChunkMeta (lightweight metadata)
  │
  ├── start, end (relative to owning block)
  ├── logprob
  └── style
```

### String Ownership

```
Root Block
├── _text: "Full document content here..."
├── start: 0
├── end: 28
└── children:
    ├── Child1 (start: 5, end: 12)
    │   └── references root._text[5:12]
    └── Child2 (start: 12, end: 28)
        └── references root._text[12:28]
```

---

## Data Structures

### ChunkMeta

```python
@dataclass
class ChunkMeta:
    """Metadata for a chunk of text within a block."""
    start: int                      # Start position relative to block.start
    end: int                        # End position relative to block.start
    logprob: float | None = None    # LLM logprob if available
    style: str | None = None        # Style label (e.g., "xml-tag", "content")
    id: str = field(default_factory=lambda: uuid4().hex[:8])

    @property
    def length(self) -> int:
        return self.end - self.start
```

### Block

```python
@dataclass
class Block:
    """Tree node with text positions and chunk metadata."""

    # --- Tree Structure ---
    parent: Block | None = None
    children: list[Block] = field(default_factory=list)

    # --- Position in shared string (absolute) ---
    start: int = 0
    end: int = 0

    # --- Chunk metadata (relative positions) ---
    chunks: list[ChunkMeta] = field(default_factory=list)

    # --- Block metadata ---
    role: str | None = None
    tags: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    # --- Shared string (only root) ---
    _text: str = ""

    # --- Properties ---

    @property
    def root(self) -> Block:
        """Get the root block of this tree."""
        node = self
        while node.parent:
            node = node.parent
        return node

    @property
    def text(self) -> str:
        """Get this block's text content."""
        return self.root._text[self.start:self.end]

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def depth(self) -> int:
        """Depth in tree (0 for root)."""
        d = 0
        node = self
        while node.parent:
            d += 1
            node = node.parent
        return d
```

---

## Core Operations

### 1. Text Access

```python
@property
def text(self) -> str:
    """Get this block's full text."""
    return self.root._text[self.start:self.end]

def get_chunk_text(self, chunk: ChunkMeta) -> str:
    """Get text for a specific chunk."""
    abs_start = self.start + chunk.start
    abs_end = self.start + chunk.end
    return self.root._text[abs_start:abs_end]
```

### 2. Streaming Append

```python
def append(self, content: str, logprob: float | None = None, style: str | None = None) -> ChunkMeta:
    """Append content to this block (for streaming)."""
    root = self.root

    # Create chunk metadata with relative position
    rel_start = self.end - self.start
    rel_end = rel_start + len(content)
    chunk = ChunkMeta(start=rel_start, end=rel_end, logprob=logprob, style=style)
    self.chunks.append(chunk)

    # Insert into string
    insert_pos = self.end
    root._text = root._text[:insert_pos] + content + root._text[insert_pos:]

    # Update this block's end
    self.end += len(content)

    # Shift all blocks after insertion point
    self._shift_positions_after(insert_pos, len(content))

    return chunk
```

### 3. Position Shifting

```python
def _shift_positions_after(self, position: int, delta: int) -> None:
    """Shift start/end of all blocks after position."""
    for block in self.root._iter_all_blocks():
        if block is self:
            continue
        if block.start >= position:
            block.start += delta
        if block.end > position:
            block.end += delta

def _iter_all_blocks(self) -> Iterator[Block]:
    """Iterate all blocks in tree (depth-first)."""
    yield self
    for child in self.children:
        yield from child._iter_all_blocks()
```

### 4. Child Operations

```python
def append_child(self, child: Block) -> Block:
    """Append a child block."""
    # Determine insertion position
    if self.children:
        insert_pos = self.children[-1].end
    else:
        insert_pos = self.end  # After this block's content

    # If child has its own text, merge it
    if child._text:
        child_text = child._text
        root = self.root

        # Insert child's text
        root._text = root._text[:insert_pos] + child_text + root._text[insert_pos:]

        # Calculate offset for remapping child positions
        offset = insert_pos - child.start

        # Remap child and descendants
        self._remap_subtree(child, offset)

        # Clear child's local text (now using root's)
        child._text = ""

        # Shift blocks after insertion
        self._shift_positions_after(insert_pos, len(child_text))

    # Add to tree
    child.parent = self
    self.children.append(child)

    return child

def _remap_subtree(self, block: Block, offset: int) -> None:
    """Remap positions in a subtree by offset."""
    block.start += offset
    block.end += offset
    for child in block.children:
        self._remap_subtree(child, offset)
```

### 5. Tree Traversal

```python
def iter_depth_first(self) -> Iterator[Block]:
    """Iterate in depth-first order (text order)."""
    yield self
    for child in self.children:
        yield from child.iter_depth_first()

def iter_ancestors(self) -> Iterator[Block]:
    """Iterate from this block to root."""
    node = self
    while node:
        yield node
        node = node.parent

def next_sibling(self) -> Block | None:
    """Get next sibling or None."""
    if not self.parent:
        return None
    siblings = self.parent.children
    idx = siblings.index(self)
    return siblings[idx + 1] if idx + 1 < len(siblings) else None
```

### 6. String Operations

```python
def find(self, pattern: str) -> int:
    """Find pattern in this block's text. Returns relative position or -1."""
    return self.text.find(pattern)

def find_all(self, pattern: str) -> list[int]:
    """Find all occurrences of pattern. Returns relative positions."""
    text = self.text
    positions = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def regex_find(self, pattern: str) -> list[re.Match]:
    """Find all regex matches in this block's text."""
    import re
    return list(re.finditer(pattern, self.text))
```

---

## Implementation Steps

### Phase 1: Core Data Structures

**Step 1.1: Create ChunkMeta class**
- File: `promptview/block/block12/chunk.py`
- Simple dataclass with start, end, logprob, style, id
- No linked list, no owner references

**Step 1.2: Create Block class skeleton**
- File: `promptview/block/block12/block.py`
- Tree fields: parent, children
- Position fields: start, end
- Metadata fields: role, tags, style, attrs, chunks
- Root text field: _text

**Step 1.3: Implement basic properties**
- `root` - walk to root
- `text` - slice from root._text
- `is_root`, `depth`

### Phase 2: Raw Text Operations

**Step 2.1: Implement _raw_append**
- Low-level append without mutator interception
- Add content to root._text
- Create ChunkMeta with relative position and style
- Shift positions of blocks after insertion

**Step 2.2: Implement _raw_prepend**
- Low-level prepend
- Shift existing chunks in block

**Step 2.3: Implement _raw_insert**
- Low-level insert at specific position
- Used by mutators for style-aware insertion

**Step 2.4: Implement _shift_positions_after**
- Iterate all blocks in tree
- Shift start/end for blocks after position

### Phase 3: Mutator System

**Step 3.1: Create MutatorMeta registry**
- File: `promptview/block/block12/mutator.py`
- Metaclass that registers mutators by style
- `get_mutator(style)` lookup

**Step 3.2: Create base Mutator class**
- `head`, `body`, `content` properties
- `append`, `prepend`, `append_child` methods
- Default behavior (delegates to _raw operations)

**Step 3.3: Integrate mutator with Block**
- `block.mutator` property with lazy initialization
- Public API methods delegate to mutator

**Step 3.4: Implement XmlMutator**
- `init(tag_name)` - create opening tag
- `commit(tag_name)` - create closing tag
- `append` - insert before postfix

**Step 3.5: Implement RootMutator**
- For parser root blocks
- Handle prefix/content/postfix structure

### Phase 4: Tree Operations

**Step 4.1: Implement _raw_append_child**
- Low-level child append
- Merge child text if present
- Remap child subtree positions

**Step 4.2: Implement _raw_prepend_child**
- Insert at start of children

**Step 4.3: Implement _raw_insert_child(index)**
- Insert at specific position

**Step 4.4: Implement remove_child**
- Remove from children
- Optionally remove text from root._text

### Phase 5: Query Operations

**Step 5.1: Implement traversal methods**
- iter_depth_first
- iter_ancestors
- next_sibling, prev_sibling

**Step 5.2: Implement search methods**
- get_by_tag, get_all_by_tag
- find, find_all, regex_find

**Step 5.3: Implement chunk queries**
- get_chunk_at_position
- get_chunks_by_style
- get_logprob_at_position

### Phase 6: Block Creation Helpers

**Step 6.1: Implement context manager**
```python
with Block("Parent") as parent:
    with parent("Child1") as child1:
        pass
    with parent("Child2") as child2:
        pass
```

**Step 6.2: Implement factory methods**
- Block.from_string(text)
- Block.from_chunks(chunks)

### Phase 7: Serialization

**Step 7.1: Implement model_dump**
- Serialize block tree to dict
- Include positions, chunks, metadata

**Step 7.2: Implement model_load**
- Deserialize from dict
- Reconstruct tree with correct positions

### Phase 8: Parser Integration

**Step 8.1: Create XmlParser adapter**
- Interface for XmlParser to work with Block12
- Map parsing events to mutator operations

**Step 8.2: Create streaming handler**
- Handle streaming chunks from LLM
- Route to appropriate mutator

### Phase 9: Testing

**Step 9.1: Unit tests for core operations**
- Raw append, insert, remove
- Position shifting correctness
- Chunk relative positions

**Step 9.2: Mutator tests**
- Default mutator behavior
- XmlMutator init/commit/append
- Style-based region queries

**Step 9.3: Integration tests**
- Streaming simulation
- Parse and render round-trip
- Tree manipulation scenarios

---

## File Structure

```
promptview/block/block12/
├── __init__.py
├── chunk.py          # ChunkMeta dataclass
├── block.py          # Block class (data + raw operations)
├── mutator.py        # MutatorMeta, Mutator base class
├── mutators.py       # XmlMutator, RootMutator, MarkdownMutator
├── parser.py         # Parser adapter (XmlParser integration)
└── helpers.py        # Factory methods, utilities

__tests__/block/block12/
├── test_chunk.py
├── test_block.py
├── test_mutator.py
├── test_xml_mutator.py
├── test_tree.py
└── test_integration.py
```

---

## Migration Notes

### From Block11 to Block12

| Block11 | Block12 |
|---------|---------|
| `block.span.content` | `block.text` |
| `block.block_text.text()` | `block.root._text` |
| `span.next` / `span.prev` | Tree traversal |
| `BlockChunkList.filter()` | List comprehension on chunks |
| `Span.prefix` / `Span.postfix` | Chunk style labels + filtering |

### Preserved Concepts

- Block tree structure (parent/children)
- Tags, role, style, attrs metadata
- Chunk logprob tracking
- Depth-first text ordering

### Removed Concepts

- Span linked list
- BlockText as separate class
- BlockChunkList wrapper
- Absolute chunk positions

---

## Success Criteria

1. **Simplicity**: 2 core classes instead of 5
2. **Performance**: O(blocks) insertion, O(1) text access
3. **Functionality**: All block11 use cases supported
4. **Testability**: Comprehensive unit tests
5. **String operations**: Native find/replace/regex work

---

## Mutator Integration

### Overview

Mutators intercept block operations and apply style-aware logic. They control WHERE content gets placed based on the block's structure.

```
User calls: block.append("content")
     ↓
Block delegates: block.mutator.append("content")
     ↓
Mutator decides: insert into "content" region (between prefix and postfix)
```

### Chunk Styles as Regions

Instead of separate position fields, chunks have style labels that define regions:

```python
# XML block chunks:
[
    ChunkMeta(start=0, end=5, style="prefix"),      # "<tag>"
    ChunkMeta(start=5, end=12, style="content"),    # "content"
    ChunkMeta(start=12, end=18, style="postfix"),   # "</tag>"
]
```

Mutator queries chunks by style to find regions:

```python
class Mutator:
    def get_content_region(self) -> tuple[int, int]:
        content_chunks = self.block.get_chunks_by_style("content")
        if content_chunks:
            return content_chunks[0].start, content_chunks[-1].end
        return self.block.length, self.block.length
```

### Mutator Base Class

```python
class Mutator:
    """Base mutator - default behavior."""
    styles = ["default"]

    def __init__(self, block: Block):
        self.block = block

    # --- Region Access ---

    @property
    def head(self) -> str:
        """Get head content (prefix + content for default)."""
        return self.block.text

    @property
    def body(self) -> list[Block]:
        """Get body blocks."""
        return self.block.children

    @property
    def content(self) -> str:
        """Get content region text."""
        chunks = self.block.get_chunks_by_style("content")
        if chunks:
            return self.block.text[chunks[0].start:chunks[-1].end]
        return self.block.text

    # --- Operations (intercepted) ---

    def append(self, text: str, logprob: float = None) -> ChunkMeta:
        """Append to content region."""
        # Default: append to end
        return self.block._raw_append(text, logprob=logprob, style="content")

    def append_prefix(self, text: str) -> ChunkMeta:
        """Append to prefix region."""
        return self.block._raw_prepend(text, style="prefix")

    def append_postfix(self, text: str) -> ChunkMeta:
        """Append to postfix region."""
        return self.block._raw_append(text, style="postfix")

    def append_child(self, child: Block) -> Block:
        """Append a child block."""
        return self.block._raw_append_child(child)
```

### XmlMutator Example

```python
class XmlMutator(Mutator):
    """Mutator for XML-structured blocks."""
    styles = ["xml"]

    @property
    def head(self) -> str:
        """Opening tag content."""
        prefix = self.block.get_chunks_by_style("prefix")
        content = self.block.get_chunks_by_style("content")
        if prefix and content:
            return self.block.text[prefix[0].start:content[-1].end]
        return ""

    @property
    def opening_tag(self) -> str:
        """Just the <tag> part."""
        prefix = self.block.get_chunks_by_style("prefix")
        if prefix:
            return self.block.text[prefix[0].start:prefix[-1].end]
        return ""

    @property
    def closing_tag(self) -> str:
        """The </tag> part."""
        postfix = self.block.get_chunks_by_style("postfix")
        if postfix:
            return self.block.text[postfix[0].start:postfix[-1].end]
        return ""

    def append(self, text: str, logprob: float = None) -> ChunkMeta:
        """Append to content region (before closing tag)."""
        # Find where content region ends (before postfix)
        postfix_chunks = self.block.get_chunks_by_style("postfix")
        if postfix_chunks:
            # Insert before postfix
            insert_pos = self.block.start + postfix_chunks[0].start
            return self.block._raw_insert(insert_pos, text, logprob=logprob, style="content")
        else:
            # No postfix yet, just append
            return self.block._raw_append(text, logprob=logprob, style="content")

    def init(self, tag_name: str, attrs: dict = None) -> None:
        """Initialize XML structure with opening tag."""
        attr_str = ""
        if attrs:
            attr_str = " " + " ".join(f'{k}="{v}"' for k, v in attrs.items())

        self.block._raw_append(f"<{tag_name}{attr_str}>", style="prefix")

    def commit(self, tag_name: str) -> None:
        """Close XML structure with closing tag."""
        self.block._raw_append(f"</{tag_name}>", style="postfix")
```

### MutatorMeta Registry

```python
class MutatorMeta(type):
    """Metaclass that registers mutators by style."""
    _registry: dict[str, type[Mutator]] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        for style in getattr(cls, 'styles', []):
            mcs._registry[style] = cls
        return cls

    @classmethod
    def get_mutator(mcs, style: str) -> type[Mutator]:
        return mcs._registry.get(style, Mutator)
```

### Block Integration

```python
class Block:
    # ... existing fields ...

    _mutator: Mutator | None = None

    @property
    def mutator(self) -> Mutator:
        """Get mutator for this block's style."""
        if self._mutator is None:
            style = self.style[0] if self.style else "default"
            mutator_cls = MutatorMeta.get_mutator(style)
            self._mutator = mutator_cls(self)
        return self._mutator

    # --- Public API (delegates to mutator) ---

    def append(self, text: str, logprob: float = None) -> ChunkMeta:
        return self.mutator.append(text, logprob=logprob)

    def append_child(self, child: Block) -> Block:
        return self.mutator.append_child(child)

    # --- Raw operations (used by mutators) ---

    def _raw_append(self, text: str, logprob: float = None, style: str = None) -> ChunkMeta:
        """Low-level append without mutator interception."""
        # ... direct implementation ...

    def _raw_insert(self, position: int, text: str, logprob: float = None, style: str = None) -> ChunkMeta:
        """Low-level insert at absolute position."""
        # ... direct implementation ...
```

### Flow Example

```python
# Create XML block
block = Block(style=["xml"])
block.mutator.init("message")  # Adds "<message>" with style="prefix"
block.append("Hello World")    # Mutator inserts with style="content"
block.mutator.commit("message") # Adds "</message>" with style="postfix"

# Chunks now:
# [("<message>", prefix), ("Hello World", content), ("</message>", postfix)]

# Text: "<message>Hello World</message>"

# Later append goes to right place:
block.append(" - Updated")
# Mutator inserts before postfix
# Text: "<message>Hello World - Updated</message>"
```

---

## Open Questions

1. **Copy semantics**: Deep copy behavior for subtrees?
2. **Event sourcing**: How to track mutations for undo/versioning?
3. **Schema blocks**: How to represent BlockSchema in new model?

---

## Appendix: Example Usage

```python
# Create root block
root = Block()

# Stream content
root.append("Hello ", logprob=-0.1)
root.append("World", logprob=-0.2)

# Access text
print(root.text)  # "Hello World"

# Add children
with root as r:
    child1 = r.append_child(Block())
    child1.append("Child 1 content")

    child2 = r.append_child(Block())
    child2.append("Child 2 content")

# Full text
print(root._text)  # "Hello WorldChild 1 contentChild 2 content"

# Child text
print(child1.text)  # "Child 1 content"

# Find pattern
pos = root.find("World")  # 6

# Regex
matches = root.regex_find(r"\w+")  # All words
```
