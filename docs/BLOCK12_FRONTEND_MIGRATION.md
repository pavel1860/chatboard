# Block12 Frontend Migration Guide

This guide documents the differences between Block11 and Block12 for frontend integration, including data serialization formats, streaming API changes, and relevant files.

## Overview

Block12 simplifies the block system by:
1. Each block owns its own local text string (no shared `BlockText`)
2. Chunks use positions relative to the block's local text
3. Uses Merkle tree storage with content-addressed deduplication
4. Simplified serialization format

---

## Data Serialization Differences

### Block11 `model_dump()` Format

```json
{
  "role": "assistant",
  "tags": ["response"],
  "style": ["xml"],
  "attrs": {},
  "span": {
    "prefix": [
      {"content": "<response>", "logprob": -0.1}
    ],
    "content": [
      {"content": "Hello world", "logprob": -0.2}
    ],
    "postfix": [
      {"content": "</response>", "logprob": -0.1}
    ]
  },
  "children": [...]
}
```

**Key characteristics:**
- `span` contains three separate chunk arrays: `prefix`, `content`, `postfix`
- Chunks have `content` and `logprob` fields
- No explicit `id` or `path` fields at block level

### Block12 `model_dump()` Format

```json
{
  "id": "a1b2c3d4",
  "path": "0.1",
  "text": "<response>Hello world</response>",
  "role": "assistant",
  "tags": ["response"],
  "style": ["xml"],
  "attrs": {},
  "chunks": [
    {"id": "x1y2z3", "start": 0, "end": 10, "logprob": -0.1, "style": "prefix"},
    {"id": "a4b5c6", "start": 10, "end": 21, "logprob": -0.2, "style": null},
    {"id": "d7e8f9", "start": 21, "end": 32, "logprob": -0.1, "style": "postfix"}
  ],
  "children": [...]
}
```

**Key characteristics:**
- `id`: Unique block identifier (8-char hex)
- `path`: Index path in tree (e.g., "0.1.2")
- `text`: Single string containing all block text
- `chunks`: Array of chunk metadata with positions relative to `text`
  - `id`: Unique chunk identifier
  - `start`/`end`: Byte positions in `text`
  - `logprob`: Log probability from LLM
  - `style`: Chunk type (`"prefix"`, `"postfix"`, `null` for content)

---

## Storage Format Differences

### Block11 Storage

Block11 stores blocks with separate prefix/content/postfix text fields:

```sql
-- block_spans table
id, prefix_text, content_text, postfix_text,
prefix_chunks (JSON), content_chunks (JSON), postfix_chunks (JSON)
```

### Block12 Storage (Merkle Tree)

Block12 uses content-addressed Merkle tree storage with four tables:

```sql
-- block_spans: Content-addressed text+chunks
id (SHA256 hash), prefix_text, content_text, postfix_text,
prefix_chunks, content_chunks, postfix_chunks

-- blocks: Merkle tree nodes
id (Merkle hash), span_id, role, tags, styles, name, attrs,
children (array of child Merkle hashes), block_type, path

-- block_trees: Tree containers with versioning
id, artifact_id, branch_id, turn_id, span_id

-- block_tree_blocks: Junction table (many-to-many)
tree_id, block_id, position, is_root
```

**Storage API:**

```python
from chatboard.model.block_models.block12_storage import BlockLog

# Store a block tree
tree = await BlockLog.add(block, branch_id, turn_id)

# Query block trees
blocks = await BlockLog.query().tail(10)
blocks = await BlockLog.query().where(role="assistant")

# Get by ID
block = await BlockLog.get(tree_id)

# Get by artifact IDs
blocks_dict = await BlockLog.get_by_artifacts([1, 2, 3])
```

---

## Streaming API Differences

### Block11 ParserEvent

```python
@dataclass
class ParserEvent:
    path: str
    type: Literal["block_stream", "block_init", "block_commit", "block_delta", "block"] | BlockSpanEvent
    value: Block | BlockChunkList
```

Event types:
- `block_init`: Block instantiated
- `block_commit`: Block completed
- `block_delta`: Content appended
- `block_stream`: Streaming marker
- `block`: Full block emitted

### Block12 ParserEvent

```python
@dataclass
class ParserEvent:
    path: str
    type: Literal["block_init", "block_commit", "block_delta"]
    block: Block
    chunks: list[BlockChunk] | None = None

    @property
    def value(self) -> Block | list[BlockChunk]:
        if self.chunks is None:
            return self.block
        return self.chunks
```

Event types:
- `block_init`: Block instantiated (chunks may contain opening tag)
- `block_commit`: Block completed (chunks may contain closing tag)
- `block_delta`: Content appended (chunks contains the delta)

**Key differences:**
- Block12 always includes the `block` reference
- Separate `chunks` field for delta content
- `value` property provides backward compatibility
- Removed `block_stream` and `block` event types

### Frontend Event Handling

**Block11:**
```typescript
switch (event.type) {
  case "block_init":
    createBlock(event.path, event.value);
    break;
  case "block_delta":
    appendToBlock(event.path, event.value.chunks);
    break;
  case "block_commit":
    finalizeBlock(event.path, event.value);
    break;
}
```

**Block12:**
```typescript
switch (event.type) {
  case "block_init":
    // event.block contains the initialized block
    // event.chunks may contain prefix/opening tag
    createBlock(event.path, event.block, event.chunks);
    break;
  case "block_delta":
    // event.chunks contains the delta text
    appendToBlock(event.path, event.block, event.chunks);
    break;
  case "block_commit":
    // event.chunks may contain postfix/closing tag
    finalizeBlock(event.path, event.block, event.chunks);
    break;
}
```

---

## Chunk Format Differences

### Block11 BlockChunk

```python
@dataclass
class BlockChunk:
    content: str
    logprob: float | None = None
    style: str | None = None

    # Helper methods
    def is_text(self) -> bool
    def is_line_end(self) -> bool
    def isspace(self) -> bool
    def split(self, index: int) -> tuple[BlockChunk, BlockChunk]
```

### Block12 ChunkMeta + BlockChunk

```python
@dataclass
class ChunkMeta:
    """Metadata for a chunk (stored with block)."""
    start: int
    end: int
    logprob: float | None = None
    style: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:8])

@dataclass
class BlockChunk:
    """Chunk with content (for operations)."""
    content: str
    logprob: float | None = None
    style: str | None = None
    meta: ChunkMeta | None = None
```

**Key difference:** Block12 separates metadata (`ChunkMeta` with positions) from content (`BlockChunk`). The block's `chunks` list contains `ChunkMeta` objects; use `block.get_chunks()` to get `BlockChunk` objects with content.

---

## Code Migration Examples

### Accessing Block Text

**Block11:**
```python
# Get content text only
content = block.span.content_text

# Get full text (prefix + content + postfix)
full_text = block.span.text

# Get chunks
content_chunks = list(block.span.content)
prefix_chunks = list(block.span.prefix)
```

**Block12:**
```python
# Get content text (via mutator, excludes styled chunks)
content = block.content

# Get full text (all local text)
full_text = block.text

# Get chunk metadata
chunk_metas = block.chunks

# Get chunks with content
chunks = block.get_chunks()
```

### Appending Content

**Block11:**
```python
events = block.append("Hello")
# Returns list of BlockChunkList or Block events
```

**Block12:**
```python
events = block.append("Hello")
# Returns list of Block or BlockChunk events
```

### Serialization for Frontend

**Block11:**
```python
data = block.model_dump()
# Access: data["span"]["content"][0]["content"]
```

**Block12:**
```python
data = block.model_dump()
# Access: data["text"][data["chunks"][0]["start"]:data["chunks"][0]["end"]]
```

---

## Relevant Files

### Block12 Core
- `chatboard/block/block12/block.py` - Block class with local text ownership
- `chatboard/block/block12/chunk.py` - ChunkMeta and BlockChunk classes
- `chatboard/block/block12/schema.py` - BlockSchema, BlockListSchema
- `chatboard/block/block12/parsers.py` - XmlParser, ParserEvent
- `chatboard/block/block12/mutator.py` - Mutator and Stylizer classes
- `chatboard/block/block12/transform.py` - Block transformation logic

### Block12 Storage
- `chatboard/model/block_models/block12_storage.py` - Merkle tree storage layer
- `chatboard/model/versioning/models.py` - BlockTree, BlockModel, BlockSpan models

### Block11 (Legacy)
- `chatboard/block/block11/block.py` - Legacy Block class
- `chatboard/block/block11/span.py` - Span, BlockChunk, BlockChunkList
- `chatboard/block/block11/parsers.py` - Legacy XmlParser
- `chatboard/block/block11/schema.py` - Legacy BlockSchema

### FBP/Streaming
- `chatboard/prompt/fbp_process.py` - Process base class, Stream, StreamController

---

## Type Mapping Summary

| Concept | Block11 | Block12 |
|---------|---------|---------|
| Text storage | `Span` with prefix/content/postfix | Block's `_text` string |
| Chunk metadata | Part of `BlockChunk` | Separate `ChunkMeta` class |
| Position tracking | Global via `BlockText` | Local to each block |
| Block ID | None (use object identity) | `block.id` (8-char hex) |
| Tree path | `block.path` (Path object) | `block.path` (IndexPath object) |
| Storage dedup | None | Merkle tree hashing |
| Parser events | 5 event types | 3 event types |
