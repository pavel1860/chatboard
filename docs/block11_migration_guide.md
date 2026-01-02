# Block11 Migration Guide: Frontend Changes

This document explains the structural differences between Block9 and Block11 for frontend developers. It covers the data model changes, API differences, and required frontend modifications.

---

## Executive Summary

Block11 is a complete rewrite of the block system with these key changes:

1. **Flat text storage** → Span-based linked list with shared `BlockText`
2. **Nested content structure** (BlockSent → BlockChunk) → Single `Span` per block
3. **Inline `model_dump()`** → Merkle-hashed content-addressed storage
4. **Direct block serialization** → Separate storage for spans, blocks, and trees

---

## Data Model Changes

### Block9 Structure

```
Block
├── content: BlockSent
│   ├── content: str
│   ├── prefix: str
│   ├── postfix: str
│   └── children: list[BlockChunk]
│       └── BlockChunk
│           ├── content: str
│           ├── logprob: float | None
│           ├── prefix: str
│           └── postfix: str
├── children: list[Block]
├── prefix: BlockSent
├── postfix: BlockSent
├── role: str | None
├── tags: list[str]
├── styles: list[str]
├── attrs: dict[str, AttrBlock]
└── artifact_id: int | None
```

### Block11 Structure

```
Block
├── span: Span                      # Single span (replaces content/prefix/postfix)
│   ├── prefix: list[BlockChunk]
│   ├── content: list[BlockChunk]
│   ├── postfix: list[BlockChunk]
│   ├── prev: Span | None           # Linked list pointer
│   ├── next: Span | None           # Linked list pointer
│   └── owner: BlockText | None     # Back-reference to shared BlockText
├── children: list[Block]
├── parent: Block | None
├── block_text: BlockText           # Shared text container (NEW)
├── mutator: Mutator                # Strategy for block manipulation (NEW)
├── role: str | None
├── tags: list[str]
├── _style: list[str]               # Renamed from 'styles'
└── attrs: dict[str, Any]
```

### Key Structural Differences

| Aspect | Block9 | Block11 |
|--------|--------|---------|
| **Content container** | `BlockSent` with nested `BlockChunk` children | Single `Span` with prefix/content/postfix lists |
| **Text ownership** | Each block owns its text | Shared `BlockText` owns all spans |
| **Chunk structure** | `BlockChunk(content, prefix, postfix)` | `BlockChunk(content, logprob)` - simpler |
| **Block prefix/postfix** | Separate `BlockSent` fields | Part of the `Span` |
| **Text ordering** | Tree traversal | Linked list in `BlockText` |
| **Mutators** | Not present | Strategy pattern for block manipulation |

---

## Serialization Format Changes

### Block9 `model_dump()` Output

```json
{
  "_type": "Block",
  "id": "abc123",
  "content": {
    "content": "",
    "prefix": "",
    "postfix": "",
    "children": [
      {"content": "Hello", "logprob": -0.5, "prefix": "", "postfix": ""}
    ]
  },
  "prefix": {"content": "", "children": []},
  "postfix": {"content": "", "children": []},
  "children": [...],
  "role": "user",
  "tags": ["message"],
  "styles": ["xml"],
  "attrs": {},
  "is_wrapper": false
}
```

### Block11 `model_dump()` Output

```json
{
  "id": "abc123",
  "span": {
    "prefix": "",
    "content": "Hello",
    "postfix": "\n",
    "prefix_chunks": [],
    "content_chunks": [{"content": "Hello", "logprob": -0.5}],
    "postfix_chunks": [{"content": "\n"}]
  },
  "children": [...],
  "role": "user",
  "tags": ["message"],
  "styles": ["xml"],
  "attrs": {},
  "name": null,
  "type_name": null
}
```

---

## Database Storage Changes

### Block9 Storage

Single table with serialized JSON:
```sql
-- block_logs table
id          INT PRIMARY KEY
artifact_id INT
nodes       JSONB  -- Entire block tree serialized
```

### Block11 Storage (Merkle Tree)

Three separate tables with content-addressed hashing:

```sql
-- block_spans: Content-addressed text spans
id           TEXT PRIMARY KEY  -- SHA256 hash of content
prefix_text  TEXT
content_text TEXT
postfix_text TEXT
prefix_chunks  JSONB
content_chunks JSONB
postfix_chunks JSONB

-- blocks: Merkle tree nodes
id         TEXT PRIMARY KEY  -- SHA256 hash of (span_id + children + metadata)
span_id    TEXT REFERENCES block_spans(id)
role       TEXT
tags       TEXT[]
styles     TEXT[]
name       TEXT
type_name  TEXT
attrs      JSONB
children   TEXT[]  -- Array of child block IDs (hashes)

-- block_trees: Versioned root references
id          INT PRIMARY KEY
artifact_id INT
root_id     TEXT REFERENCES blocks(id)
branch_id   INT
turn_id     INT
```

### Important: Content Deduplication

Block11 uses Merkle hashing - identical subtrees share storage:

```
Tree A: root_a → child_x → leaf_1
Tree B: root_b → child_x → leaf_1  (same child_x, same leaf_1!)
```

This means:
- The same block content may appear in multiple trees
- Blocks are immutable once created
- Updates create new blocks with new hashes

---

## API Response Changes

### Fetching Blocks

**Block9 Response:**
```json
{
  "artifact_id": 123,
  "nodes": { /* entire serialized block tree */ }
}
```

**Block11 Response:**
```json
{
  "artifact_id": 123,
  "tree": {
    "id": 456,
    "root_id": "a1b2c3...",  // Merkle hash
    "branch_id": 1,
    "turn_id": 10
  },
  "block": { /* deserialized Block */ }
}
```

### Key API Differences

| Operation | Block9 | Block11 |
|-----------|--------|---------|
| Get block by artifact | `GET /blocks/{artifact_id}` returns `{nodes: ...}` | Same endpoint, different response structure |
| Block identity | `artifact_id` | `root_id` (Merkle hash) + `artifact_id` (versioning) |
| Partial updates | Possible | Not possible (immutable content-addressed) |

---

## Frontend Migration Checklist

### 1. Update Type Definitions

```typescript
// OLD (Block9)
interface BlockChunk {
  content: string;
  logprob?: number;
  prefix: string;
  postfix: string;
}

interface BlockSent {
  content: string;
  prefix: string;
  postfix: string;
  children: BlockChunk[];
}

interface Block {
  id: string;
  content: BlockSent;
  prefix: BlockSent;
  postfix: BlockSent;
  children: Block[];
  role?: string;
  tags: string[];
  styles: string[];
  attrs: Record<string, AttrBlock>;
  artifact_id?: number;
}

// NEW (Block11)
interface BlockChunk {
  content: string;
  logprob?: number;
  // No prefix/postfix - these are in Span now
}

interface Span {
  prefix: string;
  content: string;
  postfix: string;
  prefix_chunks?: BlockChunk[];
  content_chunks?: BlockChunk[];
  postfix_chunks?: BlockChunk[];
}

interface Block {
  id: string;
  span: Span;  // Single span replaces content/prefix/postfix
  children: Block[];
  role?: string;
  tags: string[];
  styles: string[];  // Note: internal field is _style
  attrs: Record<string, any>;
  name?: string;      // For BlockSchema
  type_name?: string; // For BlockSchema
}
```

### 2. Update Content Access

```typescript
// OLD (Block9)
function getBlockText(block: Block): string {
  const chunks = block.content.children;
  return chunks.map(c => c.prefix + c.content + c.postfix).join('');
}

function getBlockPrefix(block: Block): string {
  return block.prefix.content;
}

// NEW (Block11)
function getBlockText(block: Block): string {
  return block.span.prefix + block.span.content + block.span.postfix;
}

function getBlockPrefix(block: Block): string {
  return block.span.prefix;
}
```

### 3. Update Chunk Access (for logprobs)

```typescript
// OLD (Block9)
function getLogprobs(block: Block): number[] {
  return block.content.children
    .filter(c => c.logprob !== undefined)
    .map(c => c.logprob!);
}

// NEW (Block11)
function getLogprobs(block: Block): number[] {
  const allChunks = [
    ...(block.span.prefix_chunks || []),
    ...(block.span.content_chunks || []),
    ...(block.span.postfix_chunks || []),
  ];
  return allChunks
    .filter(c => c.logprob !== undefined)
    .map(c => c.logprob!);
}
```

### 4. Update Block Rendering

```typescript
// OLD (Block9)
function renderBlock(block: Block): string {
  let result = '';

  // Block prefix
  result += renderBlockSent(block.prefix);

  // Content
  result += renderBlockSent(block.content);

  // Children
  for (const child of block.children) {
    result += renderBlock(child);
  }

  // Block postfix
  result += renderBlockSent(block.postfix);

  return result;
}

// NEW (Block11)
function renderBlock(block: Block): string {
  let result = '';

  // Span contains prefix + content + postfix
  result += block.span.prefix;
  result += block.span.content;

  // Children (before postfix in block11)
  for (const child of block.children) {
    result += renderBlock(child);
  }

  // Postfix after children
  result += block.span.postfix;

  return result;
}
```

### 5. Handle BlockSchema Changes

```typescript
// OLD (Block9)
interface BlockSchema extends Block {
  type: string;  // Type was a Type object reference
  name: string;
  is_list: boolean;
  is_list_item: boolean;
}

// NEW (Block11)
interface BlockSchema extends Block {
  name: string;
  type_name?: string;  // String representation of type
  attrs: Record<string, any>;
}
```

### 6. Update Storage/Cache Keys

```typescript
// OLD (Block9) - artifact_id was the unique key
const cacheKey = `block:${block.artifact_id}`;

// NEW (Block11) - Merkle hash is content identity, artifact_id is version identity
const contentKey = `block:content:${block.root_id}`;      // For deduplication
const versionKey = `block:version:${block.artifact_id}`; // For versioning
```

---

## Streaming Changes

Block11 has a new event-based streaming architecture:

### Block9 Streaming
- Direct append to `BlockSent.children`
- No formal event system

### Block11 Streaming Events

```typescript
type ParserEvent =
  | { type: 'init'; block: Block; chunks: BlockChunk[] }
  | { type: 'delta'; chunks: BlockChunk[] }
  | { type: 'commit'; chunks: BlockChunk[] };

// Parser emits events
parser.on('event', (event: ParserEvent) => {
  switch (event.type) {
    case 'init':
      // New block started
      handleNewBlock(event.block, event.chunks);
      break;
    case 'delta':
      // Content appended to current block
      handleDelta(event.chunks);
      break;
    case 'commit':
      // Block finalized
      handleCommit(event.chunks);
      break;
  }
});
```

---

## Summary of Breaking Changes

1. **`content: BlockSent`** → **`span: Span`**
2. **`block.prefix` / `block.postfix`** → **`block.span.prefix` / `block.span.postfix`**
3. **`BlockChunk.prefix/postfix`** → Removed (now in Span)
4. **`styles`** → **`_style`** (internal) but serializes as `styles`
5. **`artifact_id` as identity** → **`root_id` (Merkle hash) for content, `artifact_id` for versioning**
6. **Single table storage** → **Three-table Merkle tree storage**
7. **Mutable blocks** → **Immutable content-addressed blocks**

---

## Questions?

If you encounter issues during migration, check:
1. Are you accessing `block.span` instead of `block.content`?
2. Are you handling the chunk structure correctly (chunks in Span, not nested)?
3. Are you using the correct identity (Merkle hash vs artifact_id)?
