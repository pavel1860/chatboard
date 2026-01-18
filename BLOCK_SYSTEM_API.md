# Block System API Guide

A comprehensive guide to using the PromptView block system for building, storing, and managing structured prompts.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Creating Blocks](#creating-blocks)
4. [Saving Blocks](#saving-blocks)
5. [Retrieving Blocks](#retrieving-blocks)
6. [Diffing Blocks](#diffing-blocks)
7. [Storage Architecture](#storage-architecture)
8. [Best Practices](#best-practices)

---

## Overview

The block system provides a composable, versionable way to build prompts. Each block is:
- **Content-addressed** - Identical blocks are deduplicated automatically
- **Versioned** - Every change is tracked through the artifact system
- **Styled** - Blocks can have roles, tags, and styling metadata
- **Structured** - Blocks form trees with parent-child relationships

---

## Core Concepts

### Block
The fundamental unit of content. Contains text and optional styling metadata.

```python
from promptview.block import Block

# Simple block
block = Block("Hello, world!")

# Block with metadata
block = Block(
    "You are a helpful assistant",
    role="system",
    tags=["instruction"],
    styles=["bold"]
)
```

### BlockTree
A versioned snapshot of a complete block hierarchy, stored in the database with artifact tracking.

### BlockSignature
A content-addressed identifier combining block content + styling. Enables efficient deduplication and comparison.

---

## Creating Blocks

### Basic Block Creation

```python
from promptview.block import Block

# Create a simple block
greeting = Block("Hello!")

# Create a block with children
prompt = Block("System instructions")
prompt /= Block("Be helpful")
prompt /= Block("Be concise")

# Alternative syntax
prompt = Block("System instructions")
task = Block("Task")
task /= Block("Help the user")
prompt /= task
```

### Operators

```python
# /= : Add child block (new line)
parent /= child

# += : Append to content (continues current line)
block += " more text"

# &= : Append without separator
block &= "concatenated"
```

### Block Metadata

```python
block = Block(
    "content here",
    role="system",           # Semantic role (e.g., "system", "user", "assistant")
    tags=["important"],      # Tags for filtering/searching
    styles=["bold", "code"], # Visual styling hints
    attrs={"key": "value"}   # Custom attributes
)
```

---

## Saving Blocks

### Save Within a Turn

Blocks are saved within a `Turn` context, which provides versioning and artifact tracking.

```python
from promptview.prompt import Context
from promptview.model.block_models.block_log import BlockLog

# Create a block
prompt = Block("You are a helpful assistant")
prompt /= Block("Help the user with their questions")

# Save within a turn
async with Context().start_turn() as ctx:
    tree = await BlockLog.add(prompt)
    print(f"Saved as tree ID: {tree.id}")
```

### Automatic Context Detection

`BlockLog.add()` automatically detects the current branch and turn from the context:

```python
async with Context().start_turn() as ctx:
    # These are equivalent:
    tree = await BlockLog.add(prompt)
    tree = await BlockLog.add(prompt, branch_id=ctx.branch.id, turn_id=ctx.turn.id)
```

---

## Retrieving Blocks

### Query Recent Blocks

```python
from promptview.model.block_models.block_log import BlockLog

# Get last 5 blocks
blocks = await BlockLog.last(5)

# Get last 10 blocks
blocks = await BlockLog.last(10)

for block in blocks:
    print(block.render())
```

### Query by Span

```python
# Get blocks from a specific span
blocks = await BlockLog.span("my_component").last(5)
```

### Filter by Status

```python
from promptview.versioning.models import TurnStatus

# Only committed turns (default)
blocks = await BlockLog.last(5)

# All turns (including staged)
query = BlockLogQuery(limit=5)
blocks = await query.all()

# Specific statuses
query = BlockLogQuery(
    limit=5,
    statuses=[TurnStatus.COMMITTED, TurnStatus.STAGED]
)
blocks = await query
```

### Filter by Role

```python
query = BlockLogQuery(limit=10)

# Include only specific roles
query = query.include_roles(["system", "user"])
blocks = await query

# Exclude specific roles
query = query.exclude_roles(["debug"])
blocks = await query
```

### Direct Database Queries

For advanced queries, use the ORM directly:

```python
from promptview.versioning.models import BlockTree, BlockNode, BlockSignature, BlockModel

# Get all trees from last 7 days
trees = await BlockTree.vquery(
    limit=100,
    statuses=[TurnStatus.COMMITTED]
).where(
    lambda bt: bt.created_at > datetime.now() - timedelta(days=7)
).include(
    BlockNode.query().include(
        BlockSignature.query().include(BlockModel)
    )
).json()
```

---

## Diffing Blocks

### Compare Saved Trees

```python
from promptview.model.block_models.block_diff import diff_block_trees

# Compare two saved block trees
diff = await diff_block_trees(tree1_id=1, tree2_id=2)

# Check if there are changes
if diff.has_changes:
    print(diff.render_diff())

# Get summary
summary = diff.change_summary
print(f"Changes: +{summary['added']} -{summary['removed']} ~{summary['modified']}")
```

### Compare In-Memory Blocks

```python
from promptview.model.block_models.block_diff import diff_block_objects

# Create two blocks
block1 = Block("Version 1")
block1 /= Block("Original content")

block2 = Block("Version 1")
block2 /= Block("Modified content")  # Changed

# Diff without saving
diff = await diff_block_objects(block1, block2)
print(diff.render_diff())
```

### Analyze Differences

```python
diff = await diff_block_trees(1, 2)

# Inspect added nodes
for node in diff.added:
    print(f"Added at {node.path}: {node.content}")

# Inspect removed nodes
for node in diff.removed:
    print(f"Removed from {node.path}: {node.content}")

# Inspect modified nodes
for node in diff.modified:
    print(f"Modified at {node.path}")
    print(f"  Old signature: {node.old_signature_id[:8]}...")
    print(f"  New signature: {node.new_signature_id[:8]}...")

# Inspect moved nodes
for node in diff.moved:
    print(f"Moved from {node.old_path} to {node.new_path}")
```

---

## Storage Architecture

### Three-Layer Deduplication

The block system uses a three-tier storage architecture for maximum efficiency:

```
┌─────────────────────────────────────────────────────┐
│ 1. blocks (content only)                            │
│    - Deduplicates identical text content            │
│    - Content-addressed by SHA-256 hash              │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ 2. block_signatures (content + styling)             │
│    - Deduplicates blocks with same content + style  │
│    - Combines block_id + styles + role + tags       │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ 3. block_nodes (tree structure)                     │
│    - Stores only: tree_id + path + signature_id     │
│    - ~20 bytes per node vs 200+ bytes before        │
└─────────────────────────────────────────────────────┘
```

### Storage Efficiency Example

```python
# Two prompts with 7/8 identical blocks:
prompt1 = create_prompt("what is the weather?")
prompt2 = create_prompt("what is the capital of Italy?")

# Without deduplication: 16 blocks × 200 bytes = 3.2 KB
# With Option 1 architecture:
#   - 9 blocks (7 shared, 2 unique)
#   - 9 signatures (7 shared, 2 unique)
#   - 16 lightweight nodes (8 per tree)
#   Total: ~0.5 KB (85% reduction!)
```

### Signature-Based Comparison

Because blocks are content-addressed via signatures, comparisons are O(n):

```python
# Fast diff using signature hashes
diff = await diff_block_trees(tree1_id, tree2_id)

# If signature_id matches → identical block
# If path matches but signature differs → modified
# If signature exists but path differs → moved
```

---

## Best Practices

### 1. Use Descriptive Roles

```python
# Good: Semantic roles
Block("You are a helpful assistant", role="system")
Block("What is the weather?", role="user")
Block("The weather is sunny", role="assistant")

# Avoid: Generic or missing roles
Block("You are a helpful assistant")  # No context
```

### 2. Tag for Discoverability

```python
# Tag blocks for filtering and search
Block(
    "Critical: Handle errors gracefully",
    tags=["instruction", "error-handling", "critical"]
)

# Query by tags later
blocks = await BlockLog.last(10)
error_blocks = [b for b in blocks if "error-handling" in b.tags]
```

### 3. Build Reusable Templates

```python
def create_system_prompt(personality: str) -> Block:
    """Reusable prompt template"""
    system = Block(f"You are a {personality} assistant", role="system")

    rules = Block("# Rules")
    rules /= Block("* Be helpful and accurate")
    rules /= Block("* Cite sources when possible")
    system /= rules

    return system

# Use the template
prompt = create_system_prompt("knowledgeable")
prompt /= Block("User's question here", role="user")
```

### 4. Version Your Prompts

```python
# Save each iteration with descriptive turns
async with Context().start_turn() as ctx:
    v1 = create_prompt_v1()
    await BlockLog.add(v1)

async with Context().start_turn() as ctx:
    v2 = create_prompt_v2()  # Improved version
    await BlockLog.add(v2)

# Compare versions
diff = await diff_block_trees(tree1_id=1, tree2_id=2)
print(diff.render_diff())
```

### 5. Use Diff for Debugging

```python
# Before making changes, save current state
async with Context().start_turn() as ctx:
    baseline = await BlockLog.add(current_prompt)

# Make changes
modified_prompt = improve_prompt(current_prompt)

# Compare before saving
diff = await diff_block_objects(current_prompt, modified_prompt)
if diff.has_changes:
    print(f"About to make {diff.change_summary['total_changes']} changes")
    print(diff.render_diff())

    # Confirm and save
    async with Context().start_turn() as ctx:
        await BlockLog.add(modified_prompt)
```

### 6. Clean Up Experiments

```python
from promptview.versioning.models import TurnStatus

# Mark experimental turns as reverted
async with Context().start_turn() as ctx:
    # ... experiment ...
    if not success:
        await ctx.turn.revert("Experiment failed")

# Query only successful turns
blocks = await BlockLogQuery(
    statuses=[TurnStatus.COMMITTED]
).last(10)
```

### 7. Leverage Artifact Tracking

```python
# Every block tree has an artifact
async with Context().start_turn() as ctx:
    tree = await BlockLog.add(prompt)

    # Access artifact metadata
    artifact = tree.artifact
    print(f"Branch: {artifact.branch_id}")
    print(f"Turn: {artifact.turn_id}")
    print(f"Version: {artifact.version}")
    print(f"Created: {artifact.created_at}")

# Query by artifact properties
trees = await BlockTree.vquery(
    limit=10,
    statuses=[TurnStatus.COMMITTED]
).where(
    lambda bt: bt.artifact.branch_id == 1
).json()
```

---

## API Reference Quick Links

### Core Classes
- `Block` - [`@promptview/block`](./promptview/block/block9/block.py)
- `BlockLog` - [`@promptview/model/block_models/block_log.py`](./promptview/model/block_models/block_log.py)
- `BlockTree` - [`@promptview/model/versioning/models.py`](./promptview/model/versioning/models.py)

### Diff Utilities
- `diff_block_trees()` - [`@promptview/model/block_models/block_diff.py`](./promptview/model/block_models/block_diff.py)
- `diff_block_objects()` - [`@promptview/model/block_models/block_diff.py`](./promptview/model/block_models/block_diff.py)

### Context & Versioning
- `Context` - [`@promptview/prompt/context.py`](./promptview/prompt/context.py)
- `Turn` - [`@promptview/model/versioning/models.py`](./promptview/model/versioning/models.py)
- `Artifact` - [`@promptview/model/versioning/models.py`](./promptview/model/versioning/models.py)

---

## Examples

See the following notebooks for complete examples:

1. **Storage & Deduplication**: [`research/option1_block_storage_test.ipynb`](./research/option1_block_storage_test.ipynb)
   - Demonstrates storage efficiency
   - Shows signature reuse
   - Validates deduplication

2. **Diffing Blocks**: [`research/block_diff_test.ipynb`](./research/block_diff_test.ipynb)
   - Compare saved trees
   - In-memory diffing
   - Analyze changes

3. **Basic Usage**: [`research/example_playground.ipynb`](./research/example_playground.ipynb)
   - Initialize database
   - Save versioned models
   - Query patterns

---

## Troubleshooting

### Issue: "Cannot access attribute 'add_block'"
**Solution**: Use `BlockLog.add()` instead of `turn.add_block()` or `context.add_block()`.

```python
# ❌ Wrong
await ctx.add_block(block)
await ctx.turn.add_block(block)

# ✅ Correct
await BlockLog.add(block)
```

### Issue: "ValidationError: created_at should be a valid datetime"
**Solution**: Ensure you're using the latest code with `created_at` in the INSERT statement.

### Issue: Blocks not deduplicating
**Check**:
1. Are you creating new Block instances each time? (Expected behavior)
2. Check signature reuse in database:
```python
from promptview.utils.db_connections import PGConnectionManager

stats = await PGConnectionManager.fetch_one("""
    SELECT COUNT(DISTINCT bs.id) as unique_sigs,
           COUNT(bn.id) as total_refs
    FROM block_signatures bs
    LEFT JOIN block_nodes bn ON bn.signature_id = bs.id
""")
print(f"Reuse rate: {stats['total_refs'] / stats['unique_sigs']:.2f}x")
```

---

## Support

For issues, questions, or contributions, see the main [CLAUDE.md](./CLAUDE.md) documentation.
