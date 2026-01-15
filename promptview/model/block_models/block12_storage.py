"""
Block12 Merkle Tree Storage Layer.

Stores Block12 structures with content-addressed Merkle tree deduplication.
Identical subtrees automatically share storage.

Schema:
- block_spans: Content-addressed text+chunks storage
- blocks: Merkle tree nodes (content-addressed by Merkle hash)
- block_trees: Tree containers with versioning
- block_tree_blocks: Junction table for tree ↔ block many-to-many relationship

Junction Table Benefits:
- Efficient batch loading (3 queries instead of N recursive queries)
- Reverse lookups ("which trees contain this block?")
- Block usage counting (for garbage collection)
- Explicit ordering via position column

Block12 Architecture:
- Each block owns its local text string (no shared root string)
- Chunks store positions relative to block's local text
- Tree structure determines rendering order (depth-first concatenation)
- No start/end position management needed
"""

import hashlib
import json
import datetime as dt
from typing import TYPE_CHECKING, Any

from ..versioning.models import BlockTree, BlockModel, BlockSpan, BlockTreeBlock
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block.block12 import Block, BlockSchema, ChunkMeta
    from ..sql2.pg_query_builder import PgQueryBuilder


# =============================================================================
# Hash Computation
# =============================================================================

def compute_text_hash(
    text: str,
    chunks: list[dict],
) -> str:
    """
    Compute content-addressed hash for block text and chunks.

    Args:
        text: The block's text content
        chunks: List of chunk dicts with {start, end, logprob, style, id}
    """
    data = {
        "text": text,
        "chunks": chunks,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_block_hash(
    span_id: str | None,
    role: str | None,
    tags: list[str],
    styles: list[str],
    name: str | None,
    attrs: dict,
    children: list[str],
    block_type: str = "block",
    path: str = "",
) -> str:
    """
    Compute Merkle hash for a block.

    Includes children's IDs, so identical subtrees have identical hashes.
    """
    data = {
        "span_id": span_id,
        "role": role,
        "tags": sorted(tags) if tags else [],
        "styles": sorted(styles) if styles else [],
        "name": name,
        "attrs": attrs,
        "children": children,
        "block_type": block_type,
        "path": path,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# Serialization (Dump)
# =============================================================================

def dump_chunks_for_hash(chunks: list["ChunkMeta"]) -> list[dict]:
    """Serialize ChunkMeta for hashing (excludes random id for determinism)."""
    return [
        {
            "start": c.start,
            "end": c.end,
            "logprob": c.logprob,
            "style": c.style,
        }
        for c in chunks
    ]


def dump_chunks(chunks: list["ChunkMeta"]) -> list[dict]:
    """Serialize ChunkMeta list to list of dicts (includes id for storage)."""
    return [
        {
            "id": c.id,
            "start": c.start,
            "end": c.end,
            "logprob": c.logprob,
            "style": c.style,
        }
        for c in chunks
    ]


def dump_block(block: "Block") -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """
    Serialize a Block12 tree to storage format using Merkle hashing.

    Returns:
        (root_block_data, all_blocks_dict, all_spans_dict)

    The blocks dict is keyed by Merkle hash, spans dict by content hash.
    Duplicate subtrees will naturally have same hash and overwrite.
    """
    from ...block.block12 import BlockSchema

    all_spans: dict[str, dict] = {}
    all_blocks: dict[str, dict] = {}

    def get_block_type(blk: "Block") -> str:
        """Determine the block type string."""
        if isinstance(blk, BlockSchema):
            return "schema"
        return "block"

    def process_block(blk: "Block", path: str = "") -> str:
        """Process a block and return its Merkle hash."""
        # Process children first (post-order for Merkle)
        child_ids = []
        for i, child in enumerate(blk.children):
            child_path = f"{path}.{i}" if path else str(i)
            child_id = process_block(child, child_path)
            child_ids.append(child_id)

        # Process text and chunks
        span_id = None
        text = blk.text
        if text or blk.chunks:
            # Use hash-only chunks (no random id) for deterministic hashing
            chunks_for_hash = dump_chunks_for_hash(blk.chunks)
            span_id = compute_text_hash(text, chunks_for_hash)

            # Use full chunks (with id) for storage
            chunks_data = dump_chunks(blk.chunks)

            # Store in spans table format (reusing BlockSpan structure)
            # We store text in content_text, chunks in content_chunks
            all_spans[span_id] = {
                "id": span_id,
                "prefix_text": "",
                "content_text": text,
                "postfix_text": "",
                "prefix_chunks": [],
                "content_chunks": chunks_data,
                "postfix_chunks": [],
            }

        # Determine block type
        block_type = get_block_type(blk)

        # Get schema-specific fields
        name = None
        if isinstance(blk, BlockSchema):
            name = blk.name

        # Compute Merkle hash
        block_id = compute_block_hash(
            span_id=span_id,
            role=blk.role,
            tags=blk.tags or [],
            styles=blk.style or [],
            name=name,
            attrs=blk.attrs or {},
            children=child_ids,
            block_type=block_type,
            path=path,
        )

        # Store block data
        all_blocks[block_id] = {
            "id": block_id,
            "span_id": span_id,
            "role": blk.role,
            "tags": blk.tags or [],
            "styles": blk.style or [],
            "name": name,
            "type_name": None,
            "attrs": blk.attrs or {},
            "children": child_ids,
            "is_rendered": False,
            "block_type": block_type,
            "item_name": None,
            "path": path,
        }

        return block_id

    root_id = process_block(block, "")
    root_data = all_blocks[root_id]

    return root_data, all_blocks, all_spans


# =============================================================================
# Deserialization (Load)
# =============================================================================

def load_block(
    root_id: str,
    blocks: dict[str, dict],
    spans: dict[str, dict],
) -> "Block":
    """
    Reconstruct a Block12 tree from storage format.

    Creates proper Block/BlockSchema instances based on block_type.
    Reconstructs chunks from stored data.

    Each block gets its own local text from its span. No shared string.
    """
    from ...block.block12 import Block, BlockSchema, ChunkMeta

    def get_span_data(block_data: dict) -> tuple[str, list[dict]]:
        """Extract text and chunks from a block's span."""
        span_id = block_data.get("span_id")
        if not span_id:
            return "", []

        # Check for inline span (from include query)
        span_data = block_data.get("span")
        if span_data and isinstance(span_data, dict):
            return span_data.get("content_text", ""), span_data.get("content_chunks", [])

        # Look up in spans dict
        span_data = spans.get(span_id)
        if span_data:
            return span_data.get("content_text", ""), span_data.get("content_chunks", [])

        return "", []

    def create_block_instance(block_data: dict) -> "Block":
        """Create a Block/BlockSchema instance with metadata."""
        block_type = block_data.get("block_type", "block")
        styles = block_data.get("styles") or []
        tags = block_data.get("tags") or []
        role = block_data.get("role")
        attrs = block_data.get("attrs") or {}

        if block_type == "schema":
            return BlockSchema(
                name=block_data.get("name"),
                role=role,
                tags=tags,
                style=styles,
                attrs=attrs,
            )
        else:
            return Block(
                role=role,
                tags=tags,
                style=styles,
                attrs=attrs,
            )

    def build_tree(block_id: str) -> "Block":
        """
        Recursively build the block tree.

        Each block gets its own local text from its span.

        Args:
            block_id: ID of block to build

        Returns:
            Block instance with local text
        """
        block_data = blocks[block_id]
        blk = create_block_instance(block_data)

        # Get this block's text and chunks
        text, chunks_data = get_span_data(block_data)

        # Set local text
        blk._text = text

        # Create chunk metadata (positions are relative to local text)
        for chunk_data in chunks_data:
            blk.chunks.append(ChunkMeta(
                start=chunk_data["start"],
                end=chunk_data["end"],
                logprob=chunk_data.get("logprob"),
                style=chunk_data.get("style"),
                id=chunk_data.get("id", ""),
            ))

        # Recursively build children
        for child_id in block_data.get("children", []):
            child = build_tree(child_id)
            child.parent = blk
            blk.children.append(child)

        return blk

    return build_tree(root_id)


def parse_block_tree_query_result(trees: list[dict]) -> list["Block"]:
    """
    Parse BlockTree query results with nested blocks into Block structures.

    This function is designed to be used with PgQueryBuilder.parse() to transform
    query results that include the `blocks` relation via junction table.

    Expected input format (from query with .include(BlockModel)):
    [
        {
            'id': 1,
            'artifact_id': 1,
            'blocks': [
                {
                    'id': 'hash...',
                    'path': '',
                    'span': {'content_text': '...', 'content_chunks': [...]},
                    'children': ['child_hash1', 'child_hash2'],
                    ...
                },
                ...
            ]
        }
    ]

    Args:
        trees: List of BlockTree dicts from query result

    Returns:
        List of reconstructed Block instances
    """
    results = []

    for tree_data in trees:
        blocks_list = tree_data.get("blocks", [])
        if not blocks_list:
            continue

        # Transform flat list into dicts keyed by ID
        blocks: dict[str, dict] = {}
        spans: dict[str, dict] = {}
        root_id = None

        for block_data in blocks_list:
            block_id = block_data["id"]
            blocks[block_id] = block_data

            # Extract span if nested (from include query)
            span_data = block_data.get("span")
            if span_data and isinstance(span_data, dict):
                span_id = span_data.get("id")
                if span_id:
                    spans[span_id] = span_data

            # Find root block (path == "" or empty)
            path = block_data.get("path", "")
            if path == "" or path is None:
                root_id = block_id

        # Fallback: if no root found by path, look for block with is_root flag
        # or use the block that isn't a child of any other block
        if root_id is None:
            all_children = set()
            for block_data in blocks_list:
                for child_id in block_data.get("children", []):
                    all_children.add(child_id)

            for block_data in blocks_list:
                if block_data["id"] not in all_children:
                    root_id = block_data["id"]
                    break

        if root_id is None:
            continue  # Skip if we can't find a root

        # Reconstruct the block tree
        block = load_block(root_id, blocks, spans)
        results.append(block)

    return results


async def parse_block_trees(trees: list["BlockTree"]) -> list["Block"]:
    """
    Async parser for BlockTree model instances with included blocks relation.

    Use with PgQueryBuilder.parse(parse_block_trees, target="models")

    Args:
        trees: List of BlockTree model instances (with blocks relation loaded)

    Returns:
        List of reconstructed Block instances
    """
    # Convert model instances to dicts for parsing
    tree_dicts = []
    for tree in trees:
        tree_dict = {
            "id": tree.id,
            "artifact_id": getattr(tree, "artifact_id", None),
            "blocks": [],
        }

        # Get blocks from the relation (may be list of BlockModel instances)
        blocks_rel = getattr(tree, "blocks", [])
        for block_model in blocks_rel:
            if hasattr(block_model, "model_dump"):
                block_dict = block_model.model_dump()
            elif isinstance(block_model, dict):
                block_dict = block_model
            else:
                # Assume it's a model instance with attributes
                block_dict = {
                    "id": block_model.id,
                    "span_id": block_model.span_id,
                    "path": block_model.path,
                    "role": block_model.role,
                    "tags": block_model.tags,
                    "styles": block_model.styles,
                    "name": block_model.name,
                    "attrs": block_model.attrs,
                    "children": block_model.children,
                    "block_type": block_model.block_type,
                }
                # Include span if loaded
                if hasattr(block_model, "span") and block_model.span:
                    span = block_model.span
                    block_dict["span"] = {
                        "id": span.id,
                        "content_text": span.content_text,
                        "content_chunks": span.content_chunks,
                        "prefix_text": span.prefix_text,
                        "prefix_chunks": span.prefix_chunks,
                        "postfix_text": span.postfix_text,
                        "postfix_chunks": span.postfix_chunks,
                    }

            tree_dict["blocks"].append(block_dict)

        tree_dicts.append(tree_dict)

    return parse_block_tree_query_result(tree_dicts)


# =============================================================================
# Database Operations
# =============================================================================

async def insert_block(
    block: "Block",
    branch_id: int,
    turn_id: int,
    span_id: int | None = None,
) -> BlockTree:
    """
    Insert Block12 tree into database with Merkle deduplication.

    Uses INSERT ... ON CONFLICT DO NOTHING for automatic deduplication.
    Only new spans/blocks are inserted; existing ones are reused.
    Creates junction table entries for BlockTree ↔ BlockModel relationship.

    Args:
        block: Root block to insert
        branch_id: Branch ID for versioning
        turn_id: Turn ID for versioning
        span_id: Optional execution span ID

    Returns:
        BlockTree model instance
    """
    root_data, all_blocks, all_spans = dump_block(block)
    root_id = root_data["id"]

    async def insert_spans_and_blocks(tx):
        # --- Step 1: Insert spans (content deduplication) ---
        if all_spans:
            span_rows = [
                (
                    span["id"],
                    span["prefix_text"],
                    span["content_text"],
                    span["postfix_text"],
                    json.dumps(span["prefix_chunks"]),
                    json.dumps(span["content_chunks"]),
                    json.dumps(span["postfix_chunks"]),
                    dt.datetime.now(),
                )
                for span in all_spans.values()
            ]
            await tx.executemany(
                """INSERT INTO block_spans
                   (id, prefix_text, content_text, postfix_text,
                    prefix_chunks, content_chunks, postfix_chunks, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (id) DO NOTHING""",
                span_rows
            )

        # --- Step 2: Insert blocks (Merkle deduplication) ---
        if all_blocks:
            block_rows = [
                (
                    blk["id"],
                    blk.get("span_id"),
                    blk.get("role"),
                    blk.get("tags") or [],
                    blk.get("styles") or [],
                    blk.get("name"),
                    blk.get("type_name"),
                    json.dumps(blk.get("attrs") or {}),
                    blk.get("children") or [],
                    blk.get("is_rendered", False),
                    blk.get("block_type", "block"),
                    blk.get("item_name"),
                    blk.get("path", ""),
                    dt.datetime.now(),
                )
                for blk in all_blocks.values()
            ]
            await tx.executemany(
                """INSERT INTO blocks
                   (id, span_id, role, tags, styles, name, type_name, attrs, children, is_rendered, block_type, item_name, path, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                   ON CONFLICT (id) DO NOTHING""",
                block_rows
            )

        return True

    await PGConnectionManager.run_in_transaction(insert_spans_and_blocks)

    # Create tree record
    tree = await BlockTree(span_id=span_id).save(
        branch=branch_id,
        turn=turn_id,
    )

    # --- Step 3: Insert junction table entries ---
    # Build position map: block_id -> position (based on path)
    junction_rows = []
    for position, (block_id, blk_data) in enumerate(all_blocks.items()):
        is_root = (block_id == root_id)
        junction_rows.append((
            tree.id,
            block_id,
            position,
            is_root,
            dt.datetime.now(),
        ))

    if junction_rows:
        async def insert_junction(tx):
            await tx.executemany(
                """INSERT INTO block_tree_blocks
                   (tree_id, block_id, position, is_root, created_at)
                   VALUES ($1, $2, $3, $4, $5)""",
                junction_rows
            )
            return True

        await PGConnectionManager.run_in_transaction(insert_junction)

    return tree


async def get_block_tree(tree_id: int) -> "Block | None":
    """
    Retrieve a Block12 tree by tree ID.

    Uses junction table for efficient batch loading.

    Args:
        tree_id: The tree ID to fetch

    Returns:
        Block instance or None if not found
    """
    tree = await BlockTree.get_or_none(tree_id)
    if tree is None:
        return None

    return await load_block_from_junction(tree_id)


async def load_block_from_junction(tree_id: int) -> "Block | None":
    """
    Load a block tree using the junction table for efficient fetching.

    Fetches all blocks and spans for a tree in minimal queries:
    1. Get all block IDs from junction table
    2. Batch fetch all blocks
    3. Batch fetch all spans
    4. Reconstruct tree in memory

    Args:
        tree_id: The tree ID to load

    Returns:
        Block instance or None if no blocks found
    """
    # Step 1: Get all block IDs and their positions from junction table
    junction_rows = await PGConnectionManager.fetch(
        """SELECT block_id, position, is_root
           FROM block_tree_blocks
           WHERE tree_id = $1
           ORDER BY position""",
        tree_id
    )

    if not junction_rows:
        return None

    block_ids = [row["block_id"] for row in junction_rows]
    root_id = None
    for row in junction_rows:
        if row["is_root"]:
            root_id = row["block_id"]
            break

    # Fallback: if no is_root flag, use first block (position 0)
    if root_id is None and block_ids:
        root_id = block_ids[0]

    # Step 2: Batch fetch all blocks
    block_rows = await PGConnectionManager.fetch(
        """SELECT id, span_id, role, tags, styles, name, type_name, attrs,
                  children, is_rendered, block_type, item_name, path
           FROM blocks WHERE id = ANY($1)""",
        block_ids
    )

    blocks: dict[str, dict] = {}
    span_ids_to_fetch: set[str] = set()

    for row in block_rows:
        block_data = dict(row)
        # Parse JSON strings
        if isinstance(block_data.get("attrs"), str):
            block_data["attrs"] = json.loads(block_data["attrs"])
        if isinstance(block_data.get("children"), str):
            block_data["children"] = json.loads(block_data["children"])
        if isinstance(block_data.get("tags"), str):
            block_data["tags"] = json.loads(block_data["tags"])
        if isinstance(block_data.get("styles"), str):
            block_data["styles"] = json.loads(block_data["styles"])

        blocks[block_data["id"]] = block_data

        if block_data.get("span_id"):
            span_ids_to_fetch.add(block_data["span_id"])

    # Step 3: Batch fetch all spans
    spans: dict[str, dict] = {}
    if span_ids_to_fetch:
        span_rows = await PGConnectionManager.fetch(
            """SELECT id, prefix_text, content_text, postfix_text,
                      prefix_chunks, content_chunks, postfix_chunks
               FROM block_spans WHERE id = ANY($1)""",
            list(span_ids_to_fetch)
        )
        for row in span_rows:
            span_data = dict(row)
            for key in ("prefix_chunks", "content_chunks", "postfix_chunks"):
                if isinstance(span_data.get(key), str):
                    span_data[key] = json.loads(span_data[key])
            spans[span_data["id"]] = span_data

    # Step 4: Reconstruct tree
    return load_block(root_id, blocks, spans)


async def get_blocks(
    artifact_ids: list[int],
    dump_models: bool = False,
) -> dict[int, "Block"]:
    """
    Retrieve Block12 trees by artifact IDs.

    Args:
        artifact_ids: List of artifact IDs to fetch
        dump_models: If True, return model_dump() instead of Block

    Returns:
        Dict mapping artifact_id -> Block (or dict if dump_models=True)
    """
    if not artifact_ids:
        return {}

    trees = await BlockTree.query().where(
        BlockTree.artifact_id.isin(artifact_ids)
    )

    result = {}
    for tree in trees:
        block = await load_block_from_junction(tree.id)
        if block is not None:
            if dump_models:
                result[tree.artifact_id] = block.model_dump()
            else:
                result[tree.artifact_id] = block

    return result


async def get_trees_containing_block(block_id: str) -> list[BlockTree]:
    """
    Find all trees that contain a specific block (reverse lookup).

    Uses the junction table for efficient querying.

    Args:
        block_id: The Merkle hash ID of the block to search for

    Returns:
        List of BlockTree instances containing this block
    """
    tree_ids = await PGConnectionManager.fetch(
        """SELECT DISTINCT tree_id
           FROM block_tree_blocks
           WHERE block_id = $1""",
        block_id
    )

    if not tree_ids:
        return []

    ids = [row["tree_id"] for row in tree_ids]
    trees = await BlockTree.query().where(BlockTree.id.isin(ids))
    return list(trees)


async def get_block_usage_count(block_id: str) -> int:
    """
    Get the number of trees that use a specific block.

    Useful for determining if a block can be garbage collected.

    Args:
        block_id: The Merkle hash ID of the block

    Returns:
        Number of trees referencing this block
    """
    result = await PGConnectionManager.fetchval(
        """SELECT COUNT(DISTINCT tree_id)
           FROM block_tree_blocks
           WHERE block_id = $1""",
        block_id
    )
    return result or 0


# =============================================================================
# High-Level API
# =============================================================================

class BlockLogQuery:
    
    def __init__(self, *args, **kwargs):
        self._query = None
        self._limit = None
        self._offset = None
        self._order_by = "created_at"
        self._filters = {}
        
        
    def __await__(self):
        return self.execute().__await__()
    
    def tail(self, limit: int) -> "BlockLogQuery":
        self._limit = limit
        self._order_by = "-created_at"
        return self
    
    def head(self, limit: int) -> "BlockLogQuery":
        self._limit = limit
        self._order_by = "created_at"
        return self
    
    
    def print(self):
        query = self.build_query()
        query.print()
        return self
    
    def where(
        self, 
        role: str | None = None, 
    ) -> "BlockLogQuery":
        if role is not None:
            self._filters["role"] = role
        if not self._filters:
            raise ValueError("No filters provided")
        return self
    
    
    def build_query(self) -> "PgQueryBuilder[BlockTree]":
        from ..versioning.models import BlockTree, BlockModel, BlockSpan, BlockTreeBlock
        
        block_model_query = BlockModel.query().include(BlockSpan)
        if self._filters:
            block_model_query = block_model_query.where(**self._filters)
            
        block_tree_query = BlockTree.query().include(block_model_query)
        if self._limit is not None:
            block_tree_query = block_tree_query.limit(self._limit)
        if self._offset is not None:
            block_tree_query = block_tree_query.offset(self._offset)
        if self._order_by is not None:
            block_tree_query = block_tree_query.order_by(self._order_by)
        return block_tree_query
    
    async def execute(self) -> list["Block"]:
        query = self.build_query()
        blocks = await query.json_parse(parse_block_tree_query_result)
        return blocks


class BlockLog:
    """High-level API for Block12 storage operations."""

    @classmethod
    async def add(
        cls,
        block: "Block",
        branch_id: int | None = None,
        turn_id: int | None = None,
        span_id: int | None = None,
    ) -> BlockTree:
        """
        Add a block tree to storage.

        Uses current context for branch/turn if not specified.
        """
        from ..versioning.models import Branch, Turn, ExecutionSpan

        if branch_id is None:
            branch_id = Branch.current().id
        if turn_id is None:
            turn_id = Turn.current().id
        if span_id is None:
            curr_span = ExecutionSpan.current_or_none()
            if curr_span is not None:
                span_id = curr_span.id

        return await insert_block(block, branch_id, turn_id, span_id)
    
    @classmethod
    def query(cls, *args, **kwargs) -> "BlockLogQuery":
        return BlockLogQuery(*args, **kwargs)

    @classmethod
    async def get(cls, tree_id: int) -> "Block | None":
        """Get a block tree by ID."""
        return await get_block_tree(tree_id)

    @classmethod
    async def get_by_artifacts(
        cls,
        artifact_ids: list[int],
        dump_models: bool = False,
    ) -> dict[int, "Block"]:
        """Get blocks by artifact IDs."""
        return await get_blocks(artifact_ids, dump_models)

    @classmethod
    async def find_trees_with_block(cls, block_id: str) -> list[BlockTree]:
        """Find all trees containing a specific block."""
        return await get_trees_containing_block(block_id)

    @classmethod
    async def get_block_usage(cls, block_id: str) -> int:
        """Get number of trees using a specific block."""
        return await get_block_usage_count(block_id)
