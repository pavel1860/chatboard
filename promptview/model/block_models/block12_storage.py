"""
Block12 Merkle Tree Storage Layer.

Stores Block12 structures with content-addressed Merkle tree deduplication.
Identical subtrees automatically share storage.

Schema (reuses existing tables):
- block_spans: Content-addressed text+chunks storage
- blocks: Merkle tree nodes
- block_trees: Root references with versioning

Block12 Simplifications vs Block11:
- No prefix/content/postfix separation - just text and chunks
- Chunks store style instead of region type
- Position-based text (start/end) stored per block
"""

import hashlib
import json
import datetime as dt
from typing import TYPE_CHECKING, Any

from ..versioning.models import BlockTree, BlockModel, BlockSpan
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block.block12 import Block, BlockSchema, ChunkMeta


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

def dump_chunks(chunks: list["ChunkMeta"]) -> list[dict]:
    """Serialize ChunkMeta list to list of dicts."""
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
            chunks_data = dump_chunks(blk.chunks)
            span_id = compute_text_hash(text, chunks_data)

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
    """
    from ...block.block12 import Block, BlockSchema, ChunkMeta

    def load_block_data(block_id: str, parent_text: str = "", parent_offset: int = 0) -> "Block":
        """Recursively load a block and its children."""
        block_data = blocks[block_id]

        # Load text and chunks from span
        text = ""
        chunks: list[ChunkMeta] = []
        if block_data.get("span_id"):
            span_data = spans.get(block_data["span_id"])
            if span_data:
                text = span_data.get("content_text", "")
                for chunk_data in span_data.get("content_chunks", []):
                    chunks.append(ChunkMeta(
                        start=chunk_data["start"],
                        end=chunk_data["end"],
                        logprob=chunk_data.get("logprob"),
                        style=chunk_data.get("style"),
                        id=chunk_data.get("id", ""),
                    ))

        # Determine block type and create appropriate class
        block_type = block_data.get("block_type", "block")
        styles = block_data.get("styles") or []
        tags = block_data.get("tags") or []
        role = block_data.get("role")
        attrs = block_data.get("attrs") or {}

        if block_type == "schema":
            blk = BlockSchema(
                name=block_data.get("name"),
                role=role,
                tags=tags,
                style=styles,
                attrs=attrs,
            )
        else:
            blk = Block(
                role=role,
                tags=tags,
                style=styles,
                attrs=attrs,
            )

        # Set text and chunks directly (bypass normal append)
        if text:
            blk._text = text
            blk.start = 0
            blk.end = len(text)
            blk.chunks = chunks

        # Load children recursively
        for child_id in block_data.get("children", []):
            child = load_block_data(child_id)
            # Merge child into parent's text
            if child._text:
                insert_pos = blk.end if blk.is_root else len(blk._text)
                blk._text = blk._text + child._text
                # Remap child positions
                offset = insert_pos
                child.start = offset
                child.end = offset + len(child._text)
                child._text = ""  # Clear local text, now using parent's
            child.parent = blk
            blk.children.append(child)

        return blk

    return load_block_data(root_id)


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

    async def insert_transaction(tx):
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

    await PGConnectionManager.run_in_transaction(insert_transaction)

    # Create tree record
    tree = await BlockTree(root_id=root_id, span_id=span_id).save(
        branch=branch_id,
        turn=turn_id,
    )

    return tree


async def get_block_tree(tree_id: int) -> "Block | None":
    """
    Retrieve a Block12 tree by tree ID.

    Fetches the tree, then recursively loads all blocks and spans.
    """
    tree = await BlockTree.get_or_none(tree_id)
    if tree is None:
        return None

    return await load_block_from_db(tree.root_id)


async def load_block_from_db(root_id: str) -> "Block":
    """
    Load a block tree from database starting from root_id.

    Recursively fetches all blocks and their spans.
    """
    blocks_to_fetch = {root_id}
    blocks: dict[str, dict] = {}
    spans: dict[str, dict] = {}

    # Iteratively fetch blocks (BFS to get all needed blocks)
    while blocks_to_fetch:
        block_ids = list(blocks_to_fetch - set(blocks.keys()))
        if not block_ids:
            break

        rows = await PGConnectionManager.fetch(
            """SELECT id, span_id, role, tags, styles, name, type_name, attrs, children, is_rendered, block_type, item_name, path
               FROM blocks WHERE id = ANY($1)""",
            block_ids
        )

        span_ids_to_fetch = set()
        for row in rows:
            block_data = dict(row)
            # Parse JSON strings to proper types
            if isinstance(block_data.get("attrs"), str):
                block_data["attrs"] = json.loads(block_data["attrs"])
            if isinstance(block_data.get("children"), str):
                block_data["children"] = json.loads(block_data["children"])
            if isinstance(block_data.get("tags"), str):
                block_data["tags"] = json.loads(block_data["tags"])
            if isinstance(block_data.get("styles"), str):
                block_data["styles"] = json.loads(block_data["styles"])
            blocks[block_data["id"]] = block_data

            # Queue children for fetching
            for child_id in block_data.get("children") or []:
                if child_id not in blocks:
                    blocks_to_fetch.add(child_id)

            # Queue span for fetching
            if block_data.get("span_id"):
                span_ids_to_fetch.add(block_data["span_id"])

        # Fetch spans
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

        blocks_to_fetch -= set(blocks.keys())

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
        block = await load_block_from_db(tree.root_id)
        if dump_models:
            result[tree.artifact_id] = block.model_dump()
        else:
            result[tree.artifact_id] = block

    return result


# =============================================================================
# High-Level API
# =============================================================================

class Block12Log:
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
