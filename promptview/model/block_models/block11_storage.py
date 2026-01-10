"""
Block11 Merkle Tree Storage Layer.

Stores Block11 structures with content-addressed Merkle tree deduplication.
Identical subtrees automatically share storage.

Schema:
- block11_spans: Content-addressed spans (id=hash of content+chunks)
- block11_blocks: Merkle tree nodes (id=hash of span+metadata+children)
- block11_trees: Root references with versioning

Deduplication:
- Same span content → same span record
- Same span + metadata + children → same block record
- Identical subtrees automatically share storage
"""

import hashlib
import json
import datetime as dt
from typing import TYPE_CHECKING, Any, List, Literal

from ..versioning.models import Artifact, Branch, Turn, VersionedModel, TurnStatus, ExecutionSpan, BlockTree, BlockModel, BlockSpan
from ..model3 import Model
from ..fields import KeyField, ModelField, RelationField
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block.block11 import Block, BlockChunk, Span, BlockSchema


# =============================================================================
# Database Models
# =============================================================================

# class BlockSpan(Model):
#     """
#     Content-addressed span storage.

#     Deduplicated by hash of (prefix + content + postfix + chunks).
#     Stores both text (for queries) and structured chunks (for reconstruction).
#     """
#     _namespace_name: str = "block11_spans"

#     id: str = KeyField(primary_key=True)  # SHA256 hash
#     prefix_text: str = ModelField(default="")
#     content_text: str = ModelField(default="")
#     postfix_text: str = ModelField(default="")
#     prefix_chunks: dict = ModelField(default_factory=list)  # [{content, logprob}, ...]
#     content_chunks: dict = ModelField(default_factory=list)
#     postfix_chunks: dict = ModelField(default_factory=list)
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)


# class BlockModel(Model):
#     """
#     Merkle tree node - content-addressed by span + metadata + children.

#     The id is a hash that includes children's IDs, so identical subtrees
#     automatically have identical hashes and share storage.
#     """
#     _namespace_name: str = "block11_blocks"

#     id: str = KeyField(primary_key=True)  # Merkle hash
#     span_id: str | None = ModelField(default=None, foreign_key=True, foreign_cls=BlockSpan)  # FK to spans
#     role: str | None = ModelField(default=None)
#     tags: list[str] = ModelField(default_factory=list)
#     styles: list[str] = ModelField(default_factory=list)
#     name: str | None = ModelField(default=None)  # For BlockSchema
#     type_name: str | None = ModelField(default=None)  # For BlockSchema
#     attrs: dict = ModelField(default_factory=dict)
#     children: list[str] = ModelField(default_factory=list)  # Ordered block IDs
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

#     span: BlockSpan | None = RelationField(primary_key="span_id", foreign_key="id")


# class BlockTree(VersionedModel):
#     """
#     Root reference for a block tree, linked to versioning system.
#     """
#     _namespace_name: str = "block11_trees"
#     _artifact_kind = "block"

#     id: int = KeyField(primary_key=True)
#     root_id: str = ModelField(foreign_key=True, foreign_cls=BlockModel)  # FK to blocks
#     span_id: int | None = ModelField(default=None, foreign_key=True, foreign_cls=ExecutionSpan)  # Execution span
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

#     root: BlockModel | None = RelationField(primary_key="root_id", foreign_key="id")


# =============================================================================
# Hash Computation
# =============================================================================

def compute_span_hash(
    prefix_text: str,
    content_text: str,
    postfix_text: str,
    prefix_chunks: list[dict],
    content_chunks: list[dict],
    postfix_chunks: list[dict],
) -> str:
    """
    Compute content-addressed hash for a span.

    Includes chunk data (with logprobs) for precise deduplication.
    """
    data = {
        "prefix_text": prefix_text,
        "content_text": content_text,
        "postfix_text": postfix_text,
        "prefix_chunks": prefix_chunks,
        "content_chunks": content_chunks,
        "postfix_chunks": postfix_chunks,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_block_hash(
    span_id: str | None,
    role: str | None,
    tags: list[str],
    styles: list[str],
    name: str | None,
    type_name: str | None,
    attrs: dict,
    children: list[str],  # Child block IDs (already hashed)
    is_rendered: bool = False,
    block_type: str = "block",  # "block", "schema", "list", "list_schema"
    path: str = "",  # e.g., "0.1.2"
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
        "type_name": type_name,
        "attrs": attrs,
        "children": children,  # Order matters!
        "is_rendered": is_rendered,
        "block_type": block_type,
        "path": path,
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# Serialization (Dump)
# =============================================================================

def dump_chunks(chunks: list["BlockChunk"]) -> list[dict]:
    """Serialize chunks to list of dicts using BlockChunk.model_dump()."""
    return [c.model_dump() for c in chunks]


def dump_span(span: "Span") -> dict:
    """
    Serialize a Span to storage format.

    Uses Span.model_dump() internally and adds Merkle hash ID.
    Returns dict with text fields and chunk arrays.
    """
    # Use Span's model_dump for serialization
    data = span.model_dump(include_chunks=True)

    # Compute content-addressed hash
    span_id = compute_span_hash(
        data["prefix"],
        data["content"],
        data["postfix"],
        data["prefix_chunks"],
        data["content_chunks"],
        data["postfix_chunks"],
    )

    # Return storage format (rename keys to match DB schema)
    return {
        "id": span_id,
        "prefix_text": data["prefix"],
        "content_text": data["content"],
        "postfix_text": data["postfix"],
        "prefix_chunks": data["prefix_chunks"],
        "content_chunks": data["content_chunks"],
        "postfix_chunks": data["postfix_chunks"],
    }


def dump_block(block: "Block") -> tuple[dict, dict[str, dict], dict[str, dict]]:
    """
    Serialize a Block tree to storage format using Merkle hashing.

    Returns:
        (root_block_data, all_blocks_dict, all_spans_dict)

    The blocks dict is keyed by Merkle hash, spans dict by content hash.
    Duplicate subtrees will naturally have same hash and overwrite.
    """
    from ...block.block11 import BlockSchema, BlockList, BlockListSchema

    all_spans: dict[str, dict] = {}
    all_blocks: dict[str, dict] = {}

    def get_block_type(blk: "Block") -> str:
        """Determine the block type string."""
        if isinstance(blk, BlockListSchema):
            return "list_schema"
        elif isinstance(blk, BlockList):
            return "list"
        elif isinstance(blk, BlockSchema):
            return "schema"
        else:
            return "block"

    def process_block(blk: "Block", path: str = "") -> str:
        """Process a block and return its Merkle hash."""
        # Process children first (post-order for Merkle)
        child_ids = []
        for i, child in enumerate(blk.children):
            child_path = f"{path}.{i}" if path else str(i)
            child_id = process_block(child, child_path)
            child_ids.append(child_id)

        # Process span
        span_id = None
        if blk.span and not blk.span.is_empty:
            span_data = dump_span(blk.span)
            span_id = span_data["id"]
            all_spans[span_id] = span_data

        # Determine block type
        block_type = get_block_type(blk)

        # Get schema-specific fields
        name = None
        type_name = None
        attrs = {}
        item_name = None
        if isinstance(blk, BlockListSchema):
            name = blk.name
            type_name = str(blk.type) if blk.type else None
            attrs = blk.attrs or {}
            item_name = blk.item_name
        elif isinstance(blk, BlockSchema):
            name = blk.name
            type_name = str(blk.type) if blk.type else None
            attrs = blk.attrs or {}

        # Get mutator state
        is_rendered = blk.mutator.is_rendered if blk.mutator else False

        # Compute Merkle hash
        block_id = compute_block_hash(
            span_id=span_id,
            role=blk.role,
            tags=blk.tags or [],
            styles=blk._style or [],
            name=name,
            type_name=type_name,
            attrs=attrs,
            children=child_ids,
            is_rendered=is_rendered,
            block_type=block_type,
            path=path,
        )

        # Store block data
        all_blocks[block_id] = {
            "id": block_id,
            "span_id": span_id,
            "role": blk.role,
            "tags": blk.tags or [],
            "styles": blk._style or [],
            "name": name,
            "type_name": type_name,
            "attrs": attrs,
            "children": child_ids,
            "is_rendered": is_rendered,
            "block_type": block_type,
            "item_name": item_name,  # For BlockListSchema
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
    Reconstruct a Block tree from storage format.

    Creates a shared BlockText for all blocks in the tree.
    Uses Span.model_validate() for span reconstruction.
    Restores the correct Mutator based on stored styles.
    Restores the correct block class based on block_type.
    """
    from ...block.block11 import Block, BlockSchema, BlockList, BlockListSchema, Span, BlockText
    from ...block.block11.mutator_meta import MutatorMeta

    # Single shared BlockText for the entire tree
    shared_block_text = BlockText()

    def load_span_data(span_data: dict) -> Span:
        """Load a span from storage data using Span.model_validate()."""
        # Convert storage format keys to model_dump format
        model_data = {
            "prefix": span_data.get("prefix_text", ""),
            "content": span_data.get("content_text", ""),
            "postfix": span_data.get("postfix_text", ""),
            "prefix_chunks": span_data.get("prefix_chunks", []),
            "content_chunks": span_data.get("content_chunks", []),
            "postfix_chunks": span_data.get("postfix_chunks", []),
        }
        span = Span.model_validate(model_data)
        shared_block_text.append(span)
        return span

    def load_block_data(block_id: str) -> Block:
        """Recursively load a block and its children."""
        block_data = blocks[block_id]

        # Load span if present
        span = None
        if block_data.get("span_id"):
            span_data = spans.get(block_data["span_id"])
            if span_data:
                span = load_span_data(span_data)

        # Resolve mutator from styles
        styles = block_data.get("styles") or []
        mutator_config = MutatorMeta.resolve(styles)
        mutator = mutator_config.mutator(None)  # Create mutator, will attach to block

        # Determine block type
        block_type = block_data.get("block_type", "block")

        # Create appropriate block class based on block_type
        if block_type == "list_schema":
            blk = BlockListSchema(
                name=block_data.get("name"),
                item_name=block_data.get("item_name"),
                tags=block_data.get("tags"),
                style=styles,
                attrs=block_data.get("attrs"),
                block_text=shared_block_text,
                mutator=mutator,
            )
        elif block_type == "list":
            blk = BlockList(
                role=block_data.get("role"),
                tags=block_data.get("tags"),
                style=styles,
                block_text=shared_block_text,
                mutator=mutator,
            )
        elif block_type == "schema":
            blk = BlockSchema(
                name=block_data.get("name"),
                tags=block_data.get("tags"),
                style=styles,
                attrs=block_data.get("attrs"),
                block_text=shared_block_text,
                mutator=mutator,
            )
        else:  # "block"
            blk = Block(
                role=block_data.get("role"),
                tags=block_data.get("tags"),
                style=styles,
                block_text=shared_block_text,
                mutator=mutator,
            )

        # Set span if we loaded one
        if span:
            blk.span = span

        # Load children recursively
        for child_id in block_data.get("children", []):
            child = load_block_data(child_id)
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
    Insert Block11 tree into database with Merkle deduplication.

    Uses INSERT ... ON CONFLICT DO NOTHING for automatic deduplication.
    Only new spans/blocks are inserted; existing ones are reused.

    Args:
        block: Root block to insert
        branch_id: Branch ID for versioning
        turn_id: Turn ID for versioning
        span_id: Optional execution span ID

    Returns:
        Block11Tree model instance
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

    # First, insert spans and blocks in a transaction
    async def tx_wrapper(tx):
        return await insert_transaction(tx)

    await PGConnectionManager.run_in_transaction(tx_wrapper)

    # Now create tree record (blocks exist, FK will pass)
    tree = await BlockTree(root_id=root_id, span_id=span_id).save(
        branch=branch_id,
        turn=turn_id,
    )

    return tree


async def get_block_tree(tree_id: int) -> "Block | None":
    """
    Retrieve a Block11 tree by tree ID.

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
    # Collect all block IDs we need to fetch
    blocks_to_fetch = {root_id}
    blocks: dict[str, dict] = {}
    spans: dict[str, dict] = {}

    # Iteratively fetch blocks (BFS to get all needed blocks)
    while blocks_to_fetch:
        # Fetch blocks we don't have yet
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
                # Parse JSON strings to lists
                for key in ("prefix_chunks", "content_chunks", "postfix_chunks"):
                    if isinstance(span_data.get(key), str):
                        span_data[key] = json.loads(span_data[key])
                spans[span_data["id"]] = span_data

        # Remove fetched blocks from queue
        blocks_to_fetch -= set(blocks.keys())

    return load_block(root_id, blocks, spans)


async def get_blocks_by_artifact_ids(artifact_ids: list[int]) -> dict[int, "Block"]:
    """
    Retrieve Block11 trees by artifact IDs.

    Returns dict mapping artifact_id -> Block.
    """
    if not artifact_ids:
        return {}

    trees = await BlockTree.query().where(
        BlockTree.artifact_id.isin(artifact_ids)
    )

    result = {}
    for tree in trees:
        block = await load_block_from_db(tree.root_id)
        result[tree.artifact_id] = block

    return result


async def get_blocks(
    artifact_ids: list[int],
    dump_models: bool = False,
    include_branch_turn: bool = False,
) -> dict[int, "Block"]:
    """
    Retrieve Block11 trees by artifact IDs.

    Compatible interface with old block_log.get_blocks.

    Args:
        artifact_ids: List of artifact IDs to fetch
        dump_models: If True, return model_dump() instead of Block
        include_branch_turn: Ignored (for compatibility)

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
# Query Interface
# =============================================================================

class BlockLogQuery:
    """Query interface for Block11 trees."""

    def __init__(
        self,
        limit: int | None = None,
        offset: int | None = None,
        direction: Literal["asc", "desc"] = "desc",
        span_name: str | None = None,
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
    ):
        self.limit = limit or 5
        self._offset = offset
        self.direction = direction
        self.span_name = span_name
        self.statuses = statuses
        self.include_roles: set[str] | None = None
        self.exclude_roles: set[str] | None = None

    def __await__(self):
        return self.execute().__await__()

    async def execute(self) -> list["Block"]:
        """Execute query and return Block instances."""
        query = BlockTree.query(
            limit=self.limit,
            offset=self._offset,
            direction=self.direction,
            statuses=self.statuses,
        )

        if self.span_name:
            # Filter by execution span name
            spans = await ExecutionSpan.query().where(name=self.span_name)
            span_ids = [s.id for s in spans]
            if not span_ids:
                return []
            query = query.where(BlockTree.span_id.isin(span_ids))

        trees = await query

        results = []
        for tree in trees:
            block = await load_block_from_db(tree.root_id)

            # Apply role filters
            if self.include_roles and block.role not in self.include_roles:
                continue
            if self.exclude_roles and block.role in self.exclude_roles:
                continue

            results.append(block)

        return results

    def role(self, include: set[str] | None = None, exclude: set[str] | None = None):
        """Filter by role."""
        self.include_roles = include
        self.exclude_roles = exclude
        return self

    def status(self, statuses: list[TurnStatus]):
        """Filter by turn status."""
        self.statuses = statuses
        return self

    def last(self, limit: int):
        """Limit results."""
        self.limit = limit
        return self

    def offset(self, offset: int):
        """Offset results."""
        self._offset = offset
        return self

    def span(self, span_name: str):
        """Filter by execution span name."""
        self.span_name = span_name
        return self


class BlockLog:
    """High-level API for Block11 storage operations."""

    @classmethod
    def last(cls, limit: int) -> BlockLogQuery:
        """Get last N block trees."""
        return BlockLogQuery(limit=limit)

    @classmethod
    def span(cls, span_name: str) -> BlockLogQuery:
        """Get block trees by execution span."""
        return BlockLogQuery(span_name=span_name)

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
        if branch_id is None:
            branch_id = Branch.current().id
        if turn_id is None:
            turn_id = Turn.current().id
        if span_id is None:
            curr_span = ExecutionSpan.current_or_none()
            if curr_span is not None:
                span_id = curr_span.id

        return await insert_block(block, branch_id, turn_id, span_id)
