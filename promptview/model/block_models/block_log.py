"""
Block10 database storage layer.

Stores Block10 structures with block-level content deduplication.
Each block's content (chunks) is stored separately and deduplicated by hash.
At load time, chunks are assembled into a shared BlockText.

Schema:
- blocks: Content deduplication (id=hash, content, json_content with chunks)
- block_signatures: Content + styling (id=hash, block_id, styles, tags, role)
- block_nodes: Tree structure (tree_id, path, signature_id)
- block_trees: Root reference (id, artifact_id)
"""

import hashlib
import json
import datetime as dt
from typing import Any, Dict, List, Literal, TYPE_CHECKING

from ..versioning.models import Artifact, Branch, DataFlowNode
from ..versioning.models import BlockTree, BlockNode, BlockModel, BlockSignature, ExecutionSpan, TurnStatus
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block.block10 import Block, BlockChunk, BlockText, Span


def block_content_hash(chunks: list[dict]) -> str:
    """
    Hash block content by its chunks.

    Args:
        chunks: List of chunk dicts with 'content' and optional 'logprob'

    Returns:
        SHA256 hash of the chunks
    """
    data = json.dumps(chunks, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def signature_hash(
    block_id: str,
    styles: list[str] | None,
    role: str | None,
    tags: list[str] | None,
) -> str:
    """
    Create a unique hash for a block signature (content + styling).
    """
    data = {
        "block_id": block_id,
        "styles": sorted(styles) if styles else None,
        "role": role,
        "tags": sorted(tags) if tags else None,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def dump_chunk(chunk: "BlockChunk") -> dict:
    """Serialize a single chunk."""
    return {
        "content": chunk.content,
        "logprob": chunk.logprob,
    }


def dump_span_chunks(block: "Block", span: "Span | None") -> list[dict]:
    """
    Extract and serialize chunks covered by a span.

    Args:
        block: The block owning the BlockText
        span: The span to extract (None returns empty list)

    Returns:
        List of serialized chunk dicts
    """
    if span is None or span.start.chunk is None:
        return []

    chunks = []
    current = span.start.chunk
    while current is not None:
        # For first chunk, may start at offset > 0
        if current is span.start.chunk and span.start.offset > 0:
            # Partial first chunk
            if current is span.end.chunk:
                # Single chunk, partial on both ends
                chunks.append({
                    "content": current.content[span.start.offset:span.end.offset],
                    "logprob": current.logprob,
                })
            else:
                chunks.append({
                    "content": current.content[span.start.offset:],
                    "logprob": current.logprob,
                })
        elif current is span.end.chunk:
            # Last chunk (possibly partial)
            chunks.append({
                "content": current.content[:span.end.offset],
                "logprob": current.logprob,
            })
        else:
            # Full chunk
            chunks.append(dump_chunk(current))

        if current is span.end.chunk:
            break
        current = current.next

    return chunks


def dump_block_content(block: "Block") -> dict:
    """
    Dump a single block's content (not children).

    Returns dict with:
    - content: Full text content (for quick queries)
    - json_content: Structured chunk data for reconstruction
    """
    # Get chunks for content span
    content_chunks = dump_span_chunks(block, block.span)
    prefix_chunks = dump_span_chunks(block, block.prefix_span)
    postfix_chunks = dump_span_chunks(block, block.postfix_span)

    # Build full text for content field
    content_text = ""
    if block.span and not block.is_wrapper:
        content_text = block.span.text()

    return {
        "content": content_text,
        "json_content": {
            "chunks": content_chunks,
            "prefix_chunks": prefix_chunks,
            "postfix_chunks": postfix_chunks,
        }
    }


def dump_block(block: "Block") -> list[dict]:
    """
    Dump Block10 tree to list of node dicts for storage.

    Each node contains:
    - path: Tree path as string "0.1.2"
    - content: Full text content
    - json_content: Chunk data for reconstruction
    - styles: List of style names
    - tags: List of tags
    - role: Role string or None

    Args:
        block: Root block to dump

    Returns:
        List of node dicts in traversal order
    """
    dumps = []

    for blk in block.traverse():
        # Get path as string
        path_str = ".".join(str(i) for i in blk.path.indices)

        # Dump content
        content_data = dump_block_content(blk)

        dumps.append({
            "path": path_str,
            "content": content_data["content"],
            "json_content": content_data["json_content"],
            "styles": blk.styles,
            "tags": blk.tags,
            "role": blk.role,
        })

    return dumps


def load_block_dump(dumps: list[dict], artifact_id: int | None = None) -> "Block":
    """
    Load Block10 tree from storage dumps.

    Reconstructs blocks with a SHARED BlockText - all blocks reference
    the same underlying chunk storage.

    Args:
        dumps: List of node dicts from storage (with nested signature/block data)
        artifact_id: Optional artifact ID to attach to root block

    Returns:
        Reconstructed Block10 tree with shared BlockText
    """
    from ...block.block10 import Block, BlockChunk, BlockText, Span, SpanAnchor

    # Single shared BlockText for entire tree
    shared_block_text = BlockText()

    # Build blocks with their content
    block_lookup: dict[str, Block] = {}

    for dump in dumps:
        # Extract data from nested structure (matches query result format)
        if "signature" in dump:
            # Query result format: dump -> signature -> block
            signature = dump["signature"]
            block_data = signature["block"]
            json_content = block_data["json_content"]
            styles = signature.get("styles") or []
            tags = signature.get("tags") or []
            role = signature.get("role")
        else:
            # Direct format (from dump_block)
            json_content = dump["json_content"]
            styles = dump.get("styles") or []
            tags = dump.get("tags") or []
            role = dump.get("role")

        path_str = dump["path"]

        # Load chunks into shared BlockText
        content_chunks = []
        for chunk_data in json_content.get("chunks", []):
            chunk = BlockChunk(
                content=chunk_data["content"],
                logprob=chunk_data.get("logprob"),
            )
            shared_block_text.append(chunk)
            content_chunks.append(chunk)

        prefix_chunks = []
        for chunk_data in json_content.get("prefix_chunks", []):
            chunk = BlockChunk(
                content=chunk_data["content"],
                logprob=chunk_data.get("logprob"),
            )
            shared_block_text.append(chunk)
            prefix_chunks.append(chunk)

        postfix_chunks = []
        for chunk_data in json_content.get("postfix_chunks", []):
            chunk = BlockChunk(
                content=chunk_data["content"],
                logprob=chunk_data.get("logprob"),
            )
            shared_block_text.append(chunk)
            postfix_chunks.append(chunk)

        # Create block with shared BlockText (no content - we set span manually)
        blk = Block(
            block_text=shared_block_text,
            styles=styles,
            tags=tags,
            role=role,
            _skip_content=True,
        )

        # Set spans from loaded chunks
        if content_chunks:
            blk.span = Span.from_chunks(content_chunks)
        else:
            blk.span = None  # Wrapper block

        if prefix_chunks:
            blk.prefix_span = Span.from_chunks(prefix_chunks)

        if postfix_chunks:
            blk.postfix_span = Span.from_chunks(postfix_chunks)

        block_lookup[path_str] = blk

    # Reconstruct tree structure
    root_blk = block_lookup.get("0")
    if root_blk is None:
        raise ValueError("No root block found at path '0'")

    # Sort paths by depth to ensure parents are processed before children
    sorted_paths = sorted(block_lookup.keys(), key=lambda p: len(p.split(".")))

    for path_str in sorted_paths:
        if path_str == "0":
            continue  # Skip root

        blk = block_lookup[path_str]

        # Find parent path
        parts = path_str.split(".")
        parent_path = ".".join(parts[:-1])

        if parent_path in block_lookup:
            parent = block_lookup[parent_path]
            # Use low-level append to avoid BlockText manipulation
            # (chunks are already in correct order in shared BlockText)
            parent.children.append(blk)
            blk.parent = parent

    return root_blk


async def insert_block(block: "Block", branch_id: int, turn_id: int, span_id: int | None = None) -> BlockTree:
    """
    Insert Block10 tree into database with deduplication.

    Uses existing table structure:
    - blocks: Content deduplication by hash
    - block_signatures: Content + styling deduplication
    - block_nodes: Tree structure
    - block_trees: Root reference with versioning

    Args:
        block: Root block to insert
        branch_id: Branch ID for versioning
        turn_id: Turn ID for versioning
        span_id: Optional execution span ID

    Returns:
        BlockTree model instance
    """
    nodes = dump_block(block)

    async def insert_transaction(tx, tree_id: int):
        # --- Step 1: Insert blocks (content deduplication) ---
        block_rows = []
        seen_blocks = set()

        for node in nodes:
            # Hash by chunk content for deduplication
            blk_id = block_content_hash(node["json_content"].get("chunks", []))

            if blk_id not in seen_blocks:
                block_rows.append((
                    blk_id,
                    node["content"],
                    json.dumps(node["json_content"])
                ))
                seen_blocks.add(blk_id)

        if block_rows:
            await tx.executemany(
                "INSERT INTO blocks (id, content, json_content) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING",
                block_rows
            )

        # --- Step 2: Insert signatures (content + styling deduplication) ---
        signature_rows = []
        seen_signatures = set()
        node_signatures = {}  # Map node index -> signature_id

        for idx, node in enumerate(nodes):
            blk_id = block_content_hash(node["json_content"].get("chunks", []))
            sig_id = signature_hash(
                blk_id,
                node["styles"],
                node["role"],
                node["tags"],
            )
            node_signatures[idx] = sig_id

            if sig_id not in seen_signatures:
                styles_array = node["styles"] if node["styles"] else []
                tags_array = node["tags"] if node["tags"] else []
                signature_rows.append((
                    sig_id,
                    blk_id,
                    styles_array,
                    node["role"],
                    tags_array,
                    json.dumps({}),  # No attrs in Block10
                    dt.datetime.now()
                ))
                seen_signatures.add(sig_id)

        if signature_rows:
            await tx.executemany(
                "INSERT INTO block_signatures (id, block_id, styles, role, tags, attrs, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO NOTHING",
                signature_rows
            )

        # --- Step 3: Insert nodes (tree structure) ---
        node_rows = []
        for idx, node in enumerate(nodes):
            sig_id = node_signatures[idx]
            node_rows.append((
                tree_id,
                node["path"],
                sig_id
            ))

        if node_rows:
            await tx.executemany(
                "INSERT INTO block_nodes (tree_id, path, signature_id) VALUES ($1, $2::ltree, $3)",
                node_rows
            )

        return tree_id

    # Create tree record first
    block_tree = await BlockTree().save()

    async def tx_wrapper(tx):
        return await insert_transaction(tx, block_tree.id)

    await PGConnectionManager.run_in_transaction(tx_wrapper)
    return block_tree


async def get_blocks(artifact_ids: list[int], dump_models: bool = False) -> dict[int, "Block"]:
    """
    Retrieve Block10 trees by artifact IDs.

    Args:
        artifact_ids: List of artifact IDs to fetch
        dump_models: If True, return dict representation instead of Block

    Returns:
        Dict mapping artifact_id -> Block (or dict if dump_models=True)
    """
    if not artifact_ids:
        return {}

    block_trees = await (
        BlockTree.query(alias="bt")
        .select("*")
        .include(
            BlockNode.query(alias="bn").select("*").include(
                BlockSignature.query(alias="bs").select("*").include(
                    BlockModel.query(alias="bm").select("*")
                )
            )
        )
        .where(lambda b: b.artifact_id.isin(artifact_ids))
        .order_by("-artifact_id")
        .json()
    )

    blocks_lookup = {}
    for tree in block_trees:
        art_id = tree["artifact_id"]
        if not tree.get("nodes"):
            continue

        block = load_block_dump(tree["nodes"], artifact_id=art_id)
        blocks_lookup[art_id] = block.model_dump() if dump_models else block

    return blocks_lookup


class BlockLogQuery:

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
        self.include_roles = None
        self.exclude_roles = None

    def __await__(self):
        return self.execute().__await__()

    async def execute(self) -> List["Block"]:
        query = self._build_block_query()
        artifact_ids = []
        if self.span_name:
            span_query = self._build_span_query(self.span_name)
            spans = await span_query
            artifact_ids = [a.id for span in spans for sv in span.data for a in sv.artifacts]
            if not artifact_ids:
                return []

        if artifact_ids:
            query = query.where(lambda x: x.artifact_id.isin(artifact_ids))

        # Get raw JSON results - use execute_json() for dict output
        tree_dicts = await query.json()

        # Convert each tree to a Block
        results = []
        for tree in tree_dicts:
            if not tree.get('nodes'):
                continue
            block = load_block_dump(tree['nodes'])
            results.append(block)

        if self.include_roles:
            results = [b for b in results if b.role in self.include_roles]
        if self.exclude_roles:
            results = [b for b in results if b.role not in self.exclude_roles]
        return results

    async def json(self) -> List[dict]:
        query = self._build_block_query()
        artifact_ids = []
        if self.span_name:
            span_query = self._build_span_query(self.span_name)
            spans = await span_query
            artifact_ids = [a.id for span in spans for sv in span.data for a in sv.artifacts]
        if artifact_ids:
            query = query.where(lambda x: x.artifact_id.isin(artifact_ids))
        results = await query.json()
        return results


    def _build_block_query(self):
        """Build query using schema: BlockTree -> BlockNode -> BlockSignature -> BlockModel"""
        return BlockTree.query(
            alias="bt",
            limit=self.limit,
            offset=self._offset,
            direction=self.direction,
            statuses=self.statuses
        ).include(
            BlockNode.query(alias="bn").include(
                BlockSignature.query(alias="bs").include(BlockModel)
            )
        ).order_by("created_at")

    def _build_span_query(self, name: str):
        return ExecutionSpan.query().include(
            DataFlowNode.query().include(
                Artifact
            )
        ).where(name=name)



    def where(self, span: str | None = None):
        if span:
            self.span_name = span
        return self

    def role(self, exclude: set[str] | None = None, include: set[str] | None = None):
        self.include_roles = include
        self.exclude_roles = exclude
        return self

    def status(self, statuses: list[TurnStatus]):
        self.statuses = statuses
        return self

    def last(self, limit: int):
        self.limit = limit
        return self

    def offset(self, offset: int):
        self._offset = offset
        return self

    def all(self):
        self.statuses = []
        return self

    def span(self, span: str):
        self.span_name = span
        return self

    def print(self):
        query = self._build_block_query()
        return query.print()


class BlockLog:

    def __init__(self):
        self.query = None


    @classmethod
    def last(cls, limit: int) -> BlockLogQuery:
        return BlockLogQuery(limit=limit)

    @classmethod
    def span(cls, span: str) -> BlockLogQuery:
        return BlockLogQuery(span_name=span)



    @classmethod
    async def add(cls, block: "Block", branch_id: int | None = None, turn_id: int | None = None, span_id: int | None = None):
        from ..versioning.models import Turn, Branch
        if branch_id is None:
            branch_id = Branch.current().id
        if turn_id is None:
            turn_id = Turn.current().id
        if span_id is None:
            curr_span = ExecutionSpan.current_or_none()
            if curr_span is not None:
                span_id = curr_span.id
        return await insert_block(block, branch_id, turn_id, span_id)
