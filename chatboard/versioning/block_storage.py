"""
Simplified Block Storage Layer.

Stores entire Block trees with content-addressed deduplication.
Uses VersionedModel for automatic artifact tracking and versioning.

Key simplifications from block12_storage.py:
- Single table instead of 4 (no spans, no junction, no separate blocks)
- Hash computed on entire block tree, not Merkle nodes
- Uses existing Block.model_dump() / Block.model_load() for serialization
- Leverages VersionedModel artifact system for span linking

Usage:
    # Store a block
    block_model = await BlockLog.add(block)

    # Query blocks
    blocks = await BlockLog.query().where(role="assistant").tail(10)

    # Get by hash (dedup check)
    existing = await BlockLog.get_by_hash(content_hash)
"""

import hashlib
import json
import datetime as dt
from typing import TYPE_CHECKING, List, Literal, Type, Self

from .models import VersionedModel, Branch, Turn, TurnStatus, Artifact
from ..model.fields import KeyField, ModelField, RelationField
from ..model.base.types import ArtifactKind

if TYPE_CHECKING:
    from ..block.block12 import Block, BlockList
    from ..model.sql2.pg_query_builder import PgQueryBuilder


# =============================================================================
# Hash Computation
# =============================================================================

def compute_block_hash(block: "Block") -> str:
    """
    Compute content hash for entire block tree.

    Uses model_dump() for comprehensive serialization, excluding
    transient fields like 'id' which are randomly generated.

    Args:
        block: Block tree to hash

    Returns:
        SHA256 hex digest
    """
    # Get serialized form
    data = block.model_dump()

    # Remove transient/random fields that shouldn't affect identity
    def strip_ids(d: dict) -> dict:
        """Recursively remove 'id' fields from dict."""
        result = {}
        for k, v in d.items():
            if k == "id":
                continue  # Skip random IDs
            if k == "path":
                continue  # Skip path (position-dependent)
            if isinstance(v, dict):
                result[k] = strip_ids(v)
            elif isinstance(v, list):
                result[k] = [
                    strip_ids(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    clean_data = strip_ids(data)
    encoded = json.dumps(clean_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# =============================================================================
# Model Definition
# =============================================================================

class BlockModel(VersionedModel):
    """
    Versioned block storage with content-addressed deduplication.

    Each record stores an entire block tree as JSONB. The content_hash
    enables deduplication - identical blocks share the same record and
    are linked via artifacts.

    Attributes:
        content_hash: SHA256 of block content (for dedup lookup)
        data: Complete block tree as JSONB (Block.model_dump())
        role: Block's role for quick filtering
        span: Name of the execution span that created this block
        created_at: When this block was first stored
    """
    _namespace_name: str = "blocks"
    _artifact_kind: ArtifactKind = "block"

    id: int = KeyField(primary_key=True)
    content_hash: str = ModelField(index=True)  # For dedup lookup
    data: dict = ModelField()  # Block.model_dump()
    role: str | None = ModelField(default=None, index=True)
    span: str | None = ModelField(default=None, index=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)

    def to_block(self) -> "Block":
        """Reconstruct Block from stored data."""
        from ..block.block12 import Block
        return Block.model_load(self.data)

    @classmethod
    def from_block(cls, block: "Block", span: str | None = None) -> "BlockModel":
        """Create BlockModel instance from Block (not saved yet)."""
        return cls(
            content_hash=compute_block_hash(block),
            data=block.model_dump(),
            role=block.role if hasattr(block, '_role') and block._role else None,
            span=span,
        )


# =============================================================================
# Storage Operations
# =============================================================================

async def store_block(
    block: "Block",
    branch_id: int | None = None,
    turn_id: int | None = None,
    span_id: int | None = None,
    span: str | None = None,
    deduplicate: bool = True,
) -> BlockModel:
    """
    Store a block tree with optional deduplication.

    If deduplicate=True (default), checks for existing block with same
    content_hash and returns it instead of creating a new record.

    The block is transformed before storage to ensure consistent
    formatting (markdown headers, XML tags, etc.) is persisted.

    Args:
        block: Block tree to store
        branch_id: Branch ID (uses context if not provided)
        turn_id: Turn ID (uses context if not provided)
        span_id: Execution span ID (uses context if not provided)
        span: Execution span name (uses context if not provided)
        deduplicate: If True, reuse existing blocks with same hash

    Returns:
        BlockModel instance (new or existing)
    """
    from .dataflow_models import ExecutionSpan
    # Transform block before storage
    # block = block.transform()

    content_hash = compute_block_hash(block)

    # Check for existing block with same hash
    if deduplicate:
        existing = await BlockModel.query().where(
            content_hash=content_hash
        ).first()
        if existing:
            return existing

    # Resolve context if not provided
    if branch_id is None:
        branch_id = Branch.current().id
    if turn_id is None:
        turn_id = Turn.current().id

    curr_span = ExecutionSpan.current_or_none()
    if span_id is None and curr_span is not None:
        span_id = curr_span.id
    if span is None and curr_span is not None:
        span = curr_span.name

    # Create and save new block
    block_model = BlockModel.from_block(block, span=span)
    await block_model.save(branch=branch_id, turn=turn_id)

    return block_model


async def get_block(block_id: int) -> "Block | None":
    """
    Get a block by its ID.

    Args:
        block_id: The BlockModel ID

    Returns:
        Block instance or None if not found
    """
    block_model = await BlockModel.get_or_none(block_id)
    if block_model is None:
        return None
    return block_model.to_block()


async def get_block_by_hash(content_hash: str) -> "Block | None":
    """
    Get a block by its content hash.

    Args:
        content_hash: SHA256 hash of block content

    Returns:
        Block instance or None if not found
    """
    block_model = await BlockModel.query().where(
        content_hash=content_hash
    ).first()
    if block_model is None:
        return None
    return block_model.to_block()


# =============================================================================
# Query Builder
# =============================================================================

class BlockLogQuery:
    """
    Fluent query builder for BlockModel.

    Usage:
        blocks = await BlockLog.query().where(role="assistant").tail(10)
    """

    def __init__(self):
        self._limit: int | None = None
        self._offset: int | None = None
        self._order_by: str = "created_at"
        self._reverse: bool = False
        self._filters: dict = {}
        self._should_extract: bool = True
        self._branch: Branch | int | None = None
        self._statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED]

    def __await__(self):
        return self.execute().__await__()

    def tail(self, limit: int) -> "BlockLogQuery":
        """Get last N blocks (most recent first, then reversed)."""
        self._limit = limit
        self._order_by = "-created_at"
        self._reverse = True
        return self

    def head(self, limit: int) -> "BlockLogQuery":
        """Get first N blocks (oldest first)."""
        self._limit = limit
        self._order_by = "created_at"
        self._reverse = False
        return self

    def where(
        self,
        role: str | list[str] | None = None,
        span: str | list[str] | None = None,
        content_hash: str | None = None,
    ) -> "BlockLogQuery":
        """Add filters to query."""
        if role is not None:
            self._filters["role"] = role
        if span is not None:
            self._filters["span"] = span
        if content_hash is not None:
            self._filters["content_hash"] = content_hash
        return self

    def span(self, span: str) -> "BlockLogQuery":
        """Filter by execution span name."""
        self._filters["span"] = span
        return self

    def branch(self, branch: Branch | int) -> "BlockLogQuery":
        """Filter by branch."""
        self._branch = branch
        return self

    def extract(self, should_extract: bool = True) -> "BlockLogQuery":
        """Whether to extract blocks (remove style wrappers)."""
        self._should_extract = should_extract
        return self

    def limit(self, n: int) -> "BlockLogQuery":
        """Limit number of results."""
        self._limit = n
        return self

    def offset(self, n: int) -> "BlockLogQuery":
        """Skip first N results."""
        self._offset = n
        return self

    def build_query(self) -> "PgQueryBuilder[BlockModel]":
        """Build the underlying query."""
        query = BlockModel.query(
            branch=self._branch,
            statuses=self._statuses,
        )

        # Apply filters
        if "role" in self._filters:
            role = self._filters["role"]
            if isinstance(role, list):
                query = query.where(BlockModel.role.isin(role))
            else:
                query = query.where(role=role)

        if "content_hash" in self._filters:
            query = query.where(content_hash=self._filters["content_hash"])

        if "span" in self._filters:
            span = self._filters["span"]
            if isinstance(span, list):
                query = query.where(BlockModel.span.isin(span))
            else:
                query = query.where(span=span)

        # Apply ordering and pagination
        if self._order_by:
            query = query.order_by(self._order_by)
        if self._limit:
            query = query.limit(self._limit)
        if self._offset:
            query = query.offset(self._offset)

        return query

    async def execute(self) -> "BlockList":
        """Execute query and return BlockList."""
        from ..block.block12 import BlockList

        query = self.build_query()
        block_models = await query.execute()

        # Convert to Block instances
        blocks = [bm.to_block() for bm in block_models]

        if self._reverse:
            blocks = list(reversed(blocks))

        if self._should_extract:
            blocks = [b.extract() for b in blocks]

        return BlockList(blocks)

    async def first(self) -> "Block | None":
        """Get first matching block."""
        self._limit = 1
        results = await self.execute()
        return results[0] if results else None

    async def count(self) -> int:
        """Count matching blocks."""
        query = self.build_query()
        return await query.count()


# =============================================================================
# High-Level API
# =============================================================================

class BlockLog:
    """
    High-level API for block storage operations.

    Usage:
        # Add a block
        block_model = await BlockLog.add(block)

        # Query blocks
        blocks = await BlockLog.query().where(role="assistant").tail(10)

        # Get by ID
        block = await BlockLog.get(block_id)

        # Check if block exists (by hash)
        existing = await BlockLog.get_by_hash(content_hash)
    """

    @classmethod
    async def add(
        cls,
        block: "Block",
        branch_id: int | None = None,
        turn_id: int | None = None,
        span_id: int | None = None,
        span: str | None = None,
        deduplicate: bool = True,
    ) -> BlockModel:
        """
        Add a block tree to storage.

        Uses current context for branch/turn/span if not specified.
        Deduplicates by content hash by default.
        """
        return await store_block(
            block,
            branch_id=branch_id,
            turn_id=turn_id,
            span_id=span_id,
            span=span,
            deduplicate=deduplicate,
        )

    @classmethod
    def query(cls) -> BlockLogQuery:
        """Create a query builder for blocks."""
        return BlockLogQuery()

    @classmethod
    async def get(cls, block_id: int) -> "Block | None":
        """Get a block by ID."""
        return await get_block(block_id)

    @classmethod
    async def get_by_hash(cls, content_hash: str) -> "Block | None":
        """Get a block by content hash."""
        return await get_block_by_hash(content_hash)

    @classmethod
    async def exists(cls, block: "Block") -> bool:
        """Check if a block with the same content already exists."""
        content_hash = compute_block_hash(block)
        existing = await BlockModel.query().where(
            content_hash=content_hash
        ).first()
        return existing is not None

    @classmethod
    async def get_or_create(
        cls,
        block: "Block",
        branch_id: int | None = None,
        turn_id: int | None = None,
        span_id: int | None = None,
        span: str | None = None,
    ) -> tuple[BlockModel, bool]:
        """
        Get existing block or create new one.

        Returns:
            Tuple of (BlockModel, created) where created is True if new
        """
        content_hash = compute_block_hash(block)
        existing = await BlockModel.query().where(
            content_hash=content_hash
        ).first()

        if existing:
            return existing, False

        block_model = await store_block(
            block,
            branch_id=branch_id,
            turn_id=turn_id,
            span_id=span_id,
            span=span,
            deduplicate=False,  # Already checked
        )
        return block_model, True
