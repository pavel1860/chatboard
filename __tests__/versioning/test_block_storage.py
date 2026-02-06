"""
Unit tests for Block Storage layer.

Tests cover:
1. compute_block_hash - content hashing
2. BlockModel - model creation and serialization
3. BlockLog.add - storing blocks with deduplication
4. BlockLog.query - querying blocks
5. Round-trip tests - store and retrieve
6. Block type preservation - BlockSchema, BlockListSchema preserved
7. Deduplication - same content reuses records
"""

import os
import pytest
import pytest_asyncio
from pydantic import BaseModel, Field

# Set test database URL
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/promptview_test"

from promptview.block.block12 import Block, BlockSchema, BlockListSchema
from promptview.versioning import BlockLog, StoredBlockModel
from promptview.versioning.block_storage import compute_block_hash, BlockModel
from promptview.versioning.models import Branch, Turn
from promptview.prompt.context import Context
from promptview.model.namespace_manager2 import NamespaceManager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="module")
async def setup_db():
    """Ensure we start with a clean DB schema for the tests."""
    await NamespaceManager.initialize_clean()
    yield
    NamespaceManager.drop_all_tables()


@pytest_asyncio.fixture(scope="module")
async def main_branch(setup_db):
    """Create and return the main branch."""
    branch = await Branch.get_main()
    return branch


# =============================================================================
# compute_block_hash Tests
# =============================================================================

class TestComputeBlockHash:
    """Tests for compute_block_hash function."""

    def test_same_content_same_hash(self):
        """Same content produces same hash."""
        block1 = Block("hello world")
        block2 = Block("hello world")

        assert compute_block_hash(block1) == compute_block_hash(block2)

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        block1 = Block("hello")
        block2 = Block("world")

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_ignores_id(self):
        """Hash ignores randomly generated IDs."""
        block1 = Block("content")
        block2 = Block("content")
        # IDs are different by default
        assert block1.id != block2.id
        # But hashes should be the same
        assert compute_block_hash(block1) == compute_block_hash(block2)

    def test_hash_includes_role(self):
        """Hash includes role in computation."""
        block1 = Block("content", role="user")
        block2 = Block("content", role="assistant")

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_includes_tags(self):
        """Hash includes tags in computation."""
        block1 = Block("content", tags=["tag1"])
        block2 = Block("content", tags=["tag2"])

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_includes_style(self):
        """Hash includes style in computation."""
        block1 = Block("content", style="xml")
        block2 = Block("content", style="md")

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_includes_children(self):
        """Hash includes children content."""
        with Block("root") as block1:
            block1 /= "child1"

        with Block("root") as block2:
            block2 /= "child2"

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_with_nested_children(self):
        """Hash handles deeply nested children."""
        with Block("root") as block1:
            with block1("parent") as parent:
                parent /= "child"

        with Block("root") as block2:
            with block2("parent") as parent:
                parent /= "different"

        assert compute_block_hash(block1) != compute_block_hash(block2)

    def test_hash_deterministic(self):
        """Hash is deterministic across multiple calls."""
        block = Block("content", tags=["test"], role="user")

        hash1 = compute_block_hash(block)
        hash2 = compute_block_hash(block)
        hash3 = compute_block_hash(block)

        assert hash1 == hash2 == hash3


# =============================================================================
# BlockModel Tests
# =============================================================================

class TestBlockModel:
    """Tests for BlockModel class."""

    def test_from_block_creates_model(self):
        """BlockModel.from_block creates model from Block."""
        block = Block("hello world", role="user")
        model = BlockModel.from_block(block)

        assert model.content_hash == compute_block_hash(block)
        assert model.role == "user"
        assert "text" in model.data

    def test_to_block_reconstructs(self):
        """BlockModel.to_block reconstructs the block."""
        original = Block("hello world", tags=["test"])
        model = BlockModel.from_block(original)
        restored = model.to_block()

        assert restored.text == original.text
        assert restored.tags == original.tags

    def test_from_block_preserves_children(self):
        """BlockModel.from_block preserves children in data."""
        with Block("root") as block:
            block /= "child1"
            block /= "child2"

        model = BlockModel.from_block(block)

        assert len(model.data["children"]) == 2


# =============================================================================
# BlockLog.add Tests (requires database)
# =============================================================================

class TestBlockLogAdd:
    """Tests for BlockLog.add function."""

    @pytest.mark.asyncio
    async def test_add_simple_block(self, main_branch):
        """BlockLog.add stores a simple block."""
        block = Block("hello world")
        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(block)

            assert stored.id is not None
            assert stored.content_hash is not None

    @pytest.mark.asyncio
    async def test_add_block_with_role(self, main_branch):
        """BlockLog.add preserves role."""
        block = Block("content", role="assistant")
        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(block)

            assert stored.role == "assistant"

    @pytest.mark.asyncio
    async def test_add_block_with_children(self, main_branch):
        """BlockLog.add stores blocks with children."""
        with Block("root") as block:
            block /= "child1"
            block /= "child2"

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(block)
            restored = stored.to_block()

            assert len(restored.children) == 2

    @pytest.mark.asyncio
    async def test_add_deduplicates_same_content(self, main_branch):
        """BlockLog.add deduplicates blocks with same content."""
        block1 = Block("identical content for dedup test")
        block2 = Block("identical content for dedup test")

        async with Context().start_turn() as ctx:
            stored1 = await BlockLog.add(block1)
            stored2 = await BlockLog.add(block2)

            # Should return same record
            assert stored1.id == stored2.id
            assert stored1.content_hash == stored2.content_hash

    @pytest.mark.asyncio
    async def test_add_no_dedup_different_content(self, main_branch):
        """BlockLog.add creates new records for different content."""
        block1 = Block("content 1 unique")
        block2 = Block("content 2 unique")

        async with Context().start_turn() as ctx:
            stored1 = await BlockLog.add(block1)
            stored2 = await BlockLog.add(block2)

            assert stored1.id != stored2.id

    @pytest.mark.asyncio
    async def test_add_deduplicate_false(self, main_branch):
        """BlockLog.add with deduplicate=False creates new records."""
        block1 = Block("identical content no dedup")
        block2 = Block("identical content no dedup")

        async with Context().start_turn() as ctx:
            stored1 = await BlockLog.add(block1, deduplicate=False)
            stored2 = await BlockLog.add(block2, deduplicate=False)

            # Should create separate records
            assert stored1.id != stored2.id


# =============================================================================
# BlockLog.query Tests (requires database)
# =============================================================================

class TestBlockLogQuery:
    """Tests for BlockLog.query function."""

    @pytest.mark.asyncio
    async def test_query_tail(self, main_branch):
        """BlockLog.query().tail() returns most recent blocks."""
        async with Context().start_turn() as ctx:
            await BlockLog.add(Block("first query test"))
            await BlockLog.add(Block("second query test"))
            await BlockLog.add(Block("third query test"))

            blocks = await BlockLog.query().tail(2)

            assert len(blocks) == 2

    @pytest.mark.asyncio
    async def test_query_head(self, main_branch):
        """BlockLog.query().head() returns oldest blocks."""
        async with Context().start_turn() as ctx:
            await BlockLog.add(Block("first head test"))
            await BlockLog.add(Block("second head test"))
            await BlockLog.add(Block("third head test"))

            blocks = await BlockLog.query().head(2)

            assert len(blocks) == 2

    @pytest.mark.asyncio
    async def test_query_where_role(self, main_branch):
        """BlockLog.query().where(role=...) filters by role."""
        async with Context().start_turn() as ctx:
            await BlockLog.add(Block("user msg role test", role="user"))
            await BlockLog.add(Block("assistant msg role test", role="assistant"))
            await BlockLog.add(Block("user msg 2 role test", role="user"))

            user_blocks = await BlockLog.query().where(role="user").tail(10)
            assistant_blocks = await BlockLog.query().where(role="assistant").tail(10)

            assert len(user_blocks) >= 2
            assert len(assistant_blocks) >= 1

    @pytest.mark.asyncio
    async def test_query_count(self, main_branch):
        """BlockLog.query().count() returns correct count."""
        async with Context().start_turn() as ctx:
            initial_count = await BlockLog.query().count()

            await BlockLog.add(Block("one count test"))
            await BlockLog.add(Block("two count test"))
            await BlockLog.add(Block("three count test"))

            count = await BlockLog.query().count()

            assert count == initial_count + 3

    @pytest.mark.asyncio
    async def test_query_first(self, main_branch):
        """BlockLog.query().first() returns first matching block."""
        async with Context().start_turn() as ctx:
            await BlockLog.add(Block("first test block"))
            await BlockLog.add(Block("second test block"))

            block = await BlockLog.query().first()

            assert block is not None

    @pytest.mark.asyncio
    async def test_query_empty_result(self, main_branch):
        """BlockLog.query() returns empty list when no matches."""
        async with Context().start_turn() as ctx:
            blocks = await BlockLog.query().where(role="nonexistent_role_xyz").tail(10)

            assert len(blocks) == 0


# =============================================================================
# BlockLog.get Tests (requires database)
# =============================================================================

class TestBlockLogGet:
    """Tests for BlockLog.get and get_by_hash."""

    @pytest.mark.asyncio
    async def test_get_by_id(self, main_branch):
        """BlockLog.get retrieves block by ID."""
        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(Block("content for get test"))
            retrieved = await BlockLog.get(stored.id)

            assert retrieved is not None
            assert "content for get test" in retrieved.text

    @pytest.mark.asyncio
    async def test_get_nonexistent_id(self, main_branch):
        """BlockLog.get returns None for nonexistent ID."""
        async with Context().start_turn() as ctx:
            retrieved = await BlockLog.get(99999)

            assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_by_hash(self, main_branch):
        """BlockLog.get_by_hash retrieves block by content hash."""
        block = Block("unique content for hash test")
        content_hash = compute_block_hash(block)

        async with Context().start_turn() as ctx:
            await BlockLog.add(block)
            retrieved = await BlockLog.get_by_hash(content_hash)

            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_exists(self, main_branch):
        """BlockLog.exists checks if block exists."""
        block = Block("content to check exists")

        async with Context().start_turn() as ctx:
            assert not await BlockLog.exists(block)

            await BlockLog.add(block)

            assert await BlockLog.exists(block)


# =============================================================================
# Round-trip Tests (requires database)
# =============================================================================

class TestRoundTrip:
    """Tests for store -> retrieve round-trip."""

    @pytest.mark.asyncio
    async def test_roundtrip_simple_block(self, main_branch):
        """Simple block survives round-trip."""
        original = Block("hello world roundtrip")

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert "hello world roundtrip" in restored.text

    @pytest.mark.asyncio
    async def test_roundtrip_block_with_metadata(self, main_branch):
        """Block with metadata survives round-trip."""
        original = Block("content meta", role="user", tags=["important"], style="xml")

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert restored.role == original.role
            assert restored.tags == original.tags
            assert restored.style == original.style

    @pytest.mark.asyncio
    async def test_roundtrip_block_with_children(self, main_branch):
        """Block with children survives round-trip."""
        with Block("root roundtrip") as original:
            original /= "child1 roundtrip"
            original /= "child2 roundtrip"

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert len(restored.children) == len(original.children)

    @pytest.mark.asyncio
    async def test_roundtrip_nested_tree(self, main_branch):
        """Nested tree structure survives round-trip."""
        with Block("root nested") as original:
            with original("level1 nested") as level1:
                level1 /= "level2 nested"

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            # Count all nodes
            original_count = len(list(original.iter_depth_first()))
            restored_count = len(list(restored.iter_depth_first()))

            assert original_count == restored_count

    @pytest.mark.asyncio
    async def test_roundtrip_via_query(self, main_branch):
        """Block retrieved via query matches original."""
        original = Block("query roundtrip test", role="system")

        async with Context().start_turn() as ctx:
            await BlockLog.add(original)

            blocks = await BlockLog.query().where(role="system").tail(1)
            restored = blocks[0]

            assert "query roundtrip test" in restored.text
            assert restored.role == original.role


# =============================================================================
# Block Type Preservation Tests (requires database)
# =============================================================================

class TestBlockTypePreservation:
    """Tests for preserving BlockSchema, BlockListSchema types."""

    @pytest.mark.asyncio
    async def test_preserve_block_schema(self, main_branch):
        """BlockSchema type is preserved through storage."""
        original = BlockSchema("test_schema_preserve", style="xml")

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert type(restored).__name__ == "BlockSchema"

    @pytest.mark.asyncio
    async def test_preserve_block_list_schema(self, main_branch):
        """BlockListSchema type is preserved through storage."""
        original = BlockListSchema("item_preserve", name="items_preserve")

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert type(restored).__name__ == "BlockListSchema"

    @pytest.mark.asyncio
    async def test_preserve_nested_schema_types(self, main_branch):
        """Nested schema types are preserved."""
        with BlockSchema("root_nested_preserve", style="xml") as original:
            original.append_child(BlockSchema("child_nested_preserve", style="xml"))

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert type(restored).__name__ == "BlockSchema"
            assert type(restored.children[0]).__name__ == "BlockSchema"

    @pytest.mark.asyncio
    async def test_preserve_mixed_types(self, main_branch):
        """Mixed Block and BlockSchema types are preserved."""
        with BlockSchema("root_mixed", style="xml") as original:
            original /= "plain block child mixed"
            original.append_child(BlockSchema("schema_child_mixed", style="xml"))

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            assert type(restored).__name__ == "BlockSchema"
            # First child is plain Block
            assert type(restored.children[0]).__name__ == "Block"
            # Second child is BlockSchema
            assert type(restored.children[1]).__name__ == "BlockSchema"

    @pytest.mark.asyncio
    async def test_preserve_pydantic_schema(self, main_branch):
        """Schema with registered Pydantic models preserves structure."""

        class TestToolPreserve(BaseModel):
            """Test tool description"""
            param: str = Field(description="test parameter")

        with Block(tags=["tools_preserve"]) as original:
            with original.schema(tags=["schema_preserve"]) as schema:
                with schema.view_list("tool_preserve", key="name") as tools:
                    tools.register(TestToolPreserve)

        async with Context().start_turn() as ctx:
            stored = await BlockLog.add(original)
            restored = stored.to_block()

            # Check that schema types are preserved in nested structure
            assert type(restored.children[0]).__name__ == "BlockSchema"
            assert type(restored.children[0].children[0]).__name__ == "BlockListSchema"


# =============================================================================
# BlockLog.get_or_create Tests (requires database)
# =============================================================================

class TestBlockLogGetOrCreate:
    """Tests for BlockLog.get_or_create."""

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, main_branch):
        """get_or_create creates new block when not exists."""
        block = Block("new content get_or_create")

        async with Context().start_turn() as ctx:
            stored, created = await BlockLog.get_or_create(block)

            assert created is True
            assert stored.id is not None

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, main_branch):
        """get_or_create returns existing block."""
        block = Block("existing content get_or_create")

        async with Context().start_turn() as ctx:
            stored1, created1 = await BlockLog.get_or_create(block)
            stored2, created2 = await BlockLog.get_or_create(block)

            assert created1 is True
            assert created2 is False
            assert stored1.id == stored2.id
