"""
Unit tests for Block10 - the new block system with shared BlockText.

Tests cover:
1. Basic block creation and properties
2. Wrapper blocks vs content blocks
3. Tree operations (append_child, children, parent)
4. Content operations (append, prepend, operators)
5. Path and traversal
6. Copy and fork operations
7. Rendering
8. BlockSchema and views
"""

import pytest
from chatboard.block.block10 import Block, BlockChunk, BlockText, Span, SpanAnchor
from chatboard.block.block10.block import BlockSchema, BlockListSchema


# =============================================================================
# Basic Block Creation
# =============================================================================

class TestBlockCreation:
    """Tests for basic block creation."""

    def test_block_with_string_content(self):
        """Block with string content creates chunk and span."""
        block = Block("hello world")
        assert block.span is not None
        assert block.span.text() == "hello world"
        assert block.is_wrapper is False

    def test_block_with_empty_string(self):
        """Block with empty string creates empty span, but is_wrapper checks if span is empty."""
        block = Block("")
        assert block.span is not None
        assert block.span.text() == ""
        # Note: is_wrapper returns True for empty spans (span.is_empty)
        # This is the current behavior - empty content blocks are treated as wrappers
        assert block.is_wrapper is True

    def test_wrapper_block_no_content(self):
        """Block without content argument is a wrapper."""
        block = Block()
        assert block.span is None
        assert block.is_wrapper is True

    def test_wrapper_block_with_tags(self):
        """Wrapper block can have tags."""
        block = Block(tags=["container"])
        assert block.is_wrapper is True
        assert "container" in block.tags

    def test_block_with_role(self):
        """Block can have a role."""
        block = Block("content", role="system")
        assert block.role == "system"

    def test_block_with_tags(self):
        """Block can have multiple tags."""
        block = Block("content", tags=["tag1", "tag2"])
        assert block.tags == ["tag1", "tag2"]

    def test_block_with_style(self):
        """Block can have style(s)."""
        block = Block("content", style="xml")
        assert "xml" in block.styles

    def test_block_with_multiple_styles(self):
        """Block style string is split by space."""
        block = Block("content", style="xml markdown")
        assert "xml" in block.styles
        assert "markdown" in block.styles

    def test_block_content_str(self):
        """content_str returns text content."""
        block = Block("hello world")
        assert block.content_str == "hello world"

    def test_wrapper_block_content_str(self):
        """Wrapper block content_str returns empty string."""
        block = Block()
        assert block.content_str == ""

    def test_block_has_own_block_text(self):
        """Each block has its own BlockText by default."""
        block = Block("content")
        assert block.block_text is not None
        assert isinstance(block.block_text, BlockText)


# =============================================================================
# Tree Operations
# =============================================================================

class TestTreeOperations:
    """Tests for block tree operations."""

    def test_append_child_block(self):
        """append_child adds block as child."""
        parent = Block("parent")
        child = Block("child")
        parent.append_child(child)

        assert len(parent.children) == 1
        assert parent.children[0].span.text() == "child"

    def test_append_child_sets_parent(self):
        """append_child sets parent reference."""
        parent = Block("parent")
        child = Block("child")
        parent.append_child(child)

        # Child in parent's children has parent set
        assert parent.children[0].parent is parent

    def test_append_child_copy_true(self):
        """append_child with copy=True creates a copy."""
        parent = Block("parent")
        child = Block("child")
        parent.append_child(child, copy=True)

        # Original child unchanged
        assert child.parent is None
        # Parent's child is a copy
        assert parent.children[0] is not child

    def test_append_child_copy_false(self):
        """append_child with copy=False moves the block."""
        parent = Block("parent")
        child = Block("child")
        parent.append_child(child, copy=False)

        # Child is moved, parent is set
        assert child.parent is parent
        assert parent.children[0] is child

    def test_multiple_children(self):
        """Can append multiple children."""
        parent = Block("parent")
        parent.append_child(Block("child1"))
        parent.append_child(Block("child2"))
        parent.append_child(Block("child3"))

        assert len(parent.children) == 3

    def test_nested_children(self):
        """Children can have their own children."""
        root = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child.append_child(grandchild, copy=False)
        root.append_child(child, copy=False)

        assert len(root.children) == 1
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].span.text() == "grandchild"

    def test_wrapper_block_with_children(self):
        """Wrapper block can have children."""
        wrapper = Block(tags=["wrapper"])
        wrapper.append_child(Block("child1"))
        wrapper.append_child(Block("child2"))

        assert wrapper.is_wrapper
        assert len(wrapper.children) == 2


# =============================================================================
# Operator Tests
# =============================================================================

class TestOperators:
    """Tests for block operators."""

    def test_truediv_appends_child(self):
        """/ operator appends child."""
        parent = Block("parent")
        parent /= Block("child")

        assert len(parent.children) == 1

    def test_truediv_with_string(self):
        """/ operator with string creates child block."""
        parent = Block("parent")
        parent /= "child content"

        assert len(parent.children) == 1
        assert parent.children[0].span.text() == "child content"

    def test_add_appends_content(self):
        """+ operator appends to content span."""
        block = Block("hello")
        block += " world"

        assert "hello" in block.span.text()
        assert "world" in block.span.text()

    def test_and_appends_without_separator(self):
        """& operator appends without separator."""
        block = Block("hello")
        block &= "world"

        text = block.span.text()
        assert "hello" in text
        assert "world" in text

    def test_len_returns_children_count(self):
        """len(block) returns number of children."""
        block = Block("parent")
        assert len(block) == 0

        block /= "child1"
        assert len(block) == 1

        block /= "child2"
        assert len(block) == 2

    def test_bool_true_with_children(self):
        """bool(block) is True if has children."""
        block = Block("parent")
        assert bool(block) is False

        block /= "child"
        assert bool(block) is True


# =============================================================================
# Path and Traversal
# =============================================================================

class TestPathAndTraversal:
    """Tests for path computation and tree traversal."""

    def test_root_path(self):
        """Root block has path [0]."""
        root = Block("root")
        assert root.path.indices == (0,)

    def test_child_path(self):
        """Child has path [0, index]."""
        root = Block("root")
        root /= Block("child0")
        root /= Block("child1")

        assert root.children[0].path.indices == (0, 0)
        assert root.children[1].path.indices == (0, 1)

    def test_grandchild_path(self):
        """Grandchild has path [0, parent_index, index]."""
        root = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child.append_child(grandchild, copy=False)
        root.append_child(child, copy=False)

        assert grandchild.path.indices == (0, 0, 0)

    def test_traverse_visits_all_blocks(self):
        """traverse() visits all blocks in pre-order."""
        root = Block("root")
        root /= Block("child1")
        root /= Block("child2")
        root.children[0] /= Block("grandchild")

        blocks = list(root.traverse())
        assert len(blocks) == 4

    def test_traverse_order_is_preorder(self):
        """traverse() is pre-order (parent before children)."""
        root = Block("root")
        root /= Block("child")
        root.children[0] /= Block("grandchild")

        blocks = list(root.traverse())
        contents = [b.content_str for b in blocks]

        assert contents[0] == "root"
        assert contents[1] == "child"
        assert contents[2] == "grandchild"

    def test_depth_property(self):
        """depth returns correct tree depth (based on path length)."""
        root = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child.append_child(grandchild, copy=False)
        root.append_child(child, copy=False)

        # depth is len(path.indices) - 1, but path always includes root at index 0
        # So root has path (0,) -> depth = 1
        # child has path (0, 0) -> depth = 2
        # grandchild has path (0, 0, 0) -> depth = 3
        # Actually, depth = len(path.indices) which is the path length
        assert root.depth == 1
        assert child.depth == 2
        assert grandchild.depth == 3


# =============================================================================
# Content Property
# =============================================================================

class TestContentProperty:
    """Tests for the content property."""

    def test_content_returns_independent_block(self):
        """content property returns independent block with copied chunks."""
        block = Block("hello world")
        content = block.content

        assert content.span.text() == "hello world"
        # Should be independent
        assert content.block_text is not block.block_text

    def test_content_on_wrapper_raises(self):
        """content property on wrapper raises ValueError."""
        wrapper = Block()
        with pytest.raises(ValueError):
            _ = wrapper.content


# =============================================================================
# Render Tests
# =============================================================================

class TestRender:
    """Tests for block rendering."""

    def test_render_simple_block(self):
        """Simple block renders its content."""
        block = Block("hello world")
        assert block.render() == "hello world"

    def test_render_block_with_children(self):
        """Block with children renders all content."""
        block = Block("parent")
        block /= "child1"
        block /= "child2"

        rendered = block.render()
        assert "parent" in rendered
        assert "child1" in rendered
        assert "child2" in rendered

    def test_render_wrapper_block(self):
        """Wrapper block renders only children."""
        wrapper = Block()
        wrapper /= "child1"
        wrapper /= "child2"

        rendered = wrapper.render()
        assert "child1" in rendered
        assert "child2" in rendered


# =============================================================================
# Copy Tests
# =============================================================================

class TestCopy:
    """Tests for block copying."""

    def test_copy_simple_block(self):
        """copy() creates independent copy."""
        original = Block("content")
        copied = original.copy()

        assert copied.span.text() == "content"
        assert copied is not original
        assert copied.block_text is not original.block_text

    def test_copy_block_with_children(self):
        """copy() copies entire subtree."""
        root = Block("root")
        root /= "child1"
        root /= "child2"

        copied = root.copy()

        assert len(copied.children) == 2
        assert copied.children[0] is not root.children[0]

    def test_copy_preserves_styles_and_tags(self):
        """copy() preserves styles, tags, role."""
        original = Block("content", style="xml", tags=["tag1"], role="system")
        copied = original.copy()

        assert copied.styles == original.styles
        assert copied.tags == original.tags
        assert copied.role == original.role


# =============================================================================
# Shared BlockText Tests
# =============================================================================

class TestSharedBlockText:
    """Tests for shared BlockText behavior."""

    def test_append_child_shares_block_text(self):
        """After append_child, child shares parent's BlockText."""
        parent = Block("parent")
        child = Block("child")

        parent.append_child(child, copy=False)

        assert parent.children[0].block_text is parent.block_text

    def test_all_descendants_share_block_text(self):
        """All descendants share root's BlockText after building tree."""
        root = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child.append_child(grandchild, copy=False)
        root.append_child(child, copy=False)

        for block in root.traverse():
            assert block.block_text is root.block_text

    def test_copy_creates_new_block_text(self):
        """copy() creates new BlockText not shared with original."""
        root = Block("root")
        root /= "child"

        copied = root.copy()

        # Copied tree has its own BlockText
        assert copied.block_text is not root.block_text
        # All copied nodes share the new BlockText
        for block in copied.traverse():
            assert block.block_text is copied.block_text


# =============================================================================
# BlockSchema Tests
# =============================================================================

class TestBlockSchema:
    """Tests for BlockSchema."""

    def test_schema_creation(self):
        """BlockSchema can be created with name."""
        schema = BlockSchema("response")
        assert "response" in schema.tags

    def test_schema_view_creates_child(self):
        """view() creates child schema."""
        schema = BlockSchema("response")
        child = schema.view("thought")

        assert len(schema.children) == 1
        assert "thought" in schema.children[0].tags

    def test_schema_view_returns_child(self):
        """view() returns the created child."""
        schema = BlockSchema("response")
        child = schema.view("thought")

        assert child is schema.children[0]

    def test_schema_view_list_creates_list_schema(self):
        """view_list() creates BlockListSchema child."""
        schema = BlockSchema("response")
        list_schema = schema.view_list("items")

        assert isinstance(list_schema, BlockListSchema)
        # view_list adds "_list" suffix to name, so tag is "items_list"
        assert "items_list" in list_schema.tags

    def test_schema_with_context_manager(self):
        """BlockSchema works with context manager."""
        with BlockSchema("response") as schema:
            schema.view("thought")
            schema.view("answer")

        assert len(schema.children) == 2

    def test_schema_extract_creates_detached_copy(self):
        """extract_schema() creates detached copy."""
        with BlockSchema("response") as schema:
            schema.view("thought")

        extracted = schema.extract_schema()

        assert extracted.parent is None
        assert "response" in extracted.tags


# =============================================================================
# Boundary Tests
# =============================================================================

class TestBoundaries:
    """Tests for block boundary properties."""

    def test_start_chunk_simple_block(self):
        """start_chunk returns first chunk of content."""
        block = Block("hello")
        assert block.start_chunk is not None
        assert "hello" in block.start_chunk.content

    def test_end_chunk_simple_block(self):
        """end_chunk returns last chunk of content."""
        block = Block("hello")
        assert block.end_chunk is not None

    def test_wrapper_start_chunk_from_child(self):
        """Wrapper gets start_chunk from first child."""
        wrapper = Block()
        wrapper /= "child"

        assert wrapper.start_chunk is not None

    def test_get_chunks_returns_all(self):
        """get_chunks() returns all chunks in subtree."""
        block = Block("hello")
        block /= "child"

        chunks = block.get_chunks()
        assert len(chunks) > 0


# =============================================================================
# Context Manager Tests
# =============================================================================

class TestContextManager:
    """Tests for block context manager."""

    def test_block_as_context_manager(self):
        """Block can be used as context manager."""
        with Block("root") as root:
            root /= "child"

        assert len(root.children) == 1

    def test_nested_context_managers(self):
        """Nested context managers build tree."""
        with Block("root") as root:
            with Block("child") as child:
                child /= "grandchild"
            root.append_child(child, copy=False)

        assert len(root.children) == 1
        assert len(root.children[0].children) == 1


# =============================================================================
# Strip Tests
# =============================================================================

class TestStrip:
    """Tests for block strip() method."""

    def test_strip_postfix_newline(self):
        """strip() removes newline from postfix_span."""
        block = Block("content")
        block.add_new_line()  # Adds "\n" to postfix_span

        assert block.postfix_span is not None
        assert block.postfix_span.text() == "\n"

        block.strip()

        # Postfix should be None (all whitespace removed)
        assert block.postfix_span is None
        # Chunk should be removed from BlockText
        for chunk in block.block_text:
            assert chunk.content != "\n"

    def test_strip_postfix_spaces_and_newline(self):
        """strip() removes spaces and newlines from postfix."""
        block = Block("content")
        block.postfix_append("  \n\n  ")

        block.strip()

        assert block.postfix_span is None

    def test_strip_prefix_spaces(self):
        """strip() removes leading spaces from prefix."""
        block = Block("content")
        block.prefix_prepend("   ")

        assert block.prefix_span is not None
        assert block.prefix_span.text() == "   "

        block.strip()

        assert block.prefix_span is None

    def test_strip_preserves_content(self):
        """strip() does not affect main content span."""
        block = Block("  content  ")
        block.add_new_line()

        block.strip()

        # Main content should be unchanged
        assert block.span.text() == "  content  "
        # But postfix should be stripped
        assert block.postfix_span is None

    def test_strip_partial_whitespace_postfix(self):
        """strip() removes only trailing whitespace from postfix with mixed content."""
        block = Block("content")
        block.postfix_append("suffix  \n")

        block.strip()

        assert block.postfix_span is not None
        assert block.postfix_span.text() == "suffix"

    def test_strip_partial_whitespace_prefix(self):
        """strip() removes only leading whitespace from prefix with mixed content."""
        block = Block("content")
        block.prefix_prepend("  \nprefix")

        block.strip()

        assert block.prefix_span is not None
        assert block.prefix_span.text() == "prefix"

    def test_strip_returns_self(self):
        """strip() returns self for method chaining."""
        block = Block("content")
        block.add_new_line()

        result = block.strip()

        assert result is block

    def test_strip_no_prefix_or_postfix(self):
        """strip() on block without prefix/postfix is no-op."""
        block = Block("content")

        block.strip()

        assert block.span.text() == "content"
        assert block.prefix_span is None
        assert block.postfix_span is None

    def test_strip_multiple_calls_idempotent(self):
        """Multiple strip() calls are idempotent."""
        block = Block("content")
        block.add_new_line()

        block.strip()
        block.strip()  # Second call should be no-op

        assert block.postfix_span is None

    def test_strip_removes_chunk_from_block_text(self):
        """strip() removes whitespace chunks from underlying BlockText."""
        block = Block("content")
        block.add_new_line()

        initial_length = len(block.block_text)

        block.strip()

        # BlockText should have fewer chunks after stripping
        assert len(block.block_text) < initial_length


# =============================================================================
# BlockText Consistency Tests
# =============================================================================

class TestBlockTextConsistency:
    """Tests for BlockText chunk linkage consistency."""

    def test_nested_call_preserves_chunk_linkage(self):
        """Using __call__ for nested blocks preserves chunk linkage."""
        with Block("hello") as blk:
            blk /= "world"
            with blk("subheader") as sub:
                sub /= "this is the subheader"

        # Verify all chunks are linked
        chunks = list(blk.block_text)
        assert len(chunks) > 0, "BlockText should not be empty"

        # Verify linkage: each chunk's next.prev should point back
        for chunk in chunks:
            if chunk.next is not None:
                assert chunk.next.prev is chunk, "Forward/backward linkage broken"
            if chunk.prev is not None:
                assert chunk.prev.next is chunk, "Backward/forward linkage broken"

        # Verify head and tail
        assert blk.block_text.head is not None
        assert blk.block_text.tail is not None
        assert blk.block_text.head.prev is None
        assert blk.block_text.tail.next is None

    def test_nested_call_chunks_contain_all_content(self):
        """Nested blocks via __call__ have all content in BlockText."""
        with Block("hello") as blk:
            blk /= "world"
            with blk("subheader") as sub:
                sub /= "this is the subheader"

        # Collect all chunk contents
        chunk_contents = [c.content for c in blk.block_text]

        # Verify expected content is present
        assert "hello" in chunk_contents
        assert "world" in chunk_contents
        assert "subheader" in chunk_contents
        assert "this is the subheader" in chunk_contents

    def test_deeply_nested_blocks_share_block_text(self):
        """Deeply nested blocks all share the same BlockText."""
        with Block("root") as root:
            with root("level1") as l1:
                with l1("level2") as l2:
                    with l2("level3") as l3:
                        l3 /= "deep content"

        # All blocks should share the same BlockText
        assert l1.block_text is root.block_text
        assert l2.block_text is root.block_text
        assert l3.block_text is root.block_text

        # All content should be in the shared BlockText
        chunk_contents = [c.content for c in root.block_text]
        assert "root" in chunk_contents
        assert "level1" in chunk_contents
        assert "level2" in chunk_contents
        assert "level3" in chunk_contents
        assert "deep content" in chunk_contents

    def test_append_child_with_shared_block_text_no_duplication(self):
        """append_child with shared BlockText doesn't duplicate chunks."""
        parent = Block("parent")

        # Create child with parent's block_text (simulates __call__ behavior)
        child = Block("child", block_text=parent.block_text)

        chunks_before = len(parent.block_text)
        parent.append_child(child, copy=False)
        chunks_after = len(parent.block_text)

        # Should only add newline, not duplicate the child chunk
        # The child chunk was already in block_text when child was created
        assert chunks_after == chunks_before + 1  # +1 for newline

    def test_block_text_length_matches_iteration(self):
        """BlockText length matches actual chunk count from iteration."""
        with Block("hello") as blk:
            blk /= "world"
            with blk("nested") as nested:
                nested /= "content"

        chunks_by_iteration = list(blk.block_text)
        assert len(blk.block_text) == len(chunks_by_iteration)

    def test_all_span_chunks_in_block_text(self):
        """All chunks referenced by spans are in the BlockText."""
        with Block("hello") as blk:
            blk /= "world"
            with blk("subheader") as sub:
                sub /= "this is the subheader"

        block_text_ids = {c.id for c in blk.block_text}

        # Check all blocks' span chunks are in BlockText
        for block in blk.traverse():
            if block.span is not None:
                assert block.span.start.chunk.id in block_text_ids, \
                    f"Span start chunk not in BlockText: {block.span.start.chunk.content}"
                assert block.span.end.chunk.id in block_text_ids, \
                    f"Span end chunk not in BlockText: {block.span.end.chunk.content}"
