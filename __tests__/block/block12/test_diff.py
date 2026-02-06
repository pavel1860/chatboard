"""Tests for Block diff functionality."""
import pytest
from promptview.block.block12 import (
    Block,
    diff_blocks,
    BlockDiff,
    NodeDiff,
    FieldChange,
    get_text_diff,
    get_inline_diff,
    format_diff_tree,
)


class TestIdenticalBlocks:
    """Tests for comparing identical blocks."""

    def test_identical_simple_blocks(self):
        block_a = Block("Hello World")
        block_b = Block("Hello World")

        diff = diff_blocks(block_a, block_b)

        assert diff.is_identical
        assert diff.change_count == 0
        assert diff.hash_a == diff.hash_b

    def test_identical_blocks_with_children(self):
        with Block("Root") as block_a:
            block_a /= "Child 1"
            block_a /= "Child 2"

        with Block("Root") as block_b:
            block_b /= "Child 1"
            block_b /= "Child 2"

        diff = diff_blocks(block_a, block_b)

        assert diff.is_identical

    def test_identical_blocks_with_role(self):
        block_a = Block("Content", role="system")
        block_b = Block("Content", role="system")

        diff = diff_blocks(block_a, block_b)

        assert diff.is_identical

    def test_identical_blocks_with_tags(self):
        block_a = Block("Content", tags=["important"])
        block_b = Block("Content", tags=["important"])

        diff = diff_blocks(block_a, block_b)

        assert diff.is_identical


class TestTextChanges:
    """Tests for text content changes."""

    def test_text_change_detected(self):
        block_a = Block("Hello")
        block_b = Block("World")

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.has_content_changes
        assert diff.root.status == "modified"
        assert diff.root.text_change == ("Hello", "World")

    def test_text_change_in_child(self):
        with Block("Root") as block_a:
            block_a /= "Original"

        with Block("Root") as block_b:
            block_b /= "Modified"

        diff = diff_blocks(block_a, block_b)

        assert diff.has_content_changes
        assert len(list(diff.iter_changes())) > 0

        # Find the child that changed
        for node in diff.iter_changes():
            if node.text_change and "Original" in node.text_change[0]:
                assert node.text_change == ("Original", "Modified")
                break


class TestRoleChanges:
    """Tests for role changes."""

    def test_role_change_detected(self):
        block_a = Block("Content", role="user")
        block_b = Block("Content", role="assistant")

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.root.status == "modified"
        assert diff.root.role_change == ("user", "assistant")

    def test_role_added(self):
        block_a = Block("Content")
        block_b = Block("Content", role="system")

        diff = diff_blocks(block_a, block_b)

        assert diff.root.role_change == (None, "system")

    def test_role_removed(self):
        block_a = Block("Content", role="system")
        block_b = Block("Content")

        diff = diff_blocks(block_a, block_b)

        assert diff.root.role_change == ("system", None)


class TestTagChanges:
    """Tests for tag changes."""

    def test_tags_change_detected(self):
        block_a = Block("Content", tags=["old"])
        block_b = Block("Content", tags=["new"])

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.root.tags_change == (["old"], ["new"])

    def test_tags_added(self):
        block_a = Block("Content")
        block_b = Block("Content", tags=["important"])

        diff = diff_blocks(block_a, block_b)

        assert diff.root.tags_change is not None
        assert diff.root.tags_change[1] == ["important"]


class TestStyleChanges:
    """Tests for style changes."""

    def test_style_change_detected(self):
        block_a = Block("Content", style="xml")
        block_b = Block("Content", style="md")

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.root.style_change is not None


class TestAttrChanges:
    """Tests for attrs changes."""

    def test_attrs_change_detected(self):
        block_a = Block("Content", attrs={"key": "value1"})
        block_b = Block("Content", attrs={"key": "value2"})

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.root.attrs_change == ({"key": "value1"}, {"key": "value2"})


class TestChildChanges:
    """Tests for child additions and removals."""

    def test_child_added(self):
        block_a = Block("Root")

        with Block("Root") as block_b:
            block_b /= "New Child"

        diff = diff_blocks(block_a, block_b)

        assert diff.has_structural_changes
        added_paths = diff.get_added_paths()
        assert len(added_paths) == 1

    def test_child_removed(self):
        with Block("Root") as block_a:
            block_a /= "Child to Remove"

        block_b = Block("Root")

        diff = diff_blocks(block_a, block_b)

        assert diff.has_structural_changes
        removed_paths = diff.get_removed_paths()
        assert len(removed_paths) == 1

    def test_multiple_children_added(self):
        block_a = Block("Root")

        with Block("Root") as block_b:
            block_b /= "Child 1"
            block_b /= "Child 2"
            block_b /= "Child 3"

        diff = diff_blocks(block_a, block_b)

        added_paths = diff.get_added_paths()
        assert len(added_paths) == 3


class TestNestedChanges:
    """Tests for nested structure changes."""

    def test_nested_text_change(self):
        with Block("Root") as block_a:
            with block_a("Parent") as parent_a:
                parent_a /= "Original"

        with Block("Root") as block_b:
            with block_b("Parent") as parent_b:
                parent_b /= "Modified"

        diff = diff_blocks(block_a, block_b)

        assert diff.has_content_changes

        # Should have a change deep in the tree
        changes = list(diff.iter_changes())
        assert len(changes) > 0

    def test_deeply_nested_addition(self):
        with Block("Root") as block_a:
            with block_a("Level 1") as l1:
                l1 /= "Existing"

        with Block("Root") as block_b:
            with block_b("Level 1") as l1:
                l1 /= "Existing"
                l1 /= "New"

        diff = diff_blocks(block_a, block_b)

        assert diff.has_structural_changes
        added = diff.get_added_paths()
        assert len(added) == 1


class TestDiffIterators:
    """Tests for diff iteration methods."""

    def test_iter_all_includes_unchanged(self):
        with Block("Root") as block_a:
            block_a /= "Child 1"
            block_a /= "Child 2"

        with Block("Root") as block_b:
            block_b /= "Child 1"
            block_b /= "Modified"

        diff = diff_blocks(block_a, block_b)

        all_nodes = list(diff.iter_all())
        changed_nodes = list(diff.iter_changes())

        assert len(all_nodes) > len(changed_nodes)

    def test_iter_changes_only_modified(self):
        block_a = Block("Hello")
        block_b = Block("World")

        diff = diff_blocks(block_a, block_b)

        changes = list(diff.iter_changes())
        assert all(n.status != "unchanged" for n in changes)


class TestChangesByStatus:
    """Tests for grouping changes by status."""

    def test_get_changes_by_status(self):
        with Block("Root") as block_a:
            block_a /= "Keep"
            block_a /= "Modify"
            block_a /= "Remove"

        with Block("Root Modified") as block_b:
            block_b /= "Keep"
            block_b /= "Modified!"
            block_b /= "New"

        diff = diff_blocks(block_a, block_b)

        by_status = diff.get_changes_by_status()

        assert "added" in by_status
        assert "removed" in by_status
        assert "modified" in by_status


class TestSummary:
    """Tests for diff summary."""

    def test_summary_identical(self):
        block_a = Block("Same")
        block_b = Block("Same")

        diff = diff_blocks(block_a, block_b)

        assert "identical" in diff.summary().lower()

    def test_summary_with_changes(self):
        block_a = Block("Original")
        block_b = Block("Modified")

        diff = diff_blocks(block_a, block_b)

        summary = diff.summary()
        assert "modified" in summary.lower()


class TestFieldChanges:
    """Tests for FieldChange model."""

    def test_field_changes_list(self):
        block_a = Block("Hello", role="user", tags=["a"])
        block_b = Block("World", role="assistant", tags=["b"])

        diff = diff_blocks(block_a, block_b)

        field_changes = diff.root.field_changes
        field_names = [fc.field for fc in field_changes]

        assert "text" in field_names
        assert "role" in field_names
        assert "tags" in field_names

    def test_has_field_changes_property(self):
        block_a = Block("Hello")
        block_b = Block("World")

        diff = diff_blocks(block_a, block_b)

        assert diff.root.has_field_changes


class TestNodeDiff:
    """Tests for NodeDiff model."""

    def test_node_diff_path(self):
        with Block("Root") as block_a:
            block_a /= "Child"

        with Block("Root") as block_b:
            block_b /= "Modified"

        diff = diff_blocks(block_a, block_b)

        # Root has empty path
        assert diff.root.path == ""

        # Children have indexed paths
        if diff.root.children:
            assert diff.root.children[0].path == "0"

    def test_node_diff_repr(self):
        block_a = Block("Hello")
        block_b = Block("World")

        diff = diff_blocks(block_a, block_b)

        repr_str = repr(diff.root)
        assert "modified" in repr_str.lower()


class TestBlockDiffModel:
    """Tests for BlockDiff model."""

    def test_block_diff_repr(self):
        block_a = Block("Original")
        block_b = Block("Modified")

        diff = diff_blocks(block_a, block_b)

        repr_str = repr(diff)
        assert "BlockDiff" in repr_str


class TestTextDiffUtilities:
    """Tests for text diff utilities."""

    def test_get_text_diff(self):
        block_a = Block("Line 1\nLine 2\nLine 3")
        block_b = Block("Line 1\nModified Line\nLine 3")

        text_diff = get_text_diff(block_a, block_b)

        assert "Line 2" in text_diff or "Modified" in text_diff
        assert "---" in text_diff or "+++" in text_diff

    def test_get_inline_diff(self):
        block_a = Block("Same\nRemoved\nSame")
        block_b = Block("Same\nAdded\nSame")

        inline_diff = get_inline_diff(block_a, block_b)

        assert isinstance(inline_diff, list)
        assert all(isinstance(item, tuple) for item in inline_diff)
        assert all(len(item) == 3 for item in inline_diff)


class TestFormatDiffTree:
    """Tests for diff tree formatting."""

    def test_format_diff_tree(self):
        with Block("Root") as block_a:
            block_a /= "Original"

        with Block("Root Modified") as block_b:
            block_b /= "Changed"

        diff = diff_blocks(block_a, block_b)

        formatted = format_diff_tree(diff)

        assert isinstance(formatted, str)
        assert len(formatted) > 0


class TestBlockDiffMethod:
    """Tests for Block.diff() convenience method."""

    def test_block_diff_method(self):
        block_a = Block("Hello")
        block_b = Block("World")

        diff = block_a.diff(block_b)

        assert isinstance(diff, BlockDiff)
        assert not diff.is_identical

    def test_block_diff_text_method(self):
        block_a = Block("Line 1\nLine 2")
        block_b = Block("Line 1\nModified")

        text_diff = block_a.diff_text(block_b)

        assert isinstance(text_diff, str)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_blocks(self):
        block_a = Block()
        block_b = Block()

        diff = diff_blocks(block_a, block_b)

        assert diff.is_identical

    def test_empty_vs_content(self):
        block_a = Block()
        block_b = Block("Content")

        diff = diff_blocks(block_a, block_b)

        assert not diff.is_identical
        assert diff.root.text_change == ("", "Content")

    def test_same_block_reference(self):
        block = Block("Content")

        diff = diff_blocks(block, block)

        assert diff.is_identical

    def test_many_children_comparison(self):
        with Block("Root") as block_a:
            for i in range(10):
                block_a /= f"Child {i}"

        with Block("Root") as block_b:
            for i in range(10):
                if i == 5:
                    block_b /= "Modified Child 5"
                else:
                    block_b /= f"Child {i}"

        diff = diff_blocks(block_a, block_b)

        assert diff.has_content_changes
        changes = list(diff.iter_changes())
        # At least root and the modified child should be changed
        assert len(changes) >= 2
