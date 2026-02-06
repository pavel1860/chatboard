"""Tests for Path class and compute_path function."""

import pytest
from chatboard.block.block11 import Block, Path, compute_path, BlockSchema


class TestPathCreation:
    """Test Path construction and basic properties."""

    def test_create_from_list(self):
        path = Path([0, 1, 2])
        assert path.indices == (0, 1, 2)
        assert path.tags == ()

    def test_create_with_tags(self):
        path = Path([0, 1], ["response", "thinking"])
        assert path.indices == (0, 1)
        assert path.tags == ("response", "thinking")

    def test_create_from_tuple(self):
        path = Path((0, 1, 2), ("a", "b", "c"))
        assert path.indices == (0, 1, 2)
        assert path.tags == ("a", "b", "c")

    def test_root_path(self):
        path = Path.root()
        assert path.indices == ()
        assert path.tags == ()
        assert path.is_root

    def test_from_string(self):
        path = Path.from_string("0.2.1")
        assert path.indices == (0, 2, 1)

    def test_from_empty_string(self):
        path = Path.from_string("")
        assert path.indices == ()
        assert path.is_root


class TestPathProperties:
    """Test Path property accessors."""

    def test_depth(self):
        assert Path([]).depth == 0
        assert Path([0]).depth == 1
        assert Path([0, 1, 2]).depth == 3

    def test_len(self):
        assert len(Path([])) == 0
        assert len(Path([0, 1])) == 2

    def test_is_root(self):
        assert Path([]).is_root
        assert not Path([0]).is_root

    def test_last_index(self):
        assert Path([0, 1, 2]).last_index == 2
        assert Path([]).last_index is None

    def test_last_tag(self):
        assert Path([0, 1], ["a", "b"]).last_tag == "b"
        assert Path([0, 1]).last_tag is None


class TestPathComparisons:
    """Test Path comparison operators (tree order)."""

    def test_equality(self):
        assert Path([0, 1]) == Path([0, 1])
        assert Path([0, 1]) != Path([0, 2])

    def test_less_than(self):
        assert Path([0]) < Path([1])
        assert Path([0, 1]) < Path([0, 2])
        assert Path([0, 1]) < Path([0, 1, 0])  # ancestor < descendant

    def test_greater_than(self):
        assert Path([1]) > Path([0])
        assert Path([0, 2]) > Path([0, 1])

    def test_less_equal(self):
        assert Path([0, 1]) <= Path([0, 1])
        assert Path([0, 1]) <= Path([0, 2])

    def test_greater_equal(self):
        assert Path([0, 1]) >= Path([0, 1])
        assert Path([0, 2]) >= Path([0, 1])

    def test_hash(self):
        p1 = Path([0, 1])
        p2 = Path([0, 1])
        assert hash(p1) == hash(p2)
        # Can be used in sets/dicts
        s = {p1, p2}
        assert len(s) == 1


class TestPathRelationships:
    """Test Path relationship methods."""

    def test_is_ancestor_of(self):
        parent = Path([0, 1])
        child = Path([0, 1, 2])
        other = Path([0, 2])

        assert parent.is_ancestor_of(child)
        assert parent.is_ancestor_of(parent)  # self is ancestor
        assert not child.is_ancestor_of(parent)
        assert not parent.is_ancestor_of(other)

    def test_is_strict_ancestor_of(self):
        parent = Path([0, 1])
        child = Path([0, 1, 2])

        assert parent.is_strict_ancestor_of(child)
        assert not parent.is_strict_ancestor_of(parent)

    def test_is_descendant_of(self):
        parent = Path([0, 1])
        child = Path([0, 1, 2])

        assert child.is_descendant_of(parent)
        assert not parent.is_descendant_of(child)

    def test_is_strict_descendant_of(self):
        parent = Path([0, 1])
        child = Path([0, 1, 2])

        assert child.is_strict_descendant_of(parent)
        assert not child.is_strict_descendant_of(child)

    def test_includes(self):
        # includes is alias for is_ancestor_of
        parent = Path([0])
        child = Path([0, 1])
        assert parent.includes(child)

    def test_is_sibling_of(self):
        p1 = Path([0, 1])
        p2 = Path([0, 2])
        p3 = Path([0, 1, 0])

        assert p1.is_sibling_of(p2)
        assert not p1.is_sibling_of(p3)

    def test_root_siblings(self):
        # Both roots are siblings
        assert Path([]).is_sibling_of(Path([]))


class TestPathNavigation:
    """Test Path navigation methods."""

    def test_parent(self):
        path = Path([0, 1, 2], ["a", "b", "c"])
        parent = path.parent

        assert parent is not None
        assert parent.indices == (0, 1)
        assert parent.tags == ("a", "b")

    def test_parent_of_root(self):
        assert Path([]).parent is None

    def test_child(self):
        path = Path([0, 1], ["a", "b"])
        child = path.child(2, "c")

        assert child.indices == (0, 1, 2)
        assert child.tags == ("a", "b", "c")

    def test_child_without_tag(self):
        path = Path([0])
        child = path.child(1)

        assert child.indices == (0, 1)
        assert child.tags == ()

    def test_ancestors(self):
        path = Path([0, 1, 2])
        ancestors = list(path.ancestors())

        assert len(ancestors) == 3
        assert ancestors[0].indices == ()
        assert ancestors[1].indices == (0,)
        assert ancestors[2].indices == (0, 1)

    def test_common_ancestor(self):
        p1 = Path([0, 1, 2])
        p2 = Path([0, 1, 3])

        common = p1.common_ancestor(p2)
        assert common.indices == (0, 1)

    def test_common_ancestor_no_common(self):
        p1 = Path([0, 1])
        p2 = Path([1, 2])

        common = p1.common_ancestor(p2)
        assert common.indices == ()


class TestPathArithmetic:
    """Test Path subtraction."""

    def test_subtract_ancestor(self):
        path = Path([0, 1, 2])
        ancestor = Path([0])

        result = path - ancestor
        assert result.indices == (1, 2)

    def test_subtract_self(self):
        path = Path([0, 1])
        result = path - path
        assert result.indices == ()
        assert result.is_root

    def test_subtract_non_ancestor_raises(self):
        p1 = Path([0, 1])
        p2 = Path([0, 2])

        with pytest.raises(ValueError):
            _ = p1 - p2


class TestPathStringRepresentation:
    """Test Path string methods."""

    def test_str(self):
        assert str(Path([0, 1, 2])) == "0.1.2"
        assert str(Path([])) == ""

    def test_repr(self):
        path = Path([0, 1], ["a", "b"])
        assert repr(path) == "Path([0, 1], ['a', 'b'])"

    def test_tag_str(self):
        path = Path([0, 1], ["response", "thinking"])
        assert path.tag_str() == "response.thinking"
        assert path.tag_str("/") == "response/thinking"


class TestComputePathBasic:
    """Test compute_path with regular blocks."""

    def test_root_block_path(self):
        root = Block("root")
        path = root.path

        assert path.is_root
        assert path.indices == ()

    def test_single_child_path(self):
        with Block("root") as root:
            child = Block("child")
            root.mutator.append_child(child)

        assert child.path.indices == (0,)

    def test_nested_children_path(self):
        with Block("root") as root:
            with root("level1") as level1:
                level2 = Block("level2")
                level1.mutator.append_child(level2)

        assert level1.path.indices == (0,)
        assert level2.path.indices == (0, 0)

    def test_sibling_paths(self):
        with Block("root") as root:
            child0 = Block("child0")
            child1 = Block("child1")
            child2 = Block("child2")
            root.mutator.append_child(child0)
            root.mutator.append_child(child1)
            root.mutator.append_child(child2)

        assert child0.path.indices == (0,)
        assert child1.path.indices == (1,)
        assert child2.path.indices == (2,)

    def test_path_with_tags(self):
        with Block("root") as root:
            child = Block("child", tags=["response"])
            root.mutator.append_child(child)

        path = child.path
        assert path.indices == (0,)
        assert path.tags == ("response",)

    def test_nested_tags(self):
        with Block("root") as root:
            child1 = Block("child1", tags=["response"])
            root.mutator.append_child(child1)
            child2 = Block("child2", tags=["thinking"])
            child1.mutator.append_child(child2)

        path = child2.path
        assert path.indices == (0, 0)
        assert path.tags == ("response", "thinking")


class TestComputePathWithXmlMutator:
    """Test compute_path transparency with XmlMutator."""

    def test_xml_wrapper_transparent(self):
        """XmlMutator wrapper should be invisible to path."""
        with Block() as root:
            # Create an XML-styled view
            with root.view("thought", style="xml") as thought:
                content = Block("thinking content")
                thought.mutator.append_child(content)

        # The path should go directly from root to thought to content
        # The XML wrapper structure should be transparent
        thought_path = thought.path
        content_path = content.path

        # thought is direct child of root in logical tree
        assert thought_path.indices == (0,)
        # content is child of thought in logical tree
        assert content_path.indices == (0, 0)

    def test_multiple_xml_children(self):
        """Multiple XML children should have correct sibling indices."""
        with Block() as root:
            with root.view("first", style="xml") as first:
                pass
            with root.view("second", style="xml") as second:
                pass
            with root.view("third", style="xml") as third:
                pass

        assert first.path.indices == (0,)
        assert second.path.indices == (1,)
        assert third.path.indices == (2,)

    def test_nested_xml_views(self):
        """Nested XML views should maintain correct path depth."""
        with Block() as root:
            with root.view("outer", style="xml") as outer:
                with outer.view("inner", style="xml") as inner:
                    leaf = Block("leaf content")
                    inner.mutator.append_child(leaf)

        assert outer.path.indices == (0,)
        assert inner.path.indices == (0, 0)
        assert leaf.path.indices == (0, 0, 0)

    def test_mixed_blocks_and_views(self):
        """Mix of regular blocks and XML views should work correctly."""
        with Block() as root:
            regular1 = Block("regular1")
            root.mutator.append_child(regular1)

            with root.view("xml_view", style="xml") as xml_view:
                pass

            regular2 = Block("regular2")
            root.mutator.append_child(regular2)

        assert regular1.path.indices == (0,)
        assert xml_view.path.indices == (1,)
        assert regular2.path.indices == (2,)


class TestPathUseCases:
    """Test practical use cases for Path."""

    def test_find_block_by_path(self):
        """Navigate to a block using a path."""
        with Block() as root:
            with root("a") as a:
                with a("b") as b:
                    c = Block("c")
                    b.mutator.append_child(c)

        # Navigate from root using path
        target_path = Path([0, 0, 0])
        current = root
        for idx in target_path.indices:
            current = current.body[idx]

        assert current is c

    def test_path_comparison_for_ordering(self):
        """Use path comparison to order blocks."""
        with Block() as root:
            b1 = Block("b1")
            b2 = Block("b2")
            b3 = Block("b3")
            root.mutator.append_child(b1)
            root.mutator.append_child(b2)
            root.mutator.append_child(b3)

            b1_child = Block("b1_child")
            b1.mutator.append_child(b1_child)

        blocks = [b3, b1_child, b1, b2]
        sorted_blocks = sorted(blocks, key=lambda b: b.path)

        assert sorted_blocks == [b1, b1_child, b2, b3]

    def test_is_within_subtree(self):
        """Check if a block is within another's subtree using paths."""
        with Block() as root:
            with root("container") as container:
                with container("nested") as nested:
                    pass
            with root("other") as other:
                pass

        # nested is within container's subtree
        assert container.path.is_ancestor_of(nested.path)
        # other is NOT within container's subtree
        assert not container.path.is_ancestor_of(other.path)
