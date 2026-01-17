"""Tests for Block tree traversal."""
import pytest
from promptview.block.block12 import Block


class TestBlockDepthFirstTraversal:
    """Tests for depth-first traversal."""

    def test_iter_depth_first(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("A1", tags=["item"])
                a /= Block("A2", tags=["item"])
            root /= Block("B", tags=["section"])

        blocks = list(root.iter_depth_first())

        assert blocks[0].text == "Root"
        assert blocks[0].tags == ["root"]


class TestBlockGetAll:
    """Tests for getting blocks by tag."""

    def test_get_all_by_tag(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("A1", tags=["item"])
                a /= Block("A2", tags=["item"])
            root /= Block("B", tags=["section"])

        items = root.get_all("item")
        assert len(items) == 2
        assert items[0].text == "A1"
        assert items[1].text == "A2"


class TestBlockSiblingNavigation:
    """Tests for sibling navigation."""

    def test_next_sibling(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("A1", tags=["item"])
                a /= Block("A2", tags=["item"])

        a1, a2 = a.children
        assert a1.next_sibling().text == "A2"


class TestBlockPrevNavigation:
    """Tests for prev navigation with transform."""

    def test_prev_navigation_with_xml_transform(self):
        with Block(content="example", style="xml") as block:
            with block("item 1", tags=["item1"], style="xml") as item1:
                item1 /= "cat"
            with block("item 2", tags=["item2"], style="xml") as item2:
                item2 /= "dog"
            with block("item 3", tags=["item3"], style="xml") as item3:
                item3 /= "mouse"

        blk = block.transform()

        item1 = blk.get("item1")
        item2 = blk.get("item2")
        item3 = blk.get("item3")

        assert item2.prev() is item1.tail
        assert item3.prev() is item2.tail


class TestBlockDepth:
    """Tests for block depth."""

    def test_depth_values(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("A1", tags=["item"])
                a /= Block("A2", tags=["item"])

        assert root.depth == 0
        assert a.depth == 1
        assert a.children[0].depth == 2
