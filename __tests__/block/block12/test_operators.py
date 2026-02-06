"""Tests for Block operators."""
import pytest
from chatboard.block.block12 import Block


class TestBlockEqualityOperators:
    """Tests for Block equality operators."""

    def test_block_equality_same_content(self):
        b1 = Block("hello")
        b3 = Block("hello")
        assert b1 == b3

    def test_block_inequality_different_content(self):
        b1 = Block("hello")
        b2 = Block("world")
        assert not (b1 == b2)
        assert b1 != b2

    def test_block_self_equality(self):
        b1 = Block("hello")
        assert b1 == b1

    def test_block_string_equality(self):
        b1 = Block("hello")
        assert "hello" == b1
        assert b1 == "hello"

    def test_block_string_inequality(self):
        b1 = Block("hello")
        assert b1 != "world"


class TestBlockAddOperator:
    """Tests for Block + operator."""

    def test_add_blocks(self):
        b1 = Block("Hello")
        b2 = Block("World")
        result = b1 + b2
        assert result == "HelloWorld"

    def test_add_three_blocks(self):
        b1 = Block("A")
        b2 = Block("B")
        b3 = Block("C")
        result = b1 + b2 + b3
        assert result == "ABC"


class TestBlockInPlaceAddOperator:
    """Tests for Block += operator."""

    def test_iadd_block(self):
        b1 = Block("Hello")
        b2 = Block("World")
        b1 += b2
        assert b1 == "HelloWorld"

    def test_iadd_chain(self):
        b1 = Block("A")
        b2 = Block("B")
        b3 = Block("C")
        b1 += b2
        b1 += b3
        assert b1 == "ABC"


class TestBlockTrueDivOperator:
    """Tests for Block /= operator (append child)."""

    def test_itruediv_with_string(self):
        with Block("Header") as block:
            block /= "Item"

        assert len(block.children) == 1
        assert block.children[0].text == "Item"

    def test_itruediv_with_block(self):
        parent = Block("Parent")
        child = Block("Child")
        with parent as p:
            p /= child

        # Block content is preserved
        assert parent.children[0].text == "Child"

    def test_itruediv_multiple(self):
        with Block("Root") as block:
            block /= "A"
            block /= "B"
            block /= "C"

        assert len(block.children) == 3
        assert block.children[0].text == "A"
        assert block.children[1].text == "B"
        assert block.children[2].text == "C"
