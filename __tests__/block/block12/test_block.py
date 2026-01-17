"""Tests for Block class basic operations."""
import pytest
from promptview.block.block12 import Block, BlockChunk


class TestBlockCreation:
    """Tests for Block creation."""

    def test_create_with_string(self):
        block = Block("Hello")
        assert block.text == "Hello"

    def test_create_with_multiple_strings_concatenation(self):
        b1 = Block("Hello")
        b2 = Block("World")
        b3 = Block("!\n")

        assert b1 + b2 == "HelloWorld"
        assert b2 + b3 == "World!\n"
        assert b1 + b2 + b3 == "HelloWorld!\n"

    def test_create_with_inplace_add(self):
        b1 = Block("Hello")
        b2 = Block("World")
        b3 = Block("!\n")

        b1 += b2
        assert b1 == "HelloWorld"
        b1 += b3
        assert b1 == "HelloWorld!\n"

    def test_create_empty(self):
        block = Block()
        assert block.text == ""

    def test_create_with_role(self):
        block = Block("content", role="system")
        assert block.role == "system"

    def test_create_with_tags(self):
        block = Block("content", tags=["important", "header"])
        assert "important" in block.tags
        assert "header" in block.tags

    def test_create_with_style(self):
        block = Block("content", style="xml")
        assert "xml" in block.style

    def test_block_has_chunks(self):
        block = Block("Hello")
        assert block.chunks is not None


class TestBlockAppendPrepend:
    """Tests for append and prepend operations."""

    def test_append(self):
        block = Block("Start")
        block.append(" Middle")
        block.append(" End")
        assert block.text == "Start Middle End"

    def test_prepend(self):
        block = Block("End")
        block.prepend("Middle ")
        block.prepend("Start ")
        assert block.text == "Start Middle End"

    def test_append_with_style(self):
        block = Block("good")
        block.append(" world", style="xml")
        block.prepend("hello ", style="xml")
        assert "hello" in block.text
        assert "good" in block.text
        assert "world" in block.text


class TestBlockContextManager:
    """Tests for Block context manager."""

    def test_context_manager_basic(self):
        with Block("Header") as b:
            b /= "Hello"
            b /= "World"

        # /= operator adds newlines between elements
        assert b.render() == "Header\nHello\nWorld"
        assert len(b.children) == 2

    def test_context_manager_children_text(self):
        with Block("Header") as b:
            b /= "Hello"
            b /= "World"

        assert b.children[0].text == "Hello"
        assert b.children[1].text == "World"

    def test_nested_context_managers(self):
        with Block("Root") as root:
            with root("Parent") as parent:
                parent /= "Child1"
                parent /= "Child2"

        assert len(parent.children) == 2


class TestBlockTransform:
    """Tests for Block transform."""

    def test_transform_adds_newlines(self):
        with Block("Header") as b:
            b /= "Hello"
            b /= "World"

        blk = b.transform()

        assert blk.text == "Header\n"
        assert blk.children[0].text == "Hello\n"
        assert blk.children[1].text == "World"


class TestBlockPositionManagement:
    """Tests for position-based text storage."""

    def test_positions_basic(self):
        with Block("Root") as root:
            root /= "AAA"
            root /= "BBB"
            root /= "CCC"

        c1, c2, c3 = root.children

        assert c1.text == "AAA"
        assert c2.text == "BBB"
        assert c3.text == "CCC"

    def test_positions_after_append(self):
        with Block("Root") as root:
            root /= "AAA"
            root /= "BBB"
            root /= "CCC"

        c1, c2, c3 = root.children

        c1.append("111")

        assert c1.text == "AAA111"
        assert c2.text == "BBB"
        assert c3.text == "CCC"


class TestBlockCallOperator:
    """Tests for Block call operator."""

    def test_call_creates_child(self):
        with Block("Root") as root:
            with root("Parent") as parent:
                with parent("Child") as child:
                    child /= "Content"

        assert len(root.children) == 1
        assert len(parent.children) == 1


class TestBlockPrevNext:
    """Tests for prev/next navigation."""

    def test_prev_sibling(self):
        with Block(content="example") as block:
            with block("item 1") as item1:
                item1 /= "cat"
            with block("item 2") as item2:
                item2 /= "dog"
            with block("item 3") as item3:
                item3 /= "mouse"

        assert item3.prev() == item2.tail


class TestBlockHeadBody:
    """Tests for head/body access."""

    def test_head_and_body(self):
        with Block("ToDos", style="md li-num") as todos:
            todos /= "Buy groceries"
            todos /= "Finish project"
            todos /= "Call the bank"

        assert todos.head.text == "ToDos"
        assert len(todos.body) == 3
