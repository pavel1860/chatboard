"""Tests for Block class."""
import pytest
from promptview.block.block11 import Block, BlockChunk, Span, BlockText


class TestBlockCreation:
    """Tests for Block creation."""

    def test_create_with_string(self):
        block = Block("Hello")
        assert block.content == "Hello"
        assert block.span is not None
        assert block.span.content_text == "Hello"

    def test_create_with_int(self):
        block = Block(42)
        assert block.content == "42"

    def test_create_with_float(self):
        block = Block(3.14)
        assert block.content == "3.14"

    def test_create_with_bool(self):
        block = Block(True)
        assert block.content == "true"

    def test_create_empty(self):
        block = Block()
        assert block.content == ""
        assert block.span is not None
        assert block.span.is_empty is True

    def test_create_with_role(self):
        block = Block("content", role="system")
        assert block.role == "system"

    def test_create_with_tags(self):
        block = Block("content", tags=["important", "header"])
        assert block.tags == ["important", "header"]

    def test_create_with_style(self):
        block = Block("content", style="bold italic")
        assert block.style == ["bold", "italic"]

    def test_block_has_block_text(self):
        block = Block("Hello")
        assert block.block_text is not None
        assert isinstance(block.block_text, BlockText)


class TestBlockFactoryMethods:
    """Tests for Block factory methods."""

    def test_empty(self):
        block = Block.empty()
        assert block.content == ""
        assert block.span is not None

    def test_from_span(self):
        span = Span(chunks=[BlockChunk("Test")])
        block = Block.from_span(span)
        assert block.span is span
        assert block.content == "Test"


class TestBlockContentMutation:
    """Tests for Block content mutation."""

    def test_append_content(self):
        block = Block("Hello")
        block.append(" World")
        assert block.content == "Hello World"

    def test_prepend_content(self):
        block = Block("World")
        block.prepend("Hello ")
        assert block.content == "Hello World"

    def test_append_prefix(self):
        block = Block("text")
        block.append_prefix(">> ")
        assert block.span.prefix_text == ">> "
        assert block.span.text == ">> text"

    def test_prepend_prefix(self):
        block = Block("text")
        block.append_prefix("- ")
        block.prepend_prefix("  ")
        assert block.span.prefix_text == "  - "

    def test_append_postfix(self):
        block = Block("text")
        block.append_postfix(";")
        assert block.span.postfix_text == ";"

    def test_add_newline(self):
        block = Block("text")
        block.add_newline()
        assert block.span.postfix_text == "\n"
        assert block.has_newline() is True

    def test_method_chaining(self):
        block = (
            Block("item")
            .append_prefix("- ")
            .add_newline()
        )
        assert block.span.text == "- item\n"


class TestBlockChildren:
    """Tests for Block children operations."""

    def test_append_child(self):
        parent = Block("Parent")
        child = Block("Child")
        parent.append_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] is child
        assert child.parent is parent

    def test_prepend_child(self):
        parent = Block("Parent")
        child1 = Block("First")
        child2 = Block("Second")

        parent.append_child(child1)
        parent.prepend_child(child2)

        assert parent.children == [child2, child1]

    def test_insert_child(self):
        parent = Block("Parent")
        child1 = Block("A")
        child2 = Block("C")
        child3 = Block("B")

        parent.append_child(child1)
        parent.append_child(child2)
        parent.insert_child(1, child3)

        assert [c.content for c in parent.children] == ["A", "B", "C"]

    def test_remove_child(self):
        parent = Block("Parent")
        child1 = Block("A")
        child2 = Block("B")

        parent.append_child(child1)
        parent.append_child(child2)
        parent.remove_child(child1)

        assert len(parent.children) == 1
        assert parent.children[0] is child2
        assert child1.parent is None

    def test_child_inherits_block_text(self):
        parent = Block("Parent")
        child = Block("Child")
        parent.append_child(child)

        assert child.block_text is parent.block_text

    def test_child_span_moved_to_parent_block_text(self):
        parent = Block("Parent")
        child = Block("Child")
        parent.append_child(child)

        # Both spans should be in parent's BlockText
        assert parent.block_text.text() == "ParentChild"


class TestBlockTreeNavigation:
    """Tests for Block tree navigation."""

    def test_is_leaf(self):
        parent = Block("Parent")
        child = Block("Child")

        assert parent.is_leaf() is True
        parent.append_child(child)
        assert parent.is_leaf() is False
        assert child.is_leaf() is True

    def test_is_root(self):
        parent = Block("Parent")
        child = Block("Child")
        parent.append_child(child)

        assert parent.is_root() is True
        assert child.is_root() is False

    def test_depth(self):
        root = Block("Root")
        child = Block("Child")
        grandchild = Block("Grandchild")

        root.append_child(child)
        child.append_child(grandchild)

        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2

    def test_root(self):
        root = Block("Root")
        child = Block("Child")
        grandchild = Block("Grandchild")

        root.append_child(child)
        child.append_child(grandchild)

        assert grandchild.root() is root
        assert child.root() is root
        assert root.root() is root


class TestBlockIteration:
    """Tests for Block iteration methods."""

    def test_iter_depth_first(self):
        root = Block("Root")
        child1 = Block("C1")
        child2 = Block("C2")
        grandchild = Block("GC")

        root.append_child(child1)
        root.append_child(child2)
        child1.append_child(grandchild)

        blocks = list(root.iter_depth_first())
        assert [b.content for b in blocks] == ["Root", "C1", "GC", "C2"]

    def test_iter_breadth_first(self):
        root = Block("Root")
        child1 = Block("C1")
        child2 = Block("C2")
        grandchild = Block("GC")

        root.append_child(child1)
        root.append_child(child2)
        child1.append_child(grandchild)

        blocks = list(root.iter_breadth_first())
        assert [b.content for b in blocks] == ["Root", "C1", "C2", "GC"]

    def test_iter_leaves(self):
        root = Block("Root")
        child1 = Block("C1")
        child2 = Block("C2")
        grandchild = Block("GC")

        root.append_child(child1)
        root.append_child(child2)
        child1.append_child(grandchild)

        leaves = list(root.iter_leaves())
        assert [b.content for b in leaves] == ["GC", "C2"]

    def test_iter_spans(self):
        root = Block("Root")
        child = Block("Child")
        root.append_child(child)

        spans = list(root.iter_spans())
        assert len(spans) == 2
        assert spans[0].content_text == "Root"
        assert spans[1].content_text == "Child"


class TestBlockOperators:
    """Tests for Block operator overloading."""

    def test_itruediv_with_string(self):
        block = Block("Header")
        block /= "Item"

        assert len(block.children) == 1
        assert block.children[0].content == "Item"

    def test_itruediv_with_block(self):
        parent = Block("Parent")
        child = Block("Child")
        parent /= child

        assert parent.children[0] is child

    def test_itruediv_auto_newline(self):
        block = Block("First")
        block /= "Second"
        block /= "Third"

        # First should have newline added before Second was appended
        assert block.span.has_newline() is True

    def test_context_manager(self):
        with Block("Header") as block:
            block /= "Item 1"
            block /= "Item 2"

        assert len(block.children) == 2

    def test_call_appends_child(self):
        parent = Block("Parent")
        child = parent("Child")

        assert len(parent.children) == 1
        assert child.content == "Child"


class TestBlockCopy:
    """Tests for Block copy operations."""

    def test_deep_copy(self):
        original = Block("Parent")
        child = Block("Child")
        original.append_child(child)

        copy = original.copy(deep=True)

        assert copy.content == "Parent"
        assert len(copy.children) == 1
        assert copy.children[0].content == "Child"
        assert copy is not original
        assert copy.children[0] is not child
        assert copy.block_text is not original.block_text

    def test_deep_copy_isolation(self):
        original = Block("Original")
        copy = original.copy(deep=True)

        copy.append(" Modified")

        assert original.content == "Original"
        assert copy.content == "Original Modified"

    def test_shallow_copy(self):
        original = Block("Parent")
        child = Block("Child")
        original.append_child(child)

        copy = original.copy(deep=False)

        assert copy.span is original.span
        assert copy.block_text is original.block_text

    def test_copy_head(self):
        parent = Block("Parent", role="test")
        child1 = Block("Child1")
        child2 = Block("Child2")
        parent.append_child(child1)
        parent.append_child(child2)

        head_copy = parent.copy_head()

        assert head_copy.content == "Parent"
        assert head_copy.role == "test"
        assert len(head_copy.children) == 0
        assert head_copy.span is not parent.span
        assert head_copy.block_text is not parent.block_text


class TestBlockMutatedAccess:
    """Tests for mutator-based access (head, body, content)."""

    def test_head_returns_span(self):
        block = Block("Test")
        assert block.head is block.span

    def test_body_returns_children(self):
        block = Block("Parent")
        child = Block("Child")
        block.append_child(child)

        assert block.body == block.children

    def test_content_returns_text(self):
        block = Block("Hello")
        assert block.content == "Hello"


class TestBlockDebug:
    """Tests for Block debug output."""

    def test_repr(self):
        block = Block("Hello", role="system")
        repr_str = repr(block)

        assert "role='system'" in repr_str
        assert "content='Hello'" in repr_str

    def test_debug(self):
        parent = Block("Parent")
        child = Block("Child")
        parent.append_child(child)

        debug_str = parent.debug()

        assert "Parent" in debug_str
        assert "Child" in debug_str
        assert "children=1" in debug_str
