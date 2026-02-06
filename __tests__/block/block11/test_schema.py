"""Tests for BlockSchema, BlockList, and BlockListSchema classes."""
import pytest
from chatboard.block.block11 import (
    Block,
    BlockSchema,
    BlockList,
    BlockListSchema,
    BlockText,
)


class TestBlockSchemaCreation:
    """Tests for BlockSchema creation."""

    def test_create_basic_schema(self):
        schema = BlockSchema("response")
        assert schema.name == "response"
        assert schema.content == "response"
        assert "response" in schema.tags

    def test_create_with_type(self):
        schema = BlockSchema("count", type=int)
        assert schema.type is int

    def test_create_with_style(self):
        schema = BlockSchema("answer", style="xml")
        assert "xml" in schema.style

    def test_create_with_tags(self):
        schema = BlockSchema("response", tags=["reply", "output"])
        assert "response" in schema.tags
        assert "reply" in schema.tags
        assert "output" in schema.tags

    def test_create_with_role(self):
        schema = BlockSchema("system", role="system")
        assert schema.role == "system"

    def test_create_optional(self):
        schema = BlockSchema("notes", is_required=False)
        assert schema.is_required is False

    def test_create_with_attrs(self):
        schema = BlockSchema("config", attrs={"version": "1.0"})
        assert schema.attrs["version"] == "1.0"

    def test_name_added_to_tags(self):
        schema = BlockSchema("test")
        assert schema.tags[0] == "test"


class TestBlockSchemaInstantiate:
    """Tests for BlockSchema.instantiate()."""

    def test_instantiate_creates_block(self):
        schema = BlockSchema("response")
        block = schema.instantiate()
        assert isinstance(block, Block)

    def test_instantiate_with_content(self):
        schema = BlockSchema("answer")
        block = schema.instantiate("Hello World")
        assert block.content == "Hello World"

    def test_instantiate_inherits_tags(self):
        schema = BlockSchema("response", tags=["reply"])
        block = schema.instantiate()
        assert "response" in block.tags
        assert "reply" in block.tags


class TestBlockSchemaChildren:
    """Tests for BlockSchema with children."""

    def test_append_child_schema(self):
        parent = BlockSchema("response")
        child = BlockSchema("thinking")
        parent.append_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_itruediv_operator(self):
        response = BlockSchema("response")
        thinking = BlockSchema("thinking")
        response /= thinking

        assert len(response.children) == 1

    def test_nested_schema(self):
        response = BlockSchema("response")
        thinking = BlockSchema("thinking")
        answer = BlockSchema("answer")

        response /= thinking
        response /= answer

        assert len(response.children) == 2
        assert response.children[0].name == "thinking"
        assert response.children[1].name == "answer"


class TestBlockSchemaCopy:
    """Tests for BlockSchema copy operations."""

    def test_deep_copy(self):
        original = BlockSchema("parent")
        child = BlockSchema("child")
        original.append_child(child)

        copy = original.copy(deep=True)

        assert copy.name == "parent"
        assert len(copy.children) == 1
        assert copy.children[0].name == "child"
        assert copy is not original
        assert copy.children[0] is not child

    def test_copy_head(self):
        original = BlockSchema("parent")
        child = BlockSchema("child")
        original.append_child(child)

        head_copy = original.copy_head()

        assert head_copy.name == "parent"
        assert len(head_copy.children) == 0

    def test_shallow_copy(self):
        original = BlockSchema("test")
        copy = original.copy(deep=False)

        assert copy.name == original.name
        assert copy.span is original.span


class TestBlockSchemaRepr:
    """Tests for BlockSchema repr."""

    def test_repr_basic(self):
        schema = BlockSchema("response")
        repr_str = repr(schema)

        assert "BlockSchema" in repr_str
        assert "response" in repr_str

    def test_repr_with_type(self):
        schema = BlockSchema("count", type=int)
        repr_str = repr(schema)

        assert "type=int" in repr_str

    def test_repr_optional(self):
        schema = BlockSchema("notes", is_required=False)
        repr_str = repr(schema)

        assert "optional" in repr_str


class TestBlockListCreation:
    """Tests for BlockList creation."""

    def test_create_empty_list(self):
        block_list = BlockList()
        assert len(block_list) == 0

    def test_create_with_item_schema(self):
        item_schema = BlockSchema("item")
        block_list = BlockList(item_schema=item_schema)

        assert block_list.item_schema is item_schema

    def test_create_with_tags(self):
        block_list = BlockList(tags=["items", "list"])
        assert "items" in block_list.tags

    def test_list_is_wrapper(self):
        block_list = BlockList()
        assert block_list.is_wrapper is True


class TestBlockListOperations:
    """Tests for BlockList operations."""

    def test_append_item_block(self):
        block_list = BlockList()
        item = Block("Item 1")
        block_list.append_item(item)

        assert len(block_list) == 1
        assert block_list[0] is item

    def test_append_item_content(self):
        block_list = BlockList()
        block_list.append_item("Item 1")

        assert len(block_list) == 1
        assert block_list[0].content == "Item 1"

    def test_append_item_with_schema(self):
        item_schema = BlockSchema("item")
        block_list = BlockList(item_schema=item_schema)
        item = block_list.append_item("Content")

        assert "item" in item.tags

    def test_iterate_list(self):
        block_list = BlockList()
        block_list.append_item("A")
        block_list.append_item("B")
        block_list.append_item("C")

        contents = [item.content for item in block_list]
        assert contents == ["A", "B", "C"]

    def test_getitem(self):
        block_list = BlockList()
        block_list.append_item("First")
        block_list.append_item("Second")

        assert block_list[0].content == "First"
        assert block_list[1].content == "Second"


class TestBlockListCopy:
    """Tests for BlockList copy operations."""

    def test_deep_copy(self):
        original = BlockList()
        item1 = Block("A")
        item2 = Block("B")
        original.append_child(item1)
        original.append_child(item2)

        copy = original.copy(deep=True)

        assert len(copy) == 2
        assert copy is not original
        assert copy[0] is not original[0]
        assert copy.block_text is not original.block_text

    def test_deep_copy_with_item_schema(self):
        item_schema = BlockSchema("item")
        original = BlockList(item_schema=item_schema)

        copy = original.copy(deep=True)

        assert copy.item_schema is not None
        assert copy.item_schema is not item_schema

    def test_shallow_copy(self):
        original = BlockList()
        original.append_item("A")

        copy = original.copy(deep=False)

        assert copy.children == original.children


class TestBlockListSchemaCreation:
    """Tests for BlockListSchema creation."""

    def test_create_basic(self):
        schema = BlockListSchema("tool")
        assert schema.item_name == "tool"
        assert schema.name == "tool"

    def test_create_with_container_name(self):
        schema = BlockListSchema("tool", name="tools")
        assert schema.item_name == "tool"
        assert schema.name == "tools"

    def test_create_with_key(self):
        schema = BlockListSchema("item", key="id")
        assert schema.key == "id"

    def test_default_style(self):
        schema = BlockListSchema("item")
        assert "xml-list" in schema.style

    def test_create_with_attrs(self):
        schema = BlockListSchema("item", attrs={"type": "custom"})
        assert schema.attrs["type"] == "custom"


class TestBlockListSchemaRegistration:
    """Tests for BlockListSchema schema registration."""

    def test_register_schema(self):
        list_schema = BlockListSchema("tool", name="tools")
        item_schema = BlockSchema("tool")

        list_schema.register(item_schema)

        assert item_schema in list_schema.list_schemas
        assert item_schema in list_schema.children

    def test_get_item_schema(self):
        list_schema = BlockListSchema("tool", name="tools")
        item_schema = BlockSchema("tool")
        list_schema.register(item_schema)

        result = list_schema.get_item_schema()
        assert result is item_schema

    def test_get_item_schema_by_key(self):
        list_schema = BlockListSchema("tool", name="tools", key="type")
        schema1 = BlockSchema("tool", attrs={"type": "hammer"})
        schema2 = BlockSchema("tool", attrs={"type": "wrench"})

        list_schema.register(schema1)
        list_schema.register(schema2)

        result = list_schema.get_item_schema("wrench")
        assert result is schema2


class TestBlockListSchemaInstantiate:
    """Tests for BlockListSchema instantiate."""

    def test_instantiate_creates_block_list(self):
        schema = BlockListSchema("item")
        result = schema.instantiate()

        assert isinstance(result, BlockList)

    def test_instantiate_with_registered_schema(self):
        list_schema = BlockListSchema("tool", name="tools")
        item_schema = BlockSchema("tool")
        list_schema.register(item_schema)

        result = list_schema.instantiate()

        assert result.item_schema is item_schema

    def test_instantiate_inherits_tags(self):
        schema = BlockListSchema("item", tags=["list"])
        result = schema.instantiate()

        assert "item" in result.tags

    def test_instantiate_item(self):
        schema = BlockListSchema("tool")
        item = schema.instantiate_item("hammer")

        assert isinstance(item, Block)
        assert item.content == "hammer"
        assert "tool" in item.tags

    def test_instantiate_item_with_registered_schema(self):
        list_schema = BlockListSchema("tool", name="tools")
        item_schema = BlockSchema("tool")
        item_schema.append_child(BlockSchema("description"))
        list_schema.register(item_schema)

        item = list_schema.instantiate_item("hammer")

        assert "tool" in item.tags


class TestBlockListSchemaCopy:
    """Tests for BlockListSchema copy operations."""

    def test_deep_copy(self):
        original = BlockListSchema("tool", name="tools", key="type")
        item_schema = BlockSchema("tool")
        original.register(item_schema)

        copy = original.copy(deep=True)

        assert copy.item_name == "tool"
        assert copy.name == "tools"
        assert copy.key == "type"
        assert len(copy.list_schemas) == 1
        assert copy.list_schemas[0] is not item_schema

    def test_shallow_copy(self):
        original = BlockListSchema("tool")
        copy = original.copy(deep=False)

        assert copy.item_name == original.item_name
        assert copy.span is original.span


class TestBlockListSchemaRepr:
    """Tests for BlockListSchema repr."""

    def test_repr_basic(self):
        schema = BlockListSchema("item")
        repr_str = repr(schema)

        assert "BlockListSchema" in repr_str
        assert "item_name='item'" in repr_str

    def test_repr_with_name(self):
        schema = BlockListSchema("tool", name="tools")
        repr_str = repr(schema)

        assert "name='tools'" in repr_str

    def test_repr_with_key(self):
        schema = BlockListSchema("item", key="id")
        repr_str = repr(schema)

        assert "key='id'" in repr_str
