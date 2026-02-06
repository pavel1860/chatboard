"""Tests for Block tree traversal."""
import pytest
from chatboard.block.block12 import Block, BlockSchema
from chatboard.block.block12.schema import BlockList


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


class TestBlockGet:
    """Tests for get() method."""

    def test_get_finds_first_match(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("A1", tags=["item"])
                a /= Block("A2", tags=["item"])
            root /= Block("B", tags=["section"])

        result = root.get("item")
        assert result.text == "A1"

    def test_get_finds_nested_block(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                with a("Nested", tags=["nested"]) as nested:
                    nested /= Block("Deep", tags=["deep"])

        result = root.get("deep")
        assert result.text == "Deep"

    def test_get_raises_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])

        with pytest.raises(ValueError, match="Block with tag 'missing' not found"):
            root.get("missing")

    def test_get_finds_self_if_matches(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])

        result = root.get("root")
        assert result.text == "Root"


class TestBlockGetOrNone:
    """Tests for get_or_none() method."""

    def test_get_or_none_finds_block(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])
            root /= Block("B", tags=["item"])

        result = root.get_or_none("item")
        assert result is not None
        assert result.text == "B"

    def test_get_or_none_returns_none_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])

        result = root.get_or_none("missing")
        assert result is None

    def test_get_or_none_finds_nested_block(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["section"]) as a:
                a /= Block("Nested", tags=["target"])

        result = root.get_or_none("target")
        assert result is not None
        assert result.text == "Nested"


class TestBlockGetPath:
    """Tests for get_path() method."""

    def test_get_path_with_single_int(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["a"])
            root /= Block("B", tags=["b"])
            root /= Block("C", tags=["c"])

        result = root.get_path(1)
        assert result.text == "B"

    def test_get_path_with_tuple_depth_one(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["a"])
            root /= Block("B", tags=["b"])

        result = root.get_path((0,))
        assert result.text == "A"

    def test_get_path_with_tuple_depth_two(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["a"]) as a:
                a /= Block("A1", tags=["a1"])
                a /= Block("A2", tags=["a2"])
            root /= Block("B", tags=["b"])

        result = root.get_path((0, 1))
        assert result.text == "A2"

    def test_get_path_with_tuple_depth_three(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["a"]) as a:
                with a("A1", tags=["a1"]) as a1:
                    a1 /= Block("Deep1", tags=["deep1"])
                    a1 /= Block("Deep2", tags=["deep2"])

        result = root.get_path((0, 0, 1))
        assert result.text == "Deep2"

    def test_get_path_with_list(self):
        with Block("Root", tags=["root"]) as root:
            with root("A", tags=["a"]) as a:
                a /= Block("A1", tags=["a1"])
                a /= Block("A2", tags=["a2"])

        result = root.get_path([0, 0])
        assert result.text == "A1"

    def test_get_path_raises_on_invalid_index(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["a"])

        with pytest.raises(IndexError):
            root.get_path(5)

    def test_get_path_raises_on_empty_path(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["a"])

        with pytest.raises(IndexError):
            root.get_path(())


class TestBlockGetSchema:
    """Tests for get_schema() method."""

    def test_get_schema_finds_schema(self):
        with Block("Root", tags=["root"]) as root:
            root /= BlockSchema("thinking", tags=["thinking"])
            root /= Block("Regular", tags=["regular"])

        result = root.get_schema("thinking")
        assert isinstance(result, BlockSchema)
        assert result.text == "thinking"

    def test_get_schema_raises_when_not_schema(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("Regular", tags=["target"])

        with pytest.raises(ValueError, match="is not a BlockSchema"):
            root.get_schema("target")

    def test_get_schema_raises_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= BlockSchema("thinking", tags=["thinking"])

        with pytest.raises(ValueError, match="Block with tag 'missing' not found"):
            root.get_schema("missing")

    def test_get_schema_finds_nested_schema(self):
        with Block("Root", tags=["root"]) as root:
            with root("Section", tags=["section"]) as section:
                section /= BlockSchema("nested_schema", tags=["nested"])

        result = root.get_schema("nested")
        assert isinstance(result, BlockSchema)
        assert result.text == "nested_schema"


class TestBlockGetList:
    """Tests for get_list() method."""

    def test_get_list_finds_block_list(self):
        with Block("Root", tags=["root"]) as root:
            block_list = BlockList(tags=["items"])
            block_list /= Block("Item1")
            block_list /= Block("Item2")
            root /= block_list

        result = root.get_list("items")
        assert isinstance(result, BlockList)
        assert len(result) == 2

    def test_get_list_returns_empty_list_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])

        result = root.get_list("missing")
        assert isinstance(result, BlockList)
        assert len(result) == 0

    def test_get_list_raises_when_not_block_list(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("Regular", tags=["target"])

        with pytest.raises(ValueError, match="is not a BlockList"):
            root.get_list("target")

    def test_get_list_finds_nested_list(self):
        with Block("Root", tags=["root"]) as root:
            with root("Section", tags=["section"]) as section:
                block_list = BlockList(tags=["nested_list"])
                block_list /= Block("Item")
                section /= block_list

        result = root.get_list("nested_list")
        assert isinstance(result, BlockList)
        assert len(result) == 1


class TestBlockGetAllSchemas:
    """Tests for get_all_schemas() method."""

    def test_get_all_schemas_finds_all(self):
        with Block("Root", tags=["root"]) as root:
            root /= BlockSchema("Schema1", tags=["schema"])
            root /= BlockSchema("Schema2", tags=["schema"])
            root /= Block("Regular", tags=["other"])

        result = root.get_all_schemas("schema")
        assert len(result) == 2
        assert all(isinstance(s, BlockSchema) for s in result)
        assert result[0].text == "Schema1"
        assert result[1].text == "Schema2"

    def test_get_all_schemas_returns_empty_list_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])

        result = root.get_all_schemas("missing")
        assert result == []

    def test_get_all_schemas_raises_when_mixed_types(self):
        with Block("Root", tags=["root"]) as root:
            root /= BlockSchema("Schema", tags=["mixed"])
            root /= Block("Regular", tags=["mixed"])

        with pytest.raises(ValueError, match="are not all BlockSchemas"):
            root.get_all_schemas("mixed")

    def test_get_all_schemas_finds_nested_schemas(self):
        with Block("Root", tags=["root"]) as root:
            with root("Section", tags=["section"]) as section:
                section /= BlockSchema("Nested1", tags=["target"])
                section /= BlockSchema("Nested2", tags=["target"])

        result = root.get_all_schemas("target")
        assert len(result) == 2


class TestBlockGetChildrenByTag:
    """Tests for get_children_by_tag() method."""

    def test_get_children_by_tag_finds_direct_children(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["item"])
            root /= Block("B", tags=["item"])
            root /= Block("C", tags=["other"])

        result = root.get_children_by_tag("item")
        assert len(result) == 2
        assert result[0].text == "A"
        assert result[1].text == "B"

    def test_get_children_by_tag_ignores_nested(self):
        with Block("Root", tags=["root"]) as root:
            with root("Section", tags=["section"]) as section:
                section /= Block("Nested", tags=["item"])
            root /= Block("Direct", tags=["item"])

        result = root.get_children_by_tag("item")
        assert len(result) == 1
        assert result[0].text == "Direct"

    def test_get_children_by_tag_returns_empty_when_not_found(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["section"])
            root /= Block("B", tags=["other"])

        result = root.get_children_by_tag("missing")
        assert result == []

    def test_get_children_by_tag_with_multiple_tags(self):
        with Block("Root", tags=["root"]) as root:
            root /= Block("A", tags=["item", "special"])
            root /= Block("B", tags=["item"])
            root /= Block("C", tags=["special"])

        result = root.get_children_by_tag("special")
        assert len(result) == 2
        assert result[0].text == "A"
        assert result[1].text == "C"
