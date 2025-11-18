"""Integration tests for SpanTree with database operations."""
import pytest
from promptview.model import NamespaceManager, VersionedModel, KeyField, ModelField
from promptview.prompt import Context, component, stream
from promptview.prompt.span_tree import SpanTree
from promptview.block import Block
import pytest_asyncio

# Test model
class Post(VersionedModel):
    """Simple test model for span tree tests."""
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    text: str = ModelField()


@pytest_asyncio.fixture()
async def setup_database():
    """Initialize clean database for each test."""
    await NamespaceManager.initialize_clean()
    yield
    # Cleanup if needed


@pytest.mark.asyncio
async def test_save_parameter(setup_database):
    """Test logging a simple parameter value."""
    ctx = Context()
    target = "test string"

    async with ctx.start_turn():
        span = await SpanTree("test").save()
        value = await span.log_value(target)

        assert value.value == "test string"
        assert value._is_list == False


@pytest.mark.asyncio
async def test_save_regular_list(setup_database):
    """Test logging a list of primitive values."""
    ctx = Context()
    target = [1, 2, 3]

    async with ctx.start_turn():
        span = await SpanTree("test").save()
        value = await span.log_value(target)

        assert value.value == [1, 2, 3]
        


@pytest.mark.asyncio
async def test_save_single_artifact(setup_database):
    """Test logging a single model artifact."""
    ctx = Context()

    async with ctx.start_turn():
        post1 = await Post(title="Post 1", text="Text 1").save()
        span = await SpanTree("test_artifact").save()
        value = await span.log_value(post1)

        assert value.value.id == post1.id
        assert value.value.title == "Post 1"
        assert value._is_list == False


@pytest.mark.asyncio
async def test_save_artifact_list(setup_database):
    """Test logging a list of model artifacts."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        post1 = await Post(title="Post 1", text="Text 1").save()
        post2 = await Post(title="Post 2", text="Text 2").save()
        post3 = await Post(title="Post 3", text="Text 3").save()
        posts = [post1, post2, post3]

        span = await SpanTree("test_artifact").save()
        value = await span.log_value(posts)

        assert value._is_list == True
        assert type(value.value) == list
        assert len(value.value) == 3
        assert value.value[0].id == post1.id
        assert value.value[1].id == post2.id
        assert value.value[2].id == post3.id

    # Test loading from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.values) == 1

    list_value = span_tree.values[0]
    assert list_value._is_list == True
    assert type(list_value.value) == list
    assert len(list_value.value) == 3
    assert list_value.value[0].id == post1.id
    assert list_value.value[1].id == post2.id
    assert list_value.value[2].id == post3.id


@pytest.mark.asyncio
async def test_save_single_block(setup_database):
    """Test logging a single block."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        with Block("Test Block") as blk:
            blk /= "this is a test"

        span = await SpanTree("test_block").save()
        value = await span.log_value(blk)

        # assert value.value.id == blk.id

    # Test loading from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.values) == 1
    # assert span_tree.values[0].value.id == blk.id


@pytest.mark.asyncio
async def test_save_block_list(setup_database):
    """Test logging a list of blocks."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        with Block("Test Block 1") as blk1:
            blk1 /= "this is a test 1"
        with Block("Test Block 2") as blk2:
            blk2 /= "this is a test 2"
        with Block("Test Block 3") as blk3:
            blk3 /= "this is a test 3"

        span = await SpanTree("test_blocks").save()
        value = await span.log_value([blk1, blk2, blk3])

    # Test loading from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    value = span_tree.values[0].value

    assert type(value) == list
    assert len(value) == 3
    assert type(value[0]) == Block
    assert type(value[1]) == Block
    assert type(value[2]) == Block
    # assert value[0].id == blk1.id
    # assert value[1].id == blk2.id
    # assert value[2].id == blk3.id


@pytest.mark.asyncio
async def test_add_child_span(setup_database):
    """Test adding a child span using add_child."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        parent_span = await SpanTree("parent").save()
        child_span = await parent_span.add_child("child", span_type="component")

        # Child should be accessible via children property
        assert len(parent_span.children) == 1
        assert parent_span.children[0].name == "child"
        assert parent_span.children[0].id == child_span.id

        # Child should also be in values as a span value
        span_values = [v for v in parent_span.values if v._is_span]
        assert len(span_values) == 1
        assert span_values[0].span_tree.id == child_span.id

    # Test loading from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.children) == 1
    assert span_tree.children[0].name == "child"


@pytest.mark.asyncio
async def test_multiple_children(setup_database):
    """Test adding multiple child spans."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        parent_span = await SpanTree("parent").save()
        child1 = await parent_span.add_child("child1")
        child2 = await parent_span.add_child("child2")
        child3 = await parent_span.add_child("child3")

        assert len(parent_span.children) == 3
        assert parent_span.children[0].name == "child1"
        assert parent_span.children[1].name == "child2"
        assert parent_span.children[2].name == "child3"

    # Test loading from turn - children should be in execution order
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.children) == 3
    assert span_tree.children[0].name == "child1"
    assert span_tree.children[1].name == "child2"
    assert span_tree.children[2].name == "child3"


@pytest.mark.asyncio
async def test_nested_children(setup_database):
    """Test deeply nested child spans."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        root = await SpanTree("root").save()
        child1 = await root.add_child("child1")
        grandchild1 = await child1.add_child("grandchild1")
        great_grandchild = await grandchild1.add_child("great_grandchild")

        # Test structure
        assert len(root.children) == 1
        assert len(root.children[0].children) == 1
        assert len(root.children[0].children[0].children) == 1
        assert root.children[0].children[0].children[0].name == "great_grandchild"

    # Test loading from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.children) == 1
    assert len(span_tree.children[0].children) == 1
    assert len(span_tree.children[0].children[0].children) == 1
    assert span_tree.children[0].children[0].children[0].name == "great_grandchild"


@pytest.mark.asyncio
async def test_mixed_values_and_children(setup_database):
    """Test span with both regular values and child spans in execution order."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        post1 = await Post(title="Post 1", text="Text 1").save()
        post2 = await Post(title="Post 2", text="Text 2").save()

        parent = await SpanTree("parent").save()

        # Log value, add child, log another value
        await parent.log_value(post1, io_kind="input")
        child = await parent.add_child("child")
        await parent.log_value(post2, io_kind="output")

        # Check execution order in values
        assert len(parent.values) == 3
        assert parent.values[0]._is_span == False  # post1
        assert parent.values[0].value.id == post1.id
        assert parent.values[1]._is_span == True   # child
        assert parent.values[1].span_tree.id == child.id
        assert parent.values[2]._is_span == False  # post2
        assert parent.values[2].value.id == post2.id

        # Children property should extract only child spans
        assert len(parent.children) == 1
        assert parent.children[0].id == child.id

    # Test loading preserves order
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    assert len(span_tree.values) == 3
    assert span_tree.values[0]._is_span == False
    assert span_tree.values[1]._is_span == True
    assert span_tree.values[2]._is_span == False


@pytest.mark.asyncio
async def test_to_dict_serialization(setup_database):
    """Test that to_dict properly serializes span tree with all value types."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        post1 = await Post(title="Post 1", text="Text 1").save()

        with Block("Test Block") as blk:
            blk /= "block content"

        parent = await SpanTree("parent").save()
        await parent.log_value("string param", io_kind="input", name="param1")
        await parent.log_value(post1, io_kind="input", name="post")
        await parent.log_value(blk, io_kind="output", name="block")
        child = await parent.add_child("child")

    # Load and serialize
    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    serialized = span_tree.to_dict()

    # Check structure
    assert serialized["id"] == span_tree.id
    assert serialized["name"] == "parent"
    assert "values" in serialized
    assert len(serialized["values"]) == 4

    # Check parameter serialization
    assert serialized["values"][0]["kind"] == "parameter"
    assert serialized["values"][0]["name"] == "param1"

    # Check model serialization
    assert serialized["values"][1]["kind"] == "model"
    assert serialized["values"][1]["name"] == "post"

    # Check block serialization
    assert serialized["values"][2]["kind"] == "block"
    assert serialized["values"][2]["name"] == "block"

    # Check child span serialization (recursive)
    assert serialized["values"][3]["kind"] == "span"
    assert isinstance(serialized["values"][3]["value"], dict)
    assert serialized["values"][3]["value"]["name"] == "child"


@pytest.mark.asyncio
async def test_to_dict_no_duplicate_children(setup_database):
    """Test that to_dict doesn't duplicate children in separate field."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        parent = await SpanTree("parent").save()
        await parent.add_child("child1")
        await parent.add_child("child2")

    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]
    serialized = span_tree.to_dict()

    # Should NOT have a separate "children" field
    assert "children" not in serialized

    # Children should be in values
    span_values = [v for v in serialized["values"] if v["kind"] == "span"]
    assert len(span_values) == 2
    assert span_values[0]["value"]["name"] == "child1"
    assert span_values[1]["value"]["name"] == "child2"


@pytest.mark.asyncio
async def test_traverse_with_children(setup_database):
    """Test that traverse correctly iterates through all spans."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        root = await SpanTree("root").save()
        child1 = await root.add_child("child1")
        child2 = await root.add_child("child2")
        grandchild = await child1.add_child("grandchild")

    span_trees = await SpanTree.from_turn(ctx.turn.id)
    assert len(span_trees) == 1  # One top-level span
    span_tree = span_trees[0]

    # Traverse should visit all spans
    spans = list(span_tree.traverse())
    assert len(spans) == 4
    assert spans[0].name == "root"
    assert spans[1].name == "child1"
    assert spans[2].name == "grandchild"
    assert spans[3].name == "child2"


@pytest.mark.asyncio
async def test_multiple_top_level_spans(setup_database):
    """Test that multiple top-level spans in a single turn are properly tracked."""
    ctx = Context()

    async with ctx.start_turn() as ctx:
        # Create first top-level span
        span1 = await SpanTree("preprocessing").save()
        await span1.log_value("preprocess data", io_kind="output")

        # Create second top-level span
        span2 = await SpanTree("main_task").save()
        post = await Post(title="Post 1", text="Text 1").save()
        await span2.log_value(post, io_kind="output")

        # Create third top-level span
        span3 = await SpanTree("postprocessing").save()
        await span3.log_value("cleanup", io_kind="output")

    # Load from turn
    span_trees = await SpanTree.from_turn(ctx.turn.id)

    # Should have 3 top-level spans
    assert len(span_trees) == 3

    # Verify each span
    assert span_trees[0].name == "preprocessing"
    assert span_trees[0].root.path == "1"
    assert span_trees[0].index == 0
    assert len(span_trees[0].values) == 1

    assert span_trees[1].name == "main_task"
    assert span_trees[1].root.path == "2"
    assert span_trees[1].index == 1
    assert len(span_trees[1].values) == 1

    assert span_trees[2].name == "postprocessing"
    assert span_trees[2].root.path == "3"
    assert span_trees[2].index == 2
    assert len(span_trees[2].values) == 1


@pytest.mark.asyncio
async def test_context_tracks_multiple_top_level_spans(setup_database):
    """Test that Context properly tracks multiple top-level spans."""
    ctx = Context()

    async with ctx.start_turn():
        # Initially no spans
        assert len(ctx.top_level_spans) == 0

        # Create first top-level span
        span1 = await SpanTree("span1").save()
        assert len(ctx.top_level_spans) == 1
        assert ctx.root_span == span1
        assert ctx.top_level_spans[0] == span1

        # Create second top-level span
        span2 = await SpanTree("span2").save()
        assert len(ctx.top_level_spans) == 2
        assert ctx.root_span == span1  # root_span still points to first
        assert ctx.top_level_spans[0] == span1
        assert ctx.top_level_spans[1] == span2

        # Create third top-level span
        span3 = await SpanTree("span3").save()
        assert len(ctx.top_level_spans) == 3
        assert ctx.top_level_spans[2] == span3

        # Test get_span with multiple top-level spans
        assert ctx.get_span([0]) == span1
        assert ctx.get_span([1]) == span2
        assert ctx.get_span([2]) == span3
