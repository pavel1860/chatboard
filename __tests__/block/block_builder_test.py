"""
Unit tests for StreamingBlockBuilder and related block building functionality.

Tests cover:
- BlockBuilder lifecycle (start, append, finish)
- StreamingBlockBuilder streaming operations (open_view, append, close_view)
- build_from_dict with simple dicts, nested dicts, and lists
- BlockListSchema with Pydantic models
- Parser integration with streaming XML chunks
"""

import pytest
from pydantic import BaseModel, Field

from chatboard.block.block9.block import Block, BlockSchema, BlockListSchema
from chatboard.block.block9.block_builder import (
    BlockBuilder,
    StreamingBlockBuilder,
    traverse_dict,
    instantiate_from_dict,
)


# =============================================================================
# Test Fixtures - Pydantic Models
# =============================================================================

class Attack(BaseModel):
    """Attack an enemy with a weapon."""
    weapon: str = Field(description="The weapon to use")
    target: str = Field(description="The target to attack")


class GiveQuest(BaseModel):
    """Give a quest to the player."""
    reward: int = Field(description="The reward amount")
    description: str = Field(description="The quest description")


# =============================================================================
# Test Fixtures - Schemas
# =============================================================================

@pytest.fixture
def simple_schema():
    """A simple schema with two views."""
    with BlockSchema('response') as schema:
        schema.view('thought')
        schema.view('answer')
    return schema


@pytest.fixture
def nested_schema():
    """A schema with nested views."""
    with BlockSchema('response') as schema:
        with schema.view('metadata') as meta:
            meta.view('timestamp')
            meta.view('version')
        schema.view('content')
    return schema


@pytest.fixture
def list_schema():
    """A schema with a BlockListSchema containing Pydantic models."""
    with BlockSchema('response') as schema:
        schema.view('thought')
        schema.view('answer')
        tools = schema.view_list('tool')
        tools.key_field('name', type=str)
        tools.register(Attack)
        tools.register(GiveQuest)
    return schema


@pytest.fixture
def simple_list_schema():
    """A schema with a simple list (no Pydantic models)."""
    with BlockSchema('root') as schema:
        schema.view('title')
        items = schema.view_list('item')
        items.key_field('id', type=str)
        with items.view('basic', tags=['basic']) as basic:
            basic.field('id', type=str)
    return schema


# =============================================================================
# Tests for traverse_dict
# =============================================================================

def test_traverse_dict_simple():
    """Test traversing a flat dictionary."""
    data = {'a': 1, 'b': 2}
    result = list(traverse_dict(data))

    assert len(result) == 2
    assert result[0] == ('a', 1, [0], ['a'])
    assert result[1] == ('b', 2, [1], ['b'])


def test_traverse_dict_nested():
    """Test traversing a nested dictionary."""
    data = {'outer': {'inner': 'value'}}
    result = list(traverse_dict(data))

    assert len(result) == 2
    # First yields the outer key with None value
    assert result[0] == ('outer', None, [0], ['outer'])
    # Then yields the inner key-value
    assert result[1] == ('inner', 'value', [0, 0], ['outer', 'inner'])


def test_traverse_dict_list():
    """Test traversing a dictionary with a list."""
    data = {'items': ['a', 'b', 'c']}
    result = list(traverse_dict(data))

    assert len(result) == 3
    assert result[0] == ('items', 'a', [0], ['items'])
    assert result[1] == ('items', 'b', [0], ['items'])
    assert result[2] == ('items', 'c', [0], ['items'])


def test_traverse_dict_mixed():
    """Test traversing a dictionary with mixed types."""
    data = {
        'simple': 'value',
        'nested': {'key': 'val'},
        'list': [1, 2]
    }
    result = list(traverse_dict(data))

    keys = [r[0] for r in result]
    assert 'simple' in keys
    assert 'nested' in keys
    assert 'key' in keys
    assert 'list' in keys


# =============================================================================
# Tests for BlockBuilder
# =============================================================================

def test_block_builder_init(simple_schema):
    """Test BlockBuilder initialization."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)

    assert builder.schema == child_schema
    assert builder.block is None
    assert not builder.is_started
    assert not builder.is_finished
    assert not builder.is_list


def test_block_builder_start(simple_schema):
    """Test starting a block."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)

    block = builder.start(value='initial')

    assert builder.is_started
    assert builder.block is not None
    assert block is builder.block


def test_block_builder_start_twice_raises(simple_schema):
    """Test that starting twice raises an error."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)
    builder.start()

    with pytest.raises(ValueError, match="already started"):
        builder.start()


def test_block_builder_append_before_start_raises(simple_schema):
    """Test that appending before start raises an error."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)

    with pytest.raises(ValueError, match="not started"):
        builder.append("content")


def test_block_builder_append(simple_schema):
    """Test appending content to a block."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)
    builder.start()

    result = builder.append("hello")

    assert result is builder.block
    assert len(builder.block.children) == 1


def test_block_builder_finish(simple_schema):
    """Test finishing a block."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)
    builder.start()

    block = builder.finish(postfix='</thought>')

    assert builder.is_finished
    assert block is builder.block
    assert block.postfix is not None


def test_block_builder_finish_before_start_raises(simple_schema):
    """Test that finishing before start raises an error."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)

    with pytest.raises(ValueError, match="not started"):
        builder.finish()


def test_block_builder_get_child_schema(nested_schema):
    """Test getting a child schema."""
    meta_schema = nested_schema.get_one('metadata')
    builder = BlockBuilder(meta_schema)

    timestamp_schema = builder.get_child_schema('timestamp')

    assert timestamp_schema is not None
    assert 'timestamp' in timestamp_schema.tags


def test_block_builder_get_child_schema_not_found(simple_schema):
    """Test getting a non-existent child schema."""
    child_schema = simple_schema.get_one('thought')
    builder = BlockBuilder(child_schema)

    with pytest.raises(ValueError, match="does not exists"):
        builder.get_child_schema('nonexistent')


def test_block_builder_is_list(list_schema):
    """Test is_list property for BlockListSchema."""
    list_child = list_schema.get_one('tool')
    builder = BlockBuilder(list_child)

    assert builder.is_list


# =============================================================================
# Tests for StreamingBlockBuilder
# =============================================================================

def test_streaming_builder_init(simple_schema):
    """Test StreamingBlockBuilder initialization."""
    builder = StreamingBlockBuilder(simple_schema)

    assert builder.root_schema is not None
    assert builder.root is None
    assert len(builder.stack) == 0


def test_streaming_builder_result_before_build_raises(simple_schema):
    """Test accessing result before building raises an error."""
    builder = StreamingBlockBuilder(simple_schema)

    with pytest.raises(ValueError, match="No root block"):
        _ = builder.result


def test_streaming_builder_current_empty_stack(simple_schema):
    """Test current property with empty stack."""
    builder = StreamingBlockBuilder(simple_schema)

    assert builder.current is None


def test_streaming_builder_open_view_simple(simple_schema):
    """Test opening a simple view."""
    builder = StreamingBlockBuilder(simple_schema)

    blocks = builder.open_view('thought', 'content')

    assert len(blocks) == 1
    assert len(builder.stack) == 1
    assert builder.root is not None


def test_streaming_builder_open_and_close_view(simple_schema):
    """Test opening and closing a view."""
    builder = StreamingBlockBuilder(simple_schema)

    builder.open_view('thought', 'content')
    block = builder.close_view()

    assert len(builder.stack) == 0
    assert 'thought' in block.tags


def test_streaming_builder_nested_views(nested_schema):
    """Test nested view operations."""
    builder = StreamingBlockBuilder(nested_schema)

    # Open outer view
    builder.open_view('metadata')
    assert len(builder.stack) == 1

    # Open inner view
    builder.open_view('timestamp', 'now')
    assert len(builder.stack) == 2

    # Close inner
    builder.close_view()
    assert len(builder.stack) == 1

    # Close outer
    builder.close_view()
    assert len(builder.stack) == 0

    result = builder.result
    assert 'metadata' in result.tags


def test_streaming_builder_append(simple_schema):
    """Test appending content to current view."""
    builder = StreamingBlockBuilder(simple_schema)
    builder.open_view('thought')

    result = builder.append('hello world')

    assert result is not None


def test_streaming_builder_append_no_open_block_raises(simple_schema):
    """Test appending without an open block raises an error."""
    builder = StreamingBlockBuilder(simple_schema)

    with pytest.raises(ValueError, match="No open block"):
        builder.append('content')


def test_streaming_builder_current_path(nested_schema):
    """Test getting the current path returns a list."""
    builder = StreamingBlockBuilder(nested_schema)

    builder.open_view('metadata')
    builder.open_view('timestamp')

    path = builder.current_path()

    # current_path returns a list of tag names
    assert isinstance(path, list)
    # Stack should have 2 items after opening 2 views
    assert len(builder.stack) == 2


# =============================================================================
# Tests for build_from_dict
# =============================================================================

def test_build_from_dict_simple(simple_schema):
    """Test building from a simple dictionary."""
    builder = StreamingBlockBuilder(simple_schema)

    result = builder.build_from_dict({
        'thought': 'I am thinking...',
        'answer': 'This is my answer'
    })

    assert result is not None
    assert len(result.children) == 2


def test_build_from_dict_with_pydantic_list(list_schema):
    """Test building from dict with Pydantic models in list."""
    builder = StreamingBlockBuilder(list_schema)

    result = builder.build_from_dict({
        'thought': 'I am thinking...',
        'answer': 'This is my answer',
        'tool': [
            Attack(weapon='sword', target='head'),
            GiveQuest(reward=100, description='kill the dragon')
        ]
    })

    assert result is not None
    # Should have thought, answer, and tool children
    assert len(result.children) == 3

    # Find the tool child
    tool_child = None
    for child in result.children:
        if 'tool' in child.tags:
            tool_child = child
            break

    assert tool_child is not None
    # Tool should have 2 items (Attack and GiveQuest)
    assert len(tool_child.children) == 2


def test_build_from_dict_single_pydantic_item(list_schema):
    """Test building from dict with a single Pydantic model."""
    builder = StreamingBlockBuilder(list_schema)

    result = builder.build_from_dict({
        'thought': 'thinking',
        'answer': 'answering',
        'tool': [
            Attack(weapon='bow', target='chest')
        ]
    })

    assert result is not None

    # Find tool child
    tool_child = None
    for child in result.children:
        if 'tool' in child.tags:
            tool_child = child
            break

    assert tool_child is not None
    assert len(tool_child.children) == 1


# =============================================================================
# Tests for instantiate_from_dict
# =============================================================================

def test_instantiate_from_dict_simple(simple_schema):
    """Test simple dictionary instantiation."""
    result = instantiate_from_dict(simple_schema, {
        'thought': 'thinking...',
        'answer': 'the answer'
    })

    assert result is not None
    assert len(result.children) == 2


# =============================================================================
# Tests for BlockSchema.inst_dict
# =============================================================================

def test_inst_dict_simple(simple_schema):
    """Test inst_dict with simple values."""
    result = simple_schema.inst_dict({
        'thought': 'I am thinking...',
        'answer': 'This is my answer'
    })

    assert result is not None
    assert len(result.children) == 2

    # Check tags on children
    child_tags = [child.tags[0] for child in result.children if child.tags]
    assert 'thought' in child_tags
    assert 'answer' in child_tags


def test_inst_dict_with_list_schema(list_schema):
    """Test inst_dict with BlockListSchema and Pydantic models."""
    result = list_schema.inst_dict({
        'thought': 'I am thinking...',
        'answer': 'This is my answer',
        'tool': [
            Attack(weapon='sword', target='head'),
            GiveQuest(reward=100, description='kill the dragon')
        ]
    })

    assert result is not None
    assert isinstance(result, Block)

    # Result should have 3 direct children: thought, answer, tool
    assert len(result.children) == 3

    # Verify structure
    tags_found = set()
    for child in result.children:
        if child.tags:
            for t in child.tags:
                tags_found.add(t)

    assert 'thought' in tags_found
    assert 'answer' in tags_found
    assert 'tool' in tags_found


def test_inst_dict_result_is_block(simple_schema):
    """Test that inst_dict returns a Block instance."""
    result = simple_schema.inst_dict({
        'thought': 'test',
        'answer': 'test'
    })

    assert isinstance(result, Block)


def test_inst_dict_preserves_schema_tags(simple_schema):
    """Test that schema tags are preserved in the result."""
    result = simple_schema.inst_dict({
        'thought': 'content',
        'answer': 'content'
    })

    assert 'response' in result.tags


# =============================================================================
# Tests for Parser Integration (Async)
# =============================================================================

@pytest.mark.asyncio
async def test_parser_simple_xml():
    """Test parsing simple XML chunks."""
    from chatboard.prompt.fbp_process import Stream, Parser

    chunks = ['<', 'item', ' id=', '"1"', '>', 'hello', '</', 'item', '>']

    with Block(role="system") as blk:
        with blk.view("item", tags=["output"]) as schema:
            schema.field("id", type=int)

    parser = Parser(schema)
    stream = Stream.from_list(chunks, name="test_stream")

    pipe = stream | parser
    async for _ in pipe:
        pass

    result = pipe.result
    assert result is not None
    assert 'item' in result.tags


@pytest.mark.asyncio
async def test_parser_nested_xml():
    """Test parsing nested XML chunks."""
    from chatboard.prompt.fbp_process import Stream, Parser

    chunks = [
        '<', 'root', '>',
        '<', 'thought', '>', 'thinking', '</', 'thought', '>',
        '<', 'answer', '>', 'responding', '</', 'answer', '>',
        '</', 'root', '>'
    ]

    with Block(role="system") as blk:
        with blk.view("root") as schema:
            schema.view("thought", str)
            schema.view("answer", str)

    parser = Parser(schema)
    stream = Stream.from_list(chunks, name="test_stream")

    pipe = stream | parser
    async for _ in pipe:
        pass

    result = pipe.result
    assert result is not None


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_build_from_dict_empty(simple_schema):
    """Test building from an empty dictionary."""
    builder = StreamingBlockBuilder(simple_schema)

    result = builder.build_from_dict({})

    assert result is not None
    # Root should exist but with no children added from dict
    assert len(result.children) == 0


def test_build_from_dict_unknown_key_raises(simple_schema):
    """Test handling of unknown keys in dictionary."""
    builder = StreamingBlockBuilder(simple_schema)

    # This should raise an error for unknown key
    with pytest.raises(ValueError):
        builder.build_from_dict({
            'thought': 'valid',
            'unknown_key': 'invalid'
        })


def test_list_schema_missing_attrs_raises():
    """Test BlockListSchema without attrs raises error."""
    with BlockSchema('response') as schema:
        items = schema.view_list('item')
        items.key_field('id', type=str)
        with items.view('basic', tags=['basic']) as basic:
            basic.field('id', type=str)

    builder = StreamingBlockBuilder(schema)

    # This should raise when trying to add a list item without attrs
    with pytest.raises(ValueError):
        builder.open_view('item', attrs=None)


def test_build_from_dict_multiple_same_type_pydantic(list_schema):
    """Test multiple Pydantic models of the same type."""
    builder = StreamingBlockBuilder(list_schema)

    result = builder.build_from_dict({
        'thought': 'thinking',
        'answer': 'answering',
        'tool': [
            Attack(weapon='sword', target='head'),
            Attack(weapon='bow', target='chest'),
            Attack(weapon='axe', target='legs')
        ]
    })

    # Find tool child
    tool_child = None
    for child in result.children:
        if 'tool' in child.tags:
            tool_child = child
            break

    assert tool_child is not None
    # Should have 3 Attack items
    assert len(tool_child.children) == 3


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_schema_build_context_alias(simple_schema):
    """Test that SchemaBuildContext alias works."""
    from chatboard.block.block9.block_builder import SchemaBuildContext

    builder = SchemaBuildContext(simple_schema)

    assert isinstance(builder, StreamingBlockBuilder)


def test_block_build_context_alias(simple_schema):
    """Test that BlockBuildContext alias works."""
    from chatboard.block.block9.block_builder import BlockBuildContext

    child_schema = simple_schema.get_one('thought')
    builder = BlockBuildContext(child_schema)

    assert isinstance(builder, BlockBuilder)
