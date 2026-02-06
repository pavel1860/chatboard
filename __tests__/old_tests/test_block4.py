import pytest
from chatboard.prompt import Block as blk
from chatboard.block.style import style_manager


@pytest.fixture
def setup_style_manager():
    """Setup style manager with test rules."""
    style_manager.add_rule("task", {
        "title": "md",
        "heading_level": 2,
        "bold": True
    })

    style_manager.add_rule(".important", {
        "bullet": "*",
        "bold": True
    })

    style_manager.add_rule("rules", {
        "title": "md",
        "bullet": "number",
        "indent": 1
    })

    style_manager.add_rule("output_format", {
        "bullet": "-",
        "code": True
    })

    # Add a more complex selector
    style_manager.add_rule("task .nested", {
        "indent": 4,
        "italic": True
    })

    # # Register block types
    # blk.register(type(None), BaseBlock)
    # blk.register(str, StrBlock)
    # blk.register(dict, DictBlock)


def test_simple_block_with_tags():
    """Test creating a simple block with tags."""
    with blk(tags=["system"]) as b:
        b += "you are a helpful assistant"

    assert b.root.tags == ["system"]
    assert len(b.root.items) == 1
    assert b.root.items[0].content == "you are a helpful assistant"


def test_multiple_items_in_block():
    """Test adding multiple items to a block."""
    with blk(tags=["system"]) as b:
        b += "you are a helpful assistant"
        b += "you are an AI assistant"

    assert len(b.root.items) == 2
    assert b.root.items[0].content == "you are a helpful assistant"
    assert b.root.items[1].content == "you are an AI assistant"


def test_nested_block_with_tags():
    """Test creating nested blocks with tags."""
    with blk(tags=["system"]) as b:
        b += "you are a helpful assistant"
        with b("Task", tags=["task"]):
            b += "this is task you need to complete"

    assert len(b.root.items) == 2     
    assert b.root.items[0].content == "you are a helpful assistant"
    assert b.root.items[1].content == "Task"
    assert len(b.root.items[1].items) == 1
    assert b.root.items[1].items[0].content == "this is task you need to complete"


def test_complex_nested_structure():
    """Test creating a complex nested block structure."""
    with blk(tags=["system"]) as b:
        b += "you are a helpful assistant"
        with b("Task", tags=["task"]):
            b += "this is task you need to complete"
        with b("Rules", tags=["rules"]):
            b += "this is rule 1"
            with b("this is main rule 2"):
                b += "this is is sub sub rule 1"
                b += "this is is sub sub rule 2"
                b += "this is is sub sub rule 3"
            b += "this is rule 3"
            b += "this is rule 4"
            with b("this is main rule 5"):
                b += "this is is sub sub rule 4"
                b += "this is is sub sub rule 5"
                b += "this is is sub sub rule 6"
    
    # Add assertions to verify the complex structure
    assert len(b.root.items) == 3
    assert b.root.items[0].content == "you are a helpful assistant"
    assert b.root.items[1].content == "Task"
    assert b.root.items[2].content == "Rules"
    
    # Check Rules section structure
    rules_block = b.root.items[2]
    assert len(rules_block.items) == 5
    assert rules_block.items[0].content == "this is rule 1"
    assert rules_block.items[1].content == "this is main rule 2"
    assert len(rules_block.items[1].items) == 3
    assert rules_block.items[2].content == "this is rule 3"
    assert rules_block.items[3].content == "this is rule 4"
    assert rules_block.items[4].content == "this is main rule 5"