import pytest
from pydantic import BaseModel, Field

from chatboard.prompt.block_parser2 import (
    ActionContent,
    ParsedBlock,
    parse_blocks,
    parse_json_like
)


class Attack(BaseModel):
    name: str = Field(description="The name of the attack")
    description: str = Field(description="The description of the attack")
    damage: int = Field(description="The damage of the attack")


class RunAway(BaseModel):
    name: str = Field(description="The name of the run away")
    description: str = Field(description="The description of the run away")
    damage: int = Field(description="The damage of the run away")


class GiveQuest(BaseModel):
    name: str = Field(description="The name of the give quest")
    description: str = Field(description="The description of the give quest")
    reward: int = Field(description="The reward of the give quest")


def test_parse_json_like():
    """Test JSON-like string parsing."""
    # Test with unquoted keys
    json_like = """{
        name: "test",
        value: 123,
        nested: {
            key: "value",
        }
    }"""
    parsed = parse_json_like(json_like)
    assert parsed == {
        "name": "test",
        "value": 123,
        "nested": {"key": "value"}
    }

    # Test with trailing commas
    json_like = '{"name": "test", "values": [1, 2, 3,], }'
    parsed = parse_json_like(json_like)
    assert parsed == {"name": "test", "values": [1, 2, 3]}


def test_parse_blocks_with_registered_models():
    """Test parsing blocks with registered model classes."""
    ActionContent.register_model("attack", Attack)
    ActionContent.register_model("give_quest", GiveQuest)
    
    xml = """
    <Plan>
        <step index="1">First step</step>
        <step index="2">Second step</step>
    </Plan>
    <Thought>
        I should attack and then give a quest.
    </Thought>
    <Actions>
        <Action name="attack" id="1">
            {
                name: "Dragon Slash",
                description: "A powerful slash attack",
                damage: 50
            }
        </Action>
        <Action name="give_quest" id="2">
            {
                name: "Slay the Dragon",
                description: "Defeat the mighty dragon",
                reward: 1000
            }
        </Action>
    </Actions>
    """
    
    result = parse_blocks(xml)
    
    # Check Plan section
    assert isinstance(result.content, list)
    plan = result.content[0]
    assert plan.tag == "Plan"
    assert isinstance(plan.content, list)
    assert len(plan.content) == 2
    assert plan.content[0].tag == "step"
    assert plan.content[0].attributes["index"] == "1"
    assert plan.content[0].content == "First step"
    
    # Check Thought section
    thought = result.content[1]
    assert thought.tag == "Thought"
    assert "should attack and then give a quest" in thought.content
    
    # Check Actions section
    actions = result.content[2]
    assert actions.tag == "Actions"
    assert isinstance(actions.content, list)
    assert len(actions.content) == 2
    
    # Check first action
    attack = actions.content[0]
    assert attack.tag == "Action"
    assert attack.attributes["name"] == "attack"
    assert attack.attributes["id"] == "1"
    assert isinstance(attack.content, Attack)
    assert attack.content.name == "Dragon Slash"
    assert attack.content.damage == 50
    
    # Check second action
    quest = actions.content[1]
    assert quest.tag == "Action"
    assert quest.attributes["name"] == "give_quest"
    assert quest.attributes["id"] == "2"
    assert isinstance(quest.content, GiveQuest)
    assert quest.content.name == "Slay the Dragon"
    assert quest.content.reward == 1000


def test_parse_blocks_with_invalid_xml():
    """Test handling of invalid XML."""
    with pytest.raises(ValueError, match="Invalid XML structure"):
        parse_blocks("<unclosed>test")


def test_parse_blocks_with_invalid_json():
    """Test handling of invalid JSON in action content."""
    ActionContent.register_model("attack", Attack)
    
    xml = """
    <Plan>Test</Plan>
    <Actions>
        <Action name="attack" id="1">
            {invalid json content}
        </Action>
    </Actions>
    """
    
    result = parse_blocks(xml)
    assert len(result.content) == 2
    actions = result.content[1]
    # Should contain raw content when JSON parsing fails
    assert isinstance(actions.content[0].content, str)


def test_parse_blocks_with_empty_sections():
    """Test parsing with empty or missing sections."""
    xml = "<Plan></Plan><Thought></Thought><Actions></Actions>"
    result = parse_blocks(xml)
    
    assert isinstance(result.content, list)
    assert len(result.content) == 3
    assert result.content[0].content == ""
    assert result.content[1].content == ""
    assert result.content[2].content == [] 