import pytest
from pydantic import BaseModel, Field
from chatboard.prompt.block_parser import (
    parse_blocks,
    ActionContent,
    ParsedOutput,
    parse_json_like,
    clean_xml_text
)

# Test models
class Attack(BaseModel):
    name: str = Field(description="The name of the attack")
    description: str = Field(description="The description of the attack")
    damage: int = Field(description="The damage of the attack")

class GiveQuest(BaseModel):
    name: str = Field(description="The name of the give quest")
    description: str = Field(description="The description of the give quest")
    reward: int = Field(description="The reward of the give quest")


def test_clean_xml_text():
    """Test XML text cleaning functionality."""
    text = """
        This is a
        multiline text
        with extra    spaces
    """
    assert clean_xml_text(text) == "This is a multiline text with extra spaces"


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


def test_action_content_registration():
    """Test model registration and retrieval."""
    ActionContent.register_model("attack", Attack)
    ActionContent.register_model("give_quest", GiveQuest)
    
    assert ActionContent.get_model_class("attack") == Attack
    assert ActionContent.get_model_class("give_quest") == GiveQuest
    assert ActionContent.get_model_class("unknown") is None


def test_parse_blocks_with_registered_models():
    """Test parsing blocks with registered model classes."""
    ActionContent.register_model("attack", Attack)
    ActionContent.register_model("give_quest", GiveQuest)
    
    xml = """
    <Observation>
        The dragon appears hostile.
    </Observation>
    <Thought>
        I should attack the dragon and then give a quest.
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
    
    assert isinstance(result, ParsedOutput)
    assert "dragon appears hostile" in result.observation
    assert "should attack the dragon" in result.thought
    assert len(result.actions) == 2
    
    # Check first action
    attack = result.actions[0]
    assert attack.name == "attack"
    assert attack.id == "1"
    assert isinstance(attack.content, Attack)
    assert attack.content.name == "Dragon Slash"
    assert attack.content.damage == 50
    
    # Check second action
    quest = result.actions[1]
    assert quest.name == "give_quest"
    assert quest.id == "2"
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
    <Observation>Test</Observation>
    <Thought>Test</Thought>
    <Actions>
        <Action name="attack" id="1">
            {invalid json content}
        </Action>
    </Actions>
    """
    
    result = parse_blocks(xml)
    assert len(result.actions) == 1
    # Should contain raw content when JSON parsing fails
    assert isinstance(result.actions[0].content, str)


def test_parse_blocks_with_empty_sections():
    """Test parsing with empty or missing sections."""
    xml = "<Observation></Observation><Thought></Thought><Actions></Actions>"
    result = parse_blocks(xml)
    
    assert result.observation == ""
    assert result.thought == ""
    assert len(result.actions) == 0 