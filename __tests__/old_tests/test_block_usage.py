import pytest
from chatboard.prompt.legacy.block2 import Block
from pydantic import BaseModel, Field


@pytest.fixture
def b():
    """Create a fresh block instance for each test."""
    return Block()


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


def test_block_builder_and_parser(b):
    """Test the integration between block builder and parser."""
    
    # Create the output format
    with b.title("Output", type="xml") as output_format:
        with b.xml("Observation"):
            b += "the observation of the output"
        with b.xml("Thought"):
            b += "step by step thought"
        with b.xml("Actions"):
            with b.xml("Action", name="attack", id="1"):
                b.model_dump(Attack)
            with b.xml("Action", name="run_away", id="2"):
                b.model_dump(RunAway)
            b += "... action blocks can repeat as many times as needed"
    
    # Register models for parsing
    output_format.register_model("attack", Attack)
    output_format.register_model("run_away", RunAway)
    output_format.register_model("give_quest", GiveQuest)
    
    # Example completion text
    completion = """
    <Observation>
        The dragon appears very hostile and dangerous.
    </Observation>
    <Thought>
        I should try to run away and then give a quest to someone else.
    </Thought>
    <Actions>
        <Action name="run_away" id="1">
            {
                name: "Quick Escape",
                description: "Swiftly retreat from the dragon",
                damage: 0
            }
        </Action>
        <Action name="give_quest" id="2">
            {
                name: "Slay the Dragon",
                description: "Defeat the dangerous dragon in the cave",
                reward: 1000
            }
        </Action>
    </Actions>
    """
    
    # Parse the completion
    result = output_format.parse(completion)
    
    # Verify the parsed output
    assert isinstance(result.content, list)
    assert len(result.content) == 3
    
    # Check Observation section
    observation = result.content[0]
    assert observation.tag == "Observation"
    assert "dragon appears very hostile" in observation.content
    
    # Check Thought section
    thought = result.content[1]
    assert thought.tag == "Thought"
    assert "should try to run away" in thought.content
    
    # Check Actions section
    actions = result.content[2]
    assert actions.tag == "Actions"
    assert isinstance(actions.content, list)
    assert len(actions.content) == 2
    
    # Check first action (run_away)
    escape = actions.content[0]
    assert escape.tag == "Action"
    assert escape.attributes["name"] == "run_away"
    assert escape.attributes["id"] == "1"
    assert isinstance(escape.content, RunAway)
    assert escape.content.name == "Quick Escape"
    assert escape.content.damage == 0
    
    # Check second action (give_quest)
    quest = actions.content[1]
    assert quest.tag == "Action"
    assert quest.attributes["name"] == "give_quest"
    assert quest.attributes["id"] == "2"
    assert isinstance(quest.content, GiveQuest)
    assert quest.content.name == "Slay the Dragon"
    assert quest.content.reward == 1000


def test_block_builder_render(b):
    """Test that the block builder produces the expected XML structure."""
    
    with b.title("Output", type="xml") as output_format:
        with b.xml("Observation"):
            b += "Test observation"
        with b.xml("Thought"):
            b += "Test thought"
        with b.xml("Actions"):
            with b.xml("Action", name="test", id="1"):
                b += "Test action"
    
    rendered = output_format.render()
    
    # Parse the rendered output
    result = output_format.parse(rendered)
    
    # Verify the structure
    assert isinstance(result.content, list)
    assert len(result.content) == 3
    
    # Check Observation section
    observation = result.content[0]
    assert observation.tag == "Observation"
    assert observation.content == "Test observation"
    
    # Check Thought section
    thought = result.content[1]
    assert thought.tag == "Thought"
    assert thought.content == "Test thought"
    
    # Check Actions section
    actions = result.content[2]
    assert actions.tag == "Actions"
    assert isinstance(actions.content, list)
    assert len(actions.content) == 1
    
    # Check action
    action = actions.content[0]
    assert action.tag == "Action"
    assert action.attributes["name"] == "test"
    assert action.attributes["id"] == "1"
    assert action.content == "Test action" 