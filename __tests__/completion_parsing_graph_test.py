import pytest




from chatboard.text.llms.completion_parsing2 import OutputParser
import re
from typing import Union
from chatboard.text.llms.views import BaseModel, Field
# from pydantic import BaseModel, Field


class BasicClass(BaseModel):
    basic_field1: str
    basic_field2: int
    basic_field3: bool


basic_text = """
basic_field1: hello
basic_field2: 10
basic_field3: True
"""



def test_base_model():
    output_parser = OutputParser(BasicClass)
    output = output_parser.parse(basic_text)

    assert output.basic_field1 == 'hello'
    assert output.basic_field2 == 10
    assert output.basic_field3 == True



class NestedClass(BaseModel):
    top_field1: str
    top_field2: int
    top_field3: bool
    nested: BasicClass

nested_text = """
top_field1: hello
top_field2: 10
top_field3: True
nested:
    basic_field1: hello
    basic_field2: 10
    basic_field3: True
"""


def test_nested_model():
    output_parser = OutputParser(NestedClass)
    output = output_parser.parse(nested_text)

    assert output.top_field1 == 'hello'
    assert output.top_field2 == 10
    assert output.top_field3 == True
    assert output.nested.basic_field1 == 'hello'
    assert output.nested.basic_field2 == 10
    assert output.nested.basic_field3 == True



class MultilineClass(BaseModel):
    name: str
    story1: str
    age: int
    story2: str
    is_alive: bool

multiline_text = """
name: John
story1: Once upon a time, there was 
    a young boy named John. He was
    a brave and adventurous soul.
age: 25
story2: He went on many quests and 
    faced many challenges. But he
    always emerged victorious.
is_alive: True
"""



def test_multiline_model():
    output_parser = OutputParser(MultilineClass)
    output = output_parser.parse(multiline_text)

    assert output.name == 'John'
    assert 'Once upon a time, there was' in output.story1
    assert 'a young boy named John. He was' in output.story1
    assert 'a brave and adventurous soul.' in output.story1

    assert output.age == 25

    assert 'He went on many quests and' in output.story2
    assert 'faced many challenges. But he' in output.story2
    assert 'always emerged victorious.' in output.story2







class ComplicatedNestedClass(BaseModel):
    top_field1: str
    nested1: BasicClass
    top_field2: int
    nested2: BasicClass    
    top_field3: bool
    nested3: NestedClass
    top_field4: str
    

complicated_nested_text = """
top_field1: hello
nested1:
    basic_field1: hello
    basic_field2: 10
    basic_field3: True
top_field2: 10
nested2:
    basic_field1: hello
    basic_field2: 10
    basic_field3: True
top_field3: True
nested3:
    top_field1: hello
    top_field2: 10
    top_field3: True
    nested:
        basic_field1: hello
        basic_field2: 10
        basic_field3: True
top_field4: hello
"""

def test_nested_model2():

    output_parser = OutputParser(ComplicatedNestedClass)
    output = output_parser.parse(complicated_nested_text)

    assert output.top_field1 == 'hello'
    assert output.nested1.basic_field1 == 'hello'
    assert output.nested1.basic_field2 == 10
    assert output.nested1.basic_field3 == True

    assert output.top_field2 == 10
    assert output.nested2.basic_field1 == 'hello'
    assert output.nested2.basic_field2 == 10
    assert output.nested2.basic_field3 == True

    assert output.top_field3 == True
    assert output.nested3.top_field1 == 'hello'
    assert output.nested3.top_field2 == 10
    assert output.nested3.top_field3 == True
    assert output.nested3.nested.basic_field1 == 'hello'
    assert output.nested3.nested.basic_field2 == 10
    assert output.nested3.nested.basic_field3 == True

    assert output.top_field4 == 'hello'




class Action(BaseModel):
    pass

class AttactWithSwordAction(Action):
    """use to attack the opponent with a sword"""
    opponent: str = Field(..., description="The opponent that is being attacked")
    body_part: str = Field(..., description="The body part that is being attacked")    

class OfferQuestAction(Action):
    """use to offer a quest to the player"""
    target_person: str = Field(..., description="The person that the quest is being offered to")
    quest: str = Field(..., description="The quest that is being offered")
    reward: str = Field(..., description="The reward for completing the quest")

class TalkToPersonAction(Action):
    """use to talk to a person"""
    person: str = Field(..., description="The person that is being talked to")
    topic: str = Field(..., description="The topic of conversation")
    text: str = Field(..., description="The text of the conversation")

class OutputFormat(BaseModel):
    observation: str = Field(..., description="The observation that led to the thought")
    thought: str = Field(..., description="The thought that was generated from the observation")
    action: Union[AttactWithSwordAction, OfferQuestAction, TalkToPersonAction] = Field(..., description="The action that was taken as a result of the thought")

# Example text
union_text = """
observation: The user is seeking a quest in a friendly manner.
thought: Ahoy matey! This scallywag be lookin' fer some adventure. Time to offer a quest to this landlubber. I need to choose the right quest so  I can benifit from it. I think I have enough for this user.
action: OfferQuestAction
    target_person: user
    quest: Find the hidden treasure on Skull Island
    reward: 50 gold coins and a shiny cutlass
"""



def test_union_model():

    output_parser = OutputParser(OutputFormat)
    output = output_parser.parse(union_text)

    assert output.observation == 'The user is seeking a quest in a friendly manner.'
    assert output.thought == "Ahoy matey! This scallywag be lookin' fer some adventure. Time to offer a quest to this landlubber. I need to choose the right quest so  I can benifit from it. I think I have enough for this user."

    assert isinstance(output.action, OfferQuestAction)
    assert output.action.target_person == 'user'
    assert output.action.quest == 'Find the hidden treasure on Skull Island'
    assert output.action.reward == '50 gold coins and a shiny cutlass'