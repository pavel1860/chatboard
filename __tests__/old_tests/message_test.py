from pydantic import BaseModel, Field
import pytest
from chatboard.llms.messages import AIMessage, HumanMessage, ActionMessage, ActionCall, filter_action_calls, remove_action_calls, remove_actions, validate_msgs

class TestAction(BaseModel):
    test: str = Field(..., title="Test", description="The test value")


ai_message_calls = AIMessage(
        id="1",
        content="Hello there", 
        action_calls=[
            ActionCall(id="11", name="test_action", action=TestAction(test="test")),
            ActionCall(id="12", name="test_action", action=TestAction(test="test"))
        ]
    )

action_message1 = ActionMessage(id="11", content="response to action")
action_message2 = ActionMessage(id="12", content="response to action")
human_message = HumanMessage(id="2", content="Hello there")
ai_message = AIMessage(id="1", content="Hello there")





def test_message_validation():
    filtered_messages = filter_action_calls([
        ai_message_calls,
        action_message1,
        action_message2,
        human_message,
        ai_message,
        human_message
    ])

    assert filtered_messages == [ai_message_calls, action_message1, action_message2, human_message, ai_message, human_message]

    filtered_messages = filter_action_calls([
        ai_message_calls,
        action_message1,
        # action_message2,
        human_message,
        ai_message,
        human_message
    ])

    assert filtered_messages == [ human_message, ai_message, human_message]


    filtered_messages = filter_action_calls([
        ai_message_calls,
        # action_message1,
        action_message2,
        human_message,
        ai_message,
        human_message
    ])

    assert filtered_messages == [ human_message, ai_message, human_message]

    filtered_messages = filter_action_calls([
        # ai_message_calls,
        action_message1,
        action_message2,
        human_message,
        ai_message,
        human_message
    ])

    assert filtered_messages == [ human_message, ai_message, human_message]
    
    
    
    
def test_mixed_message_validation():
    class TestAction(BaseModel):
        test: str = Field(..., title="Test", description="The test value")


    ai_message_calls = AIMessage(
            id="1",
            content="Hello there", 
            action_calls=[
                ActionCall(id="11", name="test_action", action=TestAction(test="test")),
                ActionCall(id="13", name="test_action", action=TestAction(test="test"))
            ]
        )

    action_message1 = ActionMessage(id="11", content="response to action")
    action_message2 = ActionMessage(id="12", content="response to action")
    human_message = HumanMessage(id="2", content="Hello there")
    ai_message = AIMessage(id="1", content="Hello there")


    filtered_messages = remove_actions(remove_action_calls([
        ai_message_calls,
        action_message1,
        action_message2,
        human_message,
        ai_message,
        human_message
    ]))

    assert filtered_messages == [ai_message_calls, action_message1, human_message, ai_message, human_message]
    assert filtered_messages[0].action_calls == [ActionCall(id="11", name="test_action", action=TestAction(test="test"))]