import pytest
from chatboard.text.llms.history import History, HistoryMessage
from chatboard.text.llms.conversation import HumanMessage, AIMessage, SystemMessage
from chatboard.text.llms.mvc import BaseModel



def test_history():
    history = History()

    class TestAction(BaseModel):
        name: str

    history.add(
        message=HumanMessage(
            content="Hello"
        ),
        prompt_name="greeting",
        run_id="123",
    )

    history.add(
        message=AIMessage(
            content="Hello, there"
        ),
        prompt_name="greeting",
        run_id="123",
    )


    history.add(
        message=HumanMessage(
            content="How are you?"
        ),
        prompt_name="greeting",
        run_id="432",
    )

    history.add(
        message=TestAction(name="greeting"),
        prompt_name="greeting",
        run_id="432",
    )



    history.add_many(
        messages=[
            SystemMessage(
                content="this is a system message"
            ),
            HumanMessage(
                content="this is an example",
                is_example=True
            ),
            AIMessage(
                content="this is an example response",
                is_example=True
            ),
            HumanMessage(
                content="Hello",
                is_history=True
            ),
            AIMessage(
                content="Hello, there",
                is_history=True
            ),        
            HumanMessage(
                content="I have a wedding tomorrow"
            ),
            AIMessage(
                content="I can help you with that"
            ),        
        ],
        prompt_name="event",
        run_id="8645"
    )


    history.add_many(
        messages=[
            SystemMessage(
                content="this is a system message"
            ),
            HumanMessage(
                content="this is an example",
                is_example=True
            ),
            AIMessage(
                content="this is an example response",
                is_example=True
            ),
            HumanMessage(
                content="Hello",
                is_history=True
            ),
            AIMessage(
                content="Hello, there",
                is_history=True
            ),        
            HumanMessage(
                content="what should I eat there?"
            ),
            AIMessage(
                content="you should eat a salad"
            ),        
        ],
        prompt_name="event",
        run_id="23455"
    )


    history.add(
        message=HumanMessage(
            content="I want to gain weight"
        ),
        prompt_name="plan",
        run_id="56234"
    )


    msgs = history.get_messages(1)
    assert msgs[0].content == "I want to gain weight"

    msgs = history.get_messages(4, "greeting", add_actions=True)
    assert len(msgs) == 4
    assert "TestAction" in msgs[-1].content
    msgs = history.get_messages(4, "greeting", add_actions=False)
    assert len(msgs) == 3
    assert "How are you?" in msgs[-1].content
    msgs = history.get_messages(10, "event")
    assert len(msgs) == 4
