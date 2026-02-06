import os
os.environ["POSTGRES_URL"] = "postgresql://snack:Aa123456@localhost:5432/chatboard_test"
import pytest
import asyncio
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import asyncpg
from chatboard.artifact_log.artifact_log3 import ArtifactLog, TurnStatus
from chatboard.model.fields import ModelField, RelationField
from chatboard.model.model import Model



class Message(Model):
    content: str = ModelField(default="")
    role: str = ModelField(default="user")
    
    class Config: # do not fix this!
        database_type="postgres"
        versioned=True
        
        
@pytest_asyncio.fixture()
async def artifact_log():
    art_log = ArtifactLog()
    await art_log.initialize_tables()
    yield art_log
    await art_log.drop_tables(["messages"])



@pytest.mark.asyncio
async def test_artifact_log(artifact_log: ArtifactLog):
    async with artifact_log as art_log:
        assert art_log.head["turn_id"] is not None
        assert art_log.head["branch_id"] is not None    
        turn_list = await art_log.get_turn_list()
        assert len(turn_list) == 1
        print([turn.id for turn in turn_list])
        turn_id1 = art_log.head["turn_id"]
        await Message(
            content="Hello, how are you?",
            role="user"
        ).save()
        # message_list = await art_log.get_artifact_list(artifact_table="message")
        message_list = await Message.limit(20)
        assert len(message_list) == 1
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"
        
        turn1 = await art_log.get_turn(turn_id1)
        
        assert turn1.index == 1
        assert turn1.status == TurnStatus.STAGED
        
        await Message(
            content="I am fine, thank you!",
            role="assistant"
        ).save()
        
        message_list = await Message.limit(20)  
        assert len(message_list) == 2
        assert message_list[-1].content == "I am fine, thank you!"
        assert message_list[-1].role == "assistant"
        
        turn_id2 = await art_log.commit_turn()
        
        assert art_log.head["turn_id"] == turn_id2    
        
        turn_list = await art_log.get_turn_list()
        assert len(turn_list) == 2
        assert turn_list[0].id == turn_id2
        assert turn_list[0].status == TurnStatus.STAGED
        assert turn_list[1].id == turn_id1
        assert turn_list[1].status == TurnStatus.COMMITTED
        
        message_list = await Message.limit(20)
        assert len(message_list) == 2
        
        await Message(
            content="what is the weather in tokyo?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in tokyo is sunny",
            role="assistant"
        ).save()
        message_list = await Message.limit(20)
        assert len(message_list) == 4




@pytest.mark.asyncio
async def test_artifact_log_branching(artifact_log: ArtifactLog):
    async with artifact_log as art_log:        
        turn0 = art_log.head["turn_id"]
        await Message(
            content="Hello, how are you?",
            role="user"
        ).save()
        
        await Message(
            content="I am fine, thank you!",
            role="assistant"
        ).save()
        
        turn1 = await art_log.commit_turn()
        
        await Message(
            content="what is the weather in tokyo?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in tokyo is sunny",
            role="assistant"
        ).save()
        
        turn2 = await art_log.commit_turn()
        
        
        await Message(
            content="Wow thats great. book me a flight to tokyo",
            role="user"
        ).save()
        
        await Message(
            content="I will book you a flight to tokyo",
            role="assistant"
        ).save()
        
        await Message(
            content="there are some great deals on flights to tokyo",
            role="assistant"
        ).save()
        
        turn3 = await art_log.commit_turn()
        
        await Message(
            content="buy me the cheapest flight to tokyo",
            role="user"
        ).save()
        
        await Message(
            content="booked you a flight to tokyo for $1000",
            role="assistant"
        ).save()
        
        turn4 = await art_log.commit_turn()
        
        
        turn_list = await art_log.get_turn_list()
        assert len(turn_list) == 5
        assert turn_list[4].id == turn0
        assert turn_list[3].id == turn1
        assert turn_list[2].id == turn2
        assert turn_list[1].id == turn3
        assert turn_list[0].id == turn4
        
        
        message_list = await Message.limit(20)
        assert len(message_list) == 9
        assert message_list[-1].content == "booked you a flight to tokyo for $1000"
        assert message_list[-1].role == "assistant"
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"
        
        branch2 = await art_log.branch_from(turn_id=turn0, check_out=True)
        assert art_log.head["branch_id"] == branch2
        
        turn20 = art_log.head["turn_id"]
        
        turn_list = await art_log.get_turn_list()
        assert len(turn_list) == 2
        

        await Message(
            content="what is the weather in Rome?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in Rome is rainy",
            role="assistant"
        ).save()
        
        turn21 = await art_log.commit_turn()
        
        await Message(
            content="thats too bad. I don't like the weather in Rome",
            role="user"
        ).save()
        
        await Message(
            content="would you like to look for some other flights?",
            role="assistant"
        ).save()
        
        turn22 = await art_log.commit_turn()
        
        await Message(
            content="what is the weather in Bangkok?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in Bangkok is very hot",
            role="assistant"
        ).save()

        turn_list = await art_log.get_turn_list()
        assert len(turn_list) == 4
        assert turn_list[0].id == turn22
        assert turn_list[1].id == turn21
        assert turn_list[2].id == turn20
        assert turn_list[3].id == turn0
        
        
        
        message_list = await Message.limit(20)
        assert len(message_list) == 8
        assert message_list[-1].content == "the weather in Bangkok is very hot"
        assert message_list[-1].role == "assistant"
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"
        
        

@pytest.mark.asyncio
async def test_artifact_log_reverting(artifact_log: ArtifactLog):
    async with artifact_log as art_log:
        turn0 = art_log.head["turn_id"]
        branch1 = art_log.head["branch_id"]
        await Message(
            content="Hello, how are you?",
            role="user"
        ).save()
        
        await Message(
            content="I am fine, thank you!",
            role="assistant"
        ).save()
        
        turn1 = await art_log.commit_turn()
        
        await Message(
            content="what is the weather in tokyo?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in tokyo is sunny",
            role="assistant"
        ).save()
        
        turn2 = await art_log.commit_turn()
        
        
        await Message(
            content="Wow thats great. book me a flight to tokyo",
            role="user"
        ).save()
        
        await Message(
            content="I will book you a flight to tokyo",
            role="assistant"
        ).save()
        
        await Message(
            content="there are some great deals on flights to tokyo",
            role="assistant"
        ).save()
        
        turn3 = await art_log.commit_turn()
        
        await Message(
            content="buy me the cheapest flight to tokyo",
            role="user"
        ).save()
        
        await Message(
            content="booked you a flight to tokyo for $1000",
            role="assistant"
        ).save()
        
        turn4 = await art_log.commit_turn()
            
        
        branch2 = await art_log.branch_from(turn_id=turn0, check_out=True)
        turn20 = art_log.head["turn_id"]

        await Message(
            content="what is the weather in Rome?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in Rome is rainy",
            role="assistant"
        ).save()
        
        turn21 = await art_log.commit_turn()
        
        await Message(
            content="thats too bad. I don't like the weather in Rome",
            role="user"
        ).save()
        
        await Message(
            content="would you like to look for some other flights?",
            role="assistant"
        ).save()
        
        turn22 = await art_log.commit_turn()
        
        await Message(
            content="what is the weather in Bangkok?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in Bangkok is very hot",
            role="assistant"
        ).save()
        
        turn23 = await art_log.commit_turn()
        
        await Message(
            content="ok book me a flight to Bangkok",
            role="user"
        ).save()
        
        
        await Message(
            content="looking for some great deals on flights to Bangkok",
            role="assistant"
        ).save()
        
        await Message(
            content="booked you a flight to Bangkok",
            role="assistant"
        ).save()
        
        turn24 = await art_log.commit_turn()

        branch3 = await art_log.branch_from(turn_id=turn21, check_out=True)
        turn30 = art_log.head["turn_id"]
        
        await Message(
            content="what is the weather in new york?",
            role="user"
        ).save()
        
        await Message(
            content="the weather in new york is snowy",
            role="assistant"
        ).save()
        
        turn31 = await art_log.commit_turn()
        
        await Message(
            content="looking for some great deals on flights to new york",
            role="user"
        ).save()
        
        await Message(
            content="booked you a flight to new york",
            role="assistant"
        ).save()
        
        turn32 = await art_log.commit_turn()
        
        await Message(
            content="what is the best restaurant in new york?",
            role="user"
        ).save()
        
        await Message(
            content="the best restaurant in new york is the one on 5th avenue",
            role="assistant"
        ).save()
        
        turn33 = await art_log.commit_turn()
        
        branche_list = await art_log.get_branch_list()
        assert len(branche_list) == 3
        assert branche_list[0].id == branch3
        assert branche_list[1].id == branch2
        assert branche_list[2].id == branch1
        
        
        turn_list = await art_log.get_turn_list()
        
        assert len(turn_list) == 7
        assert turn_list[0].id == turn33
        assert turn_list[-1].id == turn0
        message_list = await Message.limit(20)
        assert len(message_list) == 12
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"
        assert message_list[-1].content == "the best restaurant in new york is the one on 5th avenue"
        assert message_list[-1].role == "assistant"
        
        
        await art_log.checkout_branch(branch_id=branch1)
        assert art_log.head["branch_id"] == branch1
        assert art_log.head["turn_id"] == turn4
        
        message_list = await Message.limit(20)
        assert len(message_list) == 9
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"        
        assert message_list[-1].content == "booked you a flight to tokyo for $1000"
        assert message_list[-1].role == "assistant"
        
        await art_log.checkout_branch(branch_id=branch2)
        assert art_log.head["branch_id"] == branch2
        assert art_log.head["turn_id"] == turn24
        
        message_list = await Message.limit(20)
        assert len(message_list) == 11
        assert message_list[0].content == "Hello, how are you?"
        assert message_list[0].role == "user"
        assert message_list[-1].content == "booked you a flight to Bangkok"
        assert message_list[-1].role == "assistant"