from typing import AsyncGenerator, List
from chatboard.block.util import StreamEvent
from chatboard.prompt import component, Depends
from chatboard.block import Block
from chatboard.llms import LLMRegistry
from chatboard.context.execution_context import ExecutionContext
import datetime as dt
from uuid import UUID, uuid4
from chatboard.model import ModelField, KeyField, RelationField
from chatboard.versioning import BlockLog, ArtifactModel
from chatboard import AuthModel

from pydantic import BaseModel, Field







class UserState(ArtifactModel):
    # id: int = KeyField(primary_key=True)
    id: UUID = ModelField(default_factory=uuid4)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    health: int = ModelField(default=100)
    gold: int = ModelField(default=0)
    strength: int = ModelField(default=10)
    agility: int = ModelField(default=10)
    intelligence: int = ModelField(default=10)
    charisma: int = ModelField(default=10)
    luck: int = ModelField(default=10)
    user_id: UUID = ModelField(foreign_key=True)




class User(AuthModel):
    name: str | None = ModelField(None)
    state: UserState | None = RelationField(None, foreign_key="user_id")
    
    
    


class AttackAction(BaseModel):
    """use this action when you want to attack the user"""
    weapon: str = Field(description="the weapon you will use to attack the user, can be a sword or a gun")
    target: str = Field(description="the body part you will attack, can be the head or the chest, or the legs")

class GiveQuestAction(BaseModel):
    """use this action when you want to give the user a quest"""
    reward: int = Field(description="the reward in gold coins")
    description: str = Field(description="the description of the quest")




     
# @component(tags=["pirate"])
# async def pirate_talk(message: Block, llm: LLM = Depends(LLM)):
#     with Block(role="system") as sys:
#         sys /= "you are a pirate by the name of Jack Black"
#         sys /= "you speak like a pirate"
        
#         with sys(tags=["response_schema"]) as schema:
#             schema /= "use the following xml format to output the response"
#             with schema.view("output", str, tags=["pirate"]) as res:
#                 with res.view("thought", str, tags=["pirate"]) as t:
#                     t /= "think step by step what to do with the information you observed and thinkg if there is a tool that can help you to complete the tasks."
#                 with res.view("answer", str, tags=["pirate"]) as r:
#                     r /= "the answer to the user's question goes here"
#                 schema /= "if you want to use a tool, you can pick one, or multiple of the actions from the following list."
#                 with res.view_list("tool", key="name", tags=["pirate"]) as tools:
#                     tools.register(AttackAction)
#                     tools.register(GiveQuestAction)
                    
#             with schema("Output Rules", tags=["output_rules"]) as r:
#                 r /= "make sure you follow the schema strictly"
#                 r /= "make sure empty xml elements are ending with '/>'"
                
#     with Block(role="user", tags=["user_message"]) as user_message:
#         user_message /= message
#     # yield llm.stream(sys, user_message).name("pirate_talk_llm").save("research/data/pirate_talk.jsonl").parse(schema)
#     yield llm.stream(sys, user_message).name("pirate_talk_llm").load("research/data/pirate_talk.jsonl", delay=0.01).parse(schema)

@component(tags=["pirate"])
async def pirate_talk(llm: LLMRegistry = Depends(LLMRegistry)):
    with Block(role="system") as sys:
        sys /= "you are a pirate by the name of Jack Black"
        sys /= "you speak like a pirate"
        
        with sys(tags=["response_schema"]) as schema:
            schema /= "use the following xml format to output the response"
            with schema.view("output", str, tags=["pirate"]) as res:
                with res.view("thought", str, tags=["pirate"]) as t:
                    t /= "think step by step what to do with the information you observed and thinkg if there is a tool that can help you to complete the tasks."
                with res.view("answer", str, tags=["pirate"]) as r:
                    r /= "the answer to the user's question goes here"
                schema /= "if you want to use a tool, you can pick one, or multiple of the actions from the following list."
                with res.view_list("tool", key="name", tags=["pirate"]) as tools:
                    tools.register(AttackAction)
                    tools.register(GiveQuestAction)
                    
            with schema("Output Rules", tags=["output_rules"]) as r:
                r /= "make sure you follow the schema strictly"
                r /= "make sure empty xml elements are ending with '/>'"
                
    yield llm.stream(sys).name("pirate_talk_llm").load("research/data/pirate_talk.jsonl", delay=0.01).parse(schema)

   

@component(tags=["pirate_agent"])  
async def pirate_agent(message: Block):
    user = User.current()
    user_state = await UserState.query().last()
    if user_state is None:
        user_state = UserState(health=100, gold=0)
        await user_state.save()
    
    thought = yield pirate_talk(message=message)
    for action in thought.get("actions"):
        if action.tag == "attack":
            # user_state.health -= 10
            # user_state = yield await user_state.save()
            yield Block(
                f"attacked the user with {action.attrs.get('weapon')} and hit {action.attrs.get('target')}", 
                role="assistant"
            )
        else:
            # user_state.gold += action.attrs.get("reward")
            # user_state = yield await user_state.save()
            yield Block(
                f"gave the user a quest with reward {action.attrs.get('reward')}", 
                role="assistant"
            )
 
