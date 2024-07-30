
from typing import List
from chatboard.text.llms.conversation import BaseMessage
import asyncio







class History:
    
    def __init__(self, add_actions=False) -> None:
        self.history: List[BaseMessage] = []
        self.add_actions = add_actions
        
                
    def add(self, context, message: BaseMessage, run_id: str, prompt: str):
        self.history.append(message)
        asyncio.create_task(self.save(context, message, run_id, prompt))
        
    async def init(self, context):
        messages = await self.load(context)
        for message in messages:
            if not isinstance(message, BaseMessage):
                raise Exception(f"message {message} is not an instance of BaseMessage")
        self.history = messages
                
    # def add_many(self, messages: List[BaseMessage]):
        # self.history.extend(messages)
        
    def get(self, from_idx=1, to_idx=None):
        if to_idx is not None:
            return self.history[-from_idx: -to_idx]
        return self.history[-from_idx:]
    
    def last(self):
        return self.history[-1]
    

    async def load(self, context):
        return self.history
    
    async def save(self, context, message: BaseMessage, run_id: str):
        return 