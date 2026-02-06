import json

import aiohttp
from ..clients.base import BaseLlmClient
from ..exceptions import LlmError
from ..messages import AIMessage


class PhiLlmClient(BaseLlmClient):
    # url: str = "http://localhost:3000/complete"
    # url: str = "http://skynet/text/complete"
    # url: str = "http://skynet/text/complete_chat"
    # url: str = "http://skynet1/text/complete_chat"
    url: str = "http://skynet1:31001/complete_chat"
    
    # url: str = "http://localhost:3000/complete_chat"
    # url: str = "http://localhost:8001/complete"
    # url: str = "http://localhost:8001/complete_chat"
    # url: str = "http://skynet1/text/complete"

    async def fetch(self, session, url, data=None):
        headers = {'Content-Type': 'application/json'}  # Ensure headers specify JSON
        async with session.post(url, data=json.dumps(data), headers=headers) as response:
            return await response.text(), response.status

    def preprocess_complete(self, msgs):
        prompt = ""
        for msg in msgs:
            if msg.role == "system":
                prompt += f"""
    Instruct: {msg.content}
    Output: Ok got it!
    """
            elif msg.role == "user":
                prompt += f"Instruct: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Output: {msg.content}\n"
        prompt += "Output:"
        return prompt

    def preprocess(self, msgs):
        return [m.dict() for m in msgs]


    async def complete(self, msgs, **kwargs):
        msgs = self.preprocess(msgs)
        async with aiohttp.ClientSession() as session:        
            content, status = await self.fetch(session, self.url, data={
                # "prompt": prompt,
                "messages": msgs,
                "max_new_tokens": kwargs.get("max_tokens", 200),
                "stop_sequences": kwargs.get("stop", [])        
            })
            if status != 200:
                raise LlmError(content)
            res_msg = json.loads(content)            
            return AIMessage(content=res_msg['content'])
