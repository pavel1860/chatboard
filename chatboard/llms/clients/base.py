from datetime import datetime
from typing import Any, List
from pydantic import BaseModel
import pickle


class BaseLlmClient(BaseModel):
    client: Any
    
    async def complete(self, msgs, **kwargs):
        raise NotImplementedError
    
    
    
    def serialize_messages(self,  run_id: str, messages: List[BaseModel], response: BaseModel | None = None):
        date_str = datetime.now().strftime("%d_%H-%M")
        obj = {
            "run_id": run_id,
            "messages": messages,
            "response": response
        }
        with open(f"tmp/messages_{date_str}_{run_id}.pkl", "wb") as f:
            pickle.dump(obj, f)
    