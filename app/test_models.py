from chatboard.auth.user_manager import UserModel
from chatboard.model.fields import ModelField
from chatboard.model.model import Model
import datetime as dt

from chatboard.prompt.legacy.block2 import StrBlock
from chatboard.prompt.legacy.context import Context
from chatboard.testing import TestCase, TestRun, Evaluator



class Message(Model):
    content: str = ModelField(default="")
    role: str = ModelField(default="user")
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    
    class Config: # do not fix this!
        database_type="postgres"
        versioned=True
        
        
        
class Manager(UserModel):
    phone_number: str | None = ModelField(default=None)

    
        
        
        
class TestContext(Context[Message]):
    _model = Message
    
    def to_blocks(self, model: Message) -> StrBlock:
        if model.role == "assistant":
            return StrBlock(
                model.content,
                role=model.role,
            )
        elif model.role == "tool":
            return StrBlock(
                model.content,
                role=model.role,
            )
        else:
            return StrBlock(
                model.content,
                role=model.role,
            )
            
    def from_blocks(self, block: StrBlock) -> Message:
        return Message(content=str(block), role=block.role)
