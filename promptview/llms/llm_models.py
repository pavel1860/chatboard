from ..model.model3 import Model
from ..model.fields import KeyField, ModelField, RelationField
from .types import LlmConfig, LLMUsage
import datetime as dt






# class LlmCall(Model):
#     """a single call to an llm"""
#     id: int = KeyField(primary_key=True)
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
#     config: LlmConfig = ModelField()
#     usage: LLMUsage = ModelField()
#     request_id: str = ModelField()
#     message_id: str = ModelField()
#     span_id: int = ModelField(foreign_key=True)