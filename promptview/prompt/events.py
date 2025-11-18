


from dataclasses import dataclass, field
import json
from typing import Any, Literal, TypedDict, Unpack
from pydantic import BaseModel, Field
import uuid
import datetime as dt

from promptview.model.db_types import Tree



LlmStreamType = Literal["stream_start", "message_delta", "stream_end", "stream_error"]

SpanEventType = Literal["span_start", "span_event", "span_end"]

MessageType = Literal["user_message", "assistant_message"]

EventType = LlmStreamType | MessageType | SpanEventType | Any



class EventParams(TypedDict, total=False):
    created_at: dt.datetime | None
    error: str | None
    index: int | None
    request_id: str | None




@dataclass
class StreamEvent:
    type: str
    payload: Any
    index: int
    path: str
    request_id: str | None = None
    created_at: dt.datetime = field(default_factory=dt.datetime.now)        
    name: str | None = None
    error: dict | None = None
    
    
    
    
    # attrs: dict | None = None
    # depth: int = 0
    # parent_value_id: int | None = None
    # value: Any | None = None
    # payload: Any | None = None    
    # created_at: dt.datetime | None = None
    # error: str | None = None
    # index: int | None = None
    # request_id: str | None = None
    # span_id: str | None = None
    # path: list[int] | None = None
        
    def payload_to_dict(self):
        if self.payload is None:
            return None
        if hasattr(self.payload, "model_dump"):
            return self.payload.model_dump()
        elif hasattr(self.payload, "to_dict"):
            return self.payload.to_dict()
        else:
            return self.payload
    
    # def attrs_to_dict(self):
    #     if self.attrs is None:
    #         return None
    #     if hasattr(self.attrs, "model_dump"):
    #         return self.attrs.model_dump()
    #     elif hasattr(self.attrs, "to_dict"):
    #         return self.attrs.to_dict()
    #     else:
    #         return self.attrs
        
    @staticmethod
    def _json_default(obj):
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        if isinstance(obj, dt.date):
            return obj.isoformat()
        if isinstance(obj, type):
            return str(obj.__name__)
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)  # Last-resort fallback

        # Add more types if needed
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        
    def to_json(self):
        payload = self.payload_to_dict()   
        # attrs = self.attrs_to_dict()
        dump = {
            "type": self.type,
            # "path": self.path,
            "name": self.name,
            "index": self.index,
            "path": self.path,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            # "attrs": attrs,
            # "depth": self.depth,
            "payload": payload,            
            "request_id": self.request_id,
            "error": self.error,
            # "span_id": self.span_id,   
            # "parent_event_id": self.parent_value_id,
            # "event": self.value.model_dump(exclude={"branch_id", "turn_id", "branch", "turn"}) if self.value else None,
            # "value": self.value.to_dict() if self.value else None,
        }        
        if self.error:
            dump["error"] = self.error
            
        if self.created_at:
            dump["created_at"] = self.created_at
            
        return json.dumps(dump, default=self._json_default)
    
    def to_ndjson(self):
        return self.to_json() + "\n"
    
    def __repr__(self):
        return f"Event(type={self.type}, path={self.path}, name={self.name}, payload={self.payload}, index={self.index}, request_id={self.request_id})"
    # def __repr__(self):
    #     return f"Event(type={self.type}, name={self.name}, attrs={self.attrs}, depth={self.depth}, payload={self.payload}, index={self.index}, request_id={self.request_id}, span_id={self.span_id}, path={self.path})"
    
    

    
from pydantic import BaseModel, Field
from typing import Any, Literal, Protocol, Union, Optional, List, Dict
import datetime as dt



class Serializeble(Protocol):
    
    def model_dump(self) -> dict:
        ...
    


# -- Base event --
class BaseEvent(BaseModel):
    turn_id: int
    timestamp: Optional[dt.datetime] = None


# -- Event: Stream Start --
class StreamStart(BaseEvent):
    type: Literal["stream_start"] = "stream_start"

class MessageDelta(BaseEvent):
    type: Literal["message_delta"] = "message_delta"
    payload: Any

class StreamEnd(BaseEvent):
    type: Literal["stream_end"] = "stream_end"
    response: Any
    done: Literal[True] = True

# -- Tool Call Info --
class ToolCallObject(BaseModel):
    id: str
    name: str
    args: Dict
    status: Literal["started", "completed", "errored"]
    output: Optional[Dict] = None
    error: Optional[str] = None


# -- Event: Tool Call --
class ToolCallEvent(BaseEvent):
    type: Literal["tool_call"]
    tool_calls: List[ToolCallObject]


# -- Event: State Update --
class StateUpdate(BaseEvent):
    type: Literal["state_update"]
    state: Dict[str, Dict]


# -- Event: Agent Final Output --
class AgentResponse(BaseEvent):
    type: Literal["agent_response"]
    message: Dict


# -- Event: Developer Log Line --
class LogEvent(BaseEvent):
    type: Literal["log"]
    content: str


# -- Event: Trace Info (execution stack, context diffs) --
class TraceEvent(BaseEvent):
    type: Literal["trace"]
    state: Dict


# -- Event: Error Info --
class ErrorEvent(BaseEvent):
    type: Literal["error"]
    content: str


# -- Union: All Stream Events --
# StreamEvent = Union[
#     StreamStart,
#     MessageDelta,
#     StreamEnd,
#     ToolCallEvent,
#     StateUpdate,
#     AgentResponse,
#     LogEvent,
#     TraceEvent,
#     ErrorEvent,
# ]
