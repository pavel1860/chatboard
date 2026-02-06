import json
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


ContentType = Literal['text', 'image', 'pdf', 'png', 'jpeg']

class TypedContentBlock(BaseModel):
    type: ContentType
    content: str

class BaseMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str | None
    content_type: ContentType = 'text'
    content_blocks: List[Dict[str, Any]] | List[TypedContentBlock] | None = None
    name: str | None = None

    is_example: Optional[bool] = False
    is_history: Optional[bool] = False
    is_output: Optional[bool] = False
    name: Optional[str] = None
    
    def is_valid(self):
        return self.content is not None

    def to_openai(self):
        oai_msg = {
            "role": self.role, # type: ignore
            "content": self.content,            
        }
        if self.name:
            oai_msg["name"] = self.name
        return oai_msg
    
    def to_anthropic(self):
        def typed_content(content: str, content_type: ContentType):
            match content_type:
                case 'pdf':
                    return {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": content
                        }
                    }
                case 'image':
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": content
                        }
                    }
                case 'jpeg' | 'png':
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{content_type}",
                            "data": content
                        }
                    }
                case _:
                    return {
                        "type": "text",
                        "text": content
                    }

        if self.content_blocks:
            content_blocks = [
                typed_content(c.content, c.type) if isinstance(c, TypedContentBlock) else c
                for c in self.content_blocks
            ]
            return {    
                "role": "user",
                "content": content_blocks
            }
        else:
            return {
                "role": "user",
                "content": [typed_content(self.content, self.content_type)]
            }


class SystemMessage(BaseMessage):
    # role: str = Field("system", const=True)
    role: Literal["system"] = "system"


class HumanMessage(BaseMessage):
    # role: str = Field("user", const=True)
    role: Literal["user"] = "user"



class ActionCall(BaseModel):
    id: str
    name: str
    action: dict | BaseModel
    
    @property
    def type(self):
        return type(self.action)
    
    
    
class LlmUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class AIMessage(BaseMessage):
    model: str | None = None
    did_finish: Optional[bool] = True
    role: Literal["assistant"] = "assistant"
    run_id: Optional[str] = None
    action_calls: Optional[List[ActionCall]] = None
    usage: Optional[LlmUsage] = None
    
    tool_calls: Optional[List[Any]] = None
    # actions: Optional[List[BaseModel]] = []
    _iterator = -1
    _tool_responses = {}
    _tools = {}
    raw: Any = None


    def to_openai(self):
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_62136354",
                    "type": "function",
                    "function": {
                        "arguments": "{'order_id': 'order_12345'}",
                        "name": "get_delivery_date"
                    }
                }
            ]
        }
        # if self.tool_calls:            
        #     oai_msg = {
        #         "role": self.role,
        #         "content": (self.content or '') + "\n".join([f"{t.function.name}\n{t.function.arguments}" for t in self.tool_calls])
        #     }
        # else:
        #     oai_msg = {
        #         "role": self.role,
        #         "content": self.content,
        #     }
        oai_msg = {"role": self.role}
        if self.action_calls:
            tool_calls = []
            for action_call in self.action_calls:
                tool_calls.append({
                  "id": action_call.id,
                    "type": "function",
                    "function": {
                        # "arguments": json.dumps(action_call.action.model_dump()),
                        "arguments": action_call.action.model_dump_json() if isinstance(action_call.action, BaseModel) else json.dumps(action_call.action),
                        "name": action_call.name
                    }                      
                })
            oai_msg["tool_calls"] = tool_calls
        else:
            oai_msg["content"] = self.content
        if self.name:
            oai_msg["name"] = self.name
        return oai_msg
    
    def to_anthropic(self):
        if self.action_calls:
            content_blocks = []
            if self.content:
                content_blocks.append({
                        "type": "text",
                        "text": self.content
                    })
            for action_call in self.action_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": action_call.id,
                    "name": action_call.name,
                    "input": json.loads(action_call.action.model_dump_json()) if isinstance(action_call.action, BaseModel) else action_call.action
                    # "input": action_call.action.model_dump()
                })
            return {
                "role": self.role,
                "content": content_blocks
            }
        else:
            return {
                "role": self.role,
                "content": self.content
            }
            
    def to_langsmith(self):
        msg = {
            "choices": [
                {
                    "message": {
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": [{
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.action.model_dump_json()
                            }
                        } for tool_call in self.action_calls]
                    }
                }
            ],            
        }
        if self.usage:
            msg["usage"] = {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            }
        return msg
                                                       
    @property
    def actions(self):
        return list(self._tools.values())
    
    @property
    def output(self):
        if not self.action_calls:
            return None
        return self.action_calls[0].action
    
    @output.setter
    def output(self, value):
        self.actions = [value]
    
    def is_valid(self):
        if self.content is not None:
            return True 
        elif self._tool_responses:
            return True
        return False
    
    def _add_tool_response(self, response: "ActionMessage"):
        self._tool_responses[response.id] = response
    
    def add_action_output(self, tool_id: str, output: BaseModel | str | dict):
        self._tool_responses[tool_id] = output
        
    def add_action(self, tool_id: str, action: BaseModel):
        self._tools[tool_id] = action

    # def to_openai(self):
    #     oai_msg = {"role": self.role}                
    #     if self.content:
    #         oai_msg["content"] = self.content
    #     # if self.tool_calls:
    #     #     responded_tools = [t for t in self.tool_calls if t.id in self._tool_responses]   
    #     #     if responded_tools:
    #     #         oai_msg['tool_calls'] = responded_tools
    #     if self._tool_responses:
    #         oai_msg['tool_calls'] = [r.tool_call for r in self._tool_responses.values()]
    #     return oai_msg
    
    # def __iter__(self):
    #     self._iterator = -1
    #     return self
    
    # def __next__(self):
    #     if self._iterator == -1:
    #         self._iterator += 1
    #         if self.content:                
    #             return "response", self.content            
    #     if self._iterator >= len(self.actions):
    #         raise StopIteration
    #     return self.tool_calls[self._iterator].id, self.actions[self._iterator]
    
    
  
class ActionMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    
    # @property
    # def id(self):
    #     return self.tool_call.id
    
    def to_openai(self):
        oai_msg = {
            "tool_call_id": self.id,
            "role": "tool",
            "content": self.content
        }
        if self.name:
            oai_msg["name"] = self.name
        return oai_msg
        
    def to_anthropic(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.id,
                    "content": self.content
                }
            ]
        }



ChatMessageType = Union[SystemMessage, HumanMessage, AIMessage]


def validate_msgs(msgs: List[BaseMessage]) -> List[BaseMessage]:
    ai_messages = {}

    validated_msgs = []
    for msg in msgs:
        if isinstance(msg, AIMessage):
            if msg.action_calls:
                ai_messages[msg.action_calls[0].id] = msg
            else:
                validated_msgs.append(msg)
        elif isinstance(msg, ActionMessage):
            ai_msg = ai_messages.get(msg.id, None)
            if not ai_msg:
                continue
            validated_msgs += [ai_msg, msg]
            # validated_msgs.append((ai_msg, msg))
        else:
            validated_msgs.append(msg)
    return validated_msgs




def remove_action_calls(messages):
    validate_msgs = []
    found_action_calls = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.action_calls:
            msg.action_calls = [action_call for action_call in msg.action_calls if action_call.id in found_action_calls]
            # validate_msgs.append(msg)
            validate_msgs.insert(0, msg)
        elif isinstance(msg, ActionMessage):
            found_action_calls.add(msg.id)
            # validate_msgs.append(msg)
            validate_msgs.insert(0, msg)
        else:
            # validate_msgs.append(msg)
            validate_msgs.insert(0, msg)
    return validate_msgs
    # return reversed(validate_msgs)


def remove_actions(messages):
    validate_msgs = []
    found_action_calls = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.action_calls:
            validate_msgs.append(msg)
            for action_call in msg.action_calls:
                found_action_calls.add(action_call.id)
        elif isinstance(msg, ActionMessage):
            if msg.id in found_action_calls:
                validate_msgs.append(msg)
        else:
            validate_msgs.append(msg)
    return validate_msgs



def filter_message_alternation(messages: List[BaseMessage]) -> List[BaseMessage]:
    validated_messages = []
    prev_role = None
    for msg in messages:
        if msg.role == prev_role:
            validated_messages.pop(-1)        
        prev_role = msg.role
        validated_messages.append(msg)
    return validated_messages


def merge_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    validated_messages = []
    prev_role = None
    for msg in messages:
        if msg.role == prev_role:
            validated_messages[-1].content += "\n" + msg.content
        else:
            validated_messages.append(msg)
        prev_role = msg.role
    return validated_messages
        
    

def validate_first_message(messages: List[BaseMessage]) -> List[BaseMessage]:
    for i in range(len(messages)):
        if messages[i].role == "user":
            if i != 0:
                print("first message must be a user message. fixing...")
            return messages[i:]
    else:
        raise ValueError("No user message found in messages. first message must be a user message")

    
def filter_action_calls(messages: List[BaseMessage], user_first: bool=False, check_alternation=False, should_merge=False) -> List[BaseMessage]:
    messages = [m.model_copy() for m in messages if (m.content or m.content_blocks)]
    if user_first:
        messages = validate_first_message(messages)
    if should_merge:
        messages = merge_messages(messages)
    if check_alternation:
        messages = filter_message_alternation(messages)    
    messages = remove_actions(remove_action_calls(messages))    
    return messages
    