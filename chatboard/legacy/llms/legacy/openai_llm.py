



from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from chatboard.llms.clients.openai_client import OpenAiLlmClient
from chatboard.llms.legacy.llm2 import LLM
from chatboard.llms.messages import AIMessage, ActionMessage, BaseMessage, HumanMessage, SystemMessage
from chatboard.llms.utils.action_manager import Actions
from chatboard.prompt.legacy.mvc import ViewBlock
from chatboard.templates.action_template import system_action_view


class OpenAiLLM(LLM):
    name: str = "OpenAiLLM"    
    client: OpenAiLlmClient = Field(default_factory=OpenAiLlmClient)
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    
    
    
    def transform(self, root_block: ViewBlock, actions: Actions | List[BaseModel] | None = None, **kwargs) -> Tuple[List[BaseMessage], Actions]:
        messages = []
        if not isinstance(actions, Actions):
            actions = Actions(actions=actions)
        actions.extend(root_block.find_actions())
        system_block = root_block.first(role="system", depth=1)
        if system_block and actions:
            system_block.push(system_action_view(actions))
        for block in root_block.find(depth=1): 
            content = self.render_block(block, **kwargs)
            if isinstance(content, list):
                content, content_blocks = None, content
            else:
                content, content_blocks = content, None

            if block.role == 'user':
                messages.append(HumanMessage(id=block.uuid, content=content, content_blocks=content_blocks, content_type=block.content_type))
            elif block.role == 'assistant':
                messages.append(AIMessage(id=block.uuid, content=content, content_blocks=content_blocks, action_calls=block.action_calls))
            elif block.role == 'system':
                messages.append(SystemMessage(id=block.uuid, content=content, content_blocks=content_blocks))
            elif block.role == 'tool':
                messages.append(ActionMessage(id=block.uuid, content=content, content_blocks=content_blocks))
            else:
                raise ValueError(f"Unsupported role: {block.role}")
        
        return messages, actions