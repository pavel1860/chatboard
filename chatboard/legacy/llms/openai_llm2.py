from abc import abstractmethod
import json
from typing import Any, AsyncGenerator, List, Type

from ..block.block7 import BlockList
from ..block.util import LLMEvent, StreamEvent
from ..tracer.langsmith_tracer import Tracer
from ..utils.model_utils import schema_to_function
from .llm import LLM, LLMToolNotFound, LlmConfig, LlmContext
from pydantic import Field, BaseModel
from .utils.action_manager import Actions
# from ..prompt.block import BaseBlock, ResponseBlock
from ..block import BlockChunk, ToolCall
import openai
import os
from openai.types.chat import ChatCompletionMessageParam





class OpenAiLLM(LlmContext):
    models = ["gpt-4o", "gpt-4o-mini"]
    client: openai.AsyncClient
    
    def __init__(self, model: str):
        super().__init__(model)
        self.client = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))
    

    def to_tool(self, tool: Type[BaseModel]) -> dict:
        schema = schema_to_function(tool)
        return schema
    
    def from_tool(self, tools: List[Type[BaseModel]], tool_call: openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall) -> BaseModel:
        tool_args = json.loads(tool_call.function.arguments)
        tool_cls = next((t for t in tools if t.__name__ == tool_call.function.name), None)
        if not tool_cls:
            raise LLMToolNotFound(tool_call.function.name)
        tool = tool_cls(**tool_args)
        return tool
    
    def to_message(self, content: str, role: str, tool_calls: List[ToolCall] | None = None, tool_call_id: str | None = None, name: str | None = None) -> ChatCompletionMessageParam:
        # message: ChatCompletionMessageParam = {
        message = {
            "role": role or "user",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                        "type": "function",
                        "function": {
                            # "arguments": json.dumps(action_call.action.model_dump()),
                            "arguments": tool_call.to_json(),
                            "name": tool_call.name
                        }                      
                } for tool_call in tool_calls
            ]
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name:
            message["name"] = name        
        return message
    
    
    
    
    
    def to_chat(self, blocks: BlockList, tools: List[Type[BaseModel]] | None = None, error_blocks: BlockList | None = None) -> List[dict]:        
        
        system_block = blocks.find("system").group()
        if self.output_model is not None:
            system_block.append(self.output_model.render(tools))
        system_message = self.to_message(system_block.render(), role="system")
        messages = (
            blocks
            .filter("system")
            .map(lambda x: self.to_message(x.render(), role=x.role or "user", tool_calls=x.tool_calls, tool_call_id=x.id))
        )        
        
        messages = [
            system_message,
            *messages,         
        ]
        if error_blocks:
            error_messages = error_blocks.map(lambda x: self.to_message(x.render(), role=x.role or "user", tool_calls=x.tool_calls, tool_call_id=x.id))
            messages.extend(error_messages)
        return messages
    
    
    async def stream(
        self,        
    ) -> AsyncGenerator[StreamEvent, None]:
        
        
        messages = [
            self.to_message(b.render(), role=b.role or "user", tool_calls=b.tool_calls, tool_call_id=b.id)
            for b in self.blocks
        ]
        llm_tools = None
        tool_choice = None
        if self.tools and self.output_model is None:
            llm_tools = [self.to_tool(tool) for tool in self.tools]        
            tool_choice = self.config.tool_choice

        try:       
            res_stream = await self.client.chat.completions.create(
                    messages=messages,
                    tools=llm_tools,
                    model=self.config.model,
                    tool_choice=tool_choice,
                    stream=True,
                    logprobs=True,                
                )
            
            block = BlockChunk(sep="")
            yield LLMEvent(type="stream_start")
            async for chunk in res_stream:
                if chunk.choices[0].delta:
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        break
                    content = choice.delta.content
                    try:
                        if choice.logprobs and choice.logprobs.content:                
                            logprob = choice.logprobs.content[0].logprob
                        else:
                            logprob = 0  
                    except:
                        raise ValueError("No logprobs")        
                    c = Chunk(content, logprob=logprob)
                    block.add_content(c)
                    yield c
            yield LLMEvent(type="stream_success")
            yield block
        except Exception as e:
            yield LLMEvent(type="stream_error", data={"error": str(e)})
            raise e

    
    
    async def client_complete(
        self, 
        blocks: BlockList, 
        tools: List[Type[BaseModel]] | None = None, 
        config: LlmConfig | None = None,
        error_blocks: List[BlockChunk] | None = None
    ) -> BlockChunk:
                
        messages = self.to_chat(blocks, tools, error_blocks)
        llm_tools = None
        tool_choice = None
        if tools and self.output_model is None:
            llm_tools = [self.to_tool(tool) for tool in tools]        
            tool_choice = self.config.tool_choice
        
        with Tracer(
            is_traceable=self.is_traceable,
            run_type="llm",
            name=self.__class__.__name__,
            inputs={"messages": messages},
            metadata={
                "ls_model_name": config.model,
                "ls_model_version": "openai",
            },
        ) as llm_run:
            response = await self.client.chat.completions.create(
                messages=messages,
                tools=llm_tools,
                model=config.model,
                tool_choice=tool_choice,
                # **kwargs
            )
            llm_run.end(outputs=response)
            
        
        output = response.choices[0].message
        tool_calls = []
        if tools and output.tool_calls:
            for tool_call in output.tool_calls:
                action_instance = self.from_tool(tools, tool_call)
                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        tool=action_instance
                    ))
                      
        response_block = BlockChunk(
            output.content or "",
            role="assistant",
            tool_calls=tool_calls,
            # id=response.id,
            model=response.model,
            tags=["generation"],
            run_id=str(llm_run.id),
        )
        return response_block
    
    
    
    
    async def client_complete2(self, blocks: BlockChunk, tools: List[Type[BaseModel]] | None = None, config: LlmConfig | None = None) -> BlockChunk:
        
        content = """
        <observation>
            this is an obeservation
        </observation>
        <thought>
            this is a thought
        </thought>        
        <message>
        Hello! How can I assist you today with your seller financing deal? If you would like to review any details of your deal, such as the asset or loan terms, please let me know!        
        </message> 
        <wait_for_response>
            user_input
        </wait_for_response>       
        """
        
        
            
                      
        response_block = BlockChunk(
            content=content,
            role="assistant",
            tool_calls=[],
            # id=response.id,
            model="openai",
            tags=["generation"]
        )
        return response_block
    
    
        



    


    

class OpenAiLLM2(LLM):
    name: str = "OpenAiLLM"    
    # client: OpenAiLlmClient = Field(default_factory=OpenAiLlmClient)
    client: openai.AsyncClient = Field(default_factory=lambda: openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY")))
    model: str = "gpt-4o"
    
    
    

    def render(self, blocks: BlockChunk):
        if not isinstance(blocks, BlockChunk):
            if isinstance(blocks, BlockChunk):
                blocks = BlockChunk.from_block(blocks)
                        
        if block.role == "user" or block.role == "system" or block.role == None:
            return {
                "role": block.role or "user", # type: ignore
                "content": block.render(),
            }
        elif block.role == "assistant":
            oai_msg: dict[str, Any] = {"role": block.role}
            if block.tool_calls:
                tool_calls = []
                for tool_call in block.tool_calls:
                    tool_calls.append({
                    "id": tool_call.id,
                        "type": "function",
                        "function": {
                            # "arguments": json.dumps(action_call.action.model_dump()),
                            "arguments": tool_call.to_json(),
                            "name": tool_call.name
                        }                      
                    })
                oai_msg["tool_calls"] = tool_calls
            else:
                oai_msg["content"] = block.render()
            if block.name:
                oai_msg["name"] = block.name
            return oai_msg
        elif block.role == "tool":
            oai_msg = {
                "tool_call_id": block.uuid,
                "role": "tool",
                "content": block.render()
            }
            if block.name:
                oai_msg["name"] = block.name
            return oai_msg
        else:
            raise ValueError(f"Invalid block role: {block.role}")
      
    # def to_chat(self, blocks: BlockStream) -> List[BaseBlock]:
    #     return blocks.get([
    #         {
    #             "content": [
    #                 "background",
    #                 "task",     
    #                 "rules",
    #             ],
    #             "role": "system"
    #         },            
    #         "history",
    #         "user_message"
    #     ])
        
        
    def to_tools(self, actions: List[Type[BaseModel]]) -> List[dict] | None:
        return Actions(actions).to_openai()
        
    async def complete(
        self, 
        messages: List[dict],   
        tools: List[dict] | None = None, 
        model: str = "gpt-4o", 
        tool_choice: str | None = None, 
        **kwargs
    ):
        openai_completion = await self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
            # **kwargs
        )
        return openai_completion

        
    def parse_response(self, response: openai.types.chat.chat_completion.ChatCompletion, actions: List[Type[BaseModel]] | None):
        output = response.choices[0].message
        tool_calls = []
        if output.tool_calls and actions:
            action_parser = Actions(actions)
            for tool_call in output.tool_calls:
                action_instance = action_parser.from_openai(tool_call)
                # tool_calls.append({"id": tool_call.id, "action": action_instance})
                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        tool=action_instance
                    ))
                
                
        response_block = BlockChunk(
            content=output.content,
            role="assistant",
            tool_calls=tool_calls,
            id=response.id,
            model=response.model,
        )
        return response_block
        
        response_block = BlockChunk(
            output.content,
            role="assistant",
            # id=response.id,
            # model=response.model,
            # content=output.content,
            # action_calls=tool_calls, 
            # raw=response,
            # usage=LlmUsage(
            #     prompt_tokens=response.usage.prompt_tokens,
            #     completion_tokens= response.usage.completion_tokens,
            #     total_tokens= response.usage.total_tokens,
            # )
        )
        return response_block