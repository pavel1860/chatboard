
import json
from time import time
from typing import Any, Coroutine, Dict, Iterable, List, Optional, Tuple, TypeVar, Generic, Union

import aiohttp
from chatboard.clients.openai_client import build_async_openai_client
from langchain.chat_models import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool

from chatboard.text.llms.completion_parsing2 import OutputParser

# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
# from .system_conversation import AIMessage, Conversation, HumanMessage, SystemMessage, from_langchain_message
from .conversation import AIMessage, Conversation, HumanMessage, SystemMessage, from_langchain_message
from pydantic import BaseModel, Field, validator
from .tracer import Tracer
import tiktoken
import yaml
import re
import os

import openai
from openai.types.chat.chat_completion import ChatCompletion


DEFAULT_MODEL = "gpt-3.5-turbo-0125"

default_prompt_config = {
    "model": DEFAULT_MODEL,
    "temperature": 0,
    "max_tokens": None,
    "num_examples": 3,
    "stop_sequences": None,
    "ignore": [],
}



chat_models = [
    "gpt-4o",
    "gpt-3.5-turbo-0125",
    'gpt-3.5-turbo',
    'gpt-4-1106-preview',
    'gpt-3.5-turbo-1106',
    "gpt-4-0125-preview",
]



class LlmError(Exception):
    pass



def encode_logits(string: str, bias_value: int, encoding_model: str) -> int:
    """Returns the number of tokens in a text string."""
    
    return {en: bias_value for en in encoding_model.encode(string)}


def encode_logits_dict(logits, encoding_model = None):
    if encoding_model is None:
        encoding_model = tiktoken.get_encoding("cl100k_base")
    encoded_logits = {}
    for key, value in logits.items():
        item_logits = encode_logits(key, value, encoding_model)
        encoded_logits.update(item_logits)
    return encoded_logits



class LlmChunk(BaseModel):
    content: str
    finish: Optional[bool] = False


class PhiLlmClient(BaseModel):
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
    # async def complete(self, msgs, max_tokens=200, stop_sequences=[]):
    #     prompt = self.preprocess(msgs)
    #     async with aiohttp.ClientSession() as session:        
    #         content, status = await self.fetch(session, self.url, data={
    #             "prompt": prompt,
    #             "max_new_tokens": max_tokens,
    #             "stop_sequences": stop_sequences        
    #         })
    #         return content
    # async def complete(self, msgs, **kwargs):
    #     prompt = self.preprocess(msgs)
    #     async with aiohttp.ClientSession() as session:        
    #         content, status = await self.fetch(session, self.url, data={
    #             # "prompt": prompt,
    #             "max_new_tokens": kwargs.get("max_tokens", 200),
    #             "stop_sequences": kwargs.get("stop", [])        
    #         })
    #         # return content
    #         return AIMessage(content=content)

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


class OpenAiLlmClient:


    def __init__(self, api_key=None, api_version=None, azure_endpoint=None, azure_deployment=None):
        self.client = build_async_openai_client()
        # if azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", None):
        #     self.client = openai.AsyncAzureOpenAI(
        #         api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
        #         api_version=api_version or os.getenv("OPENAI_API_VERSION", "2023-12-01-preview"),
        #         azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        #         azure_deployment=azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        #     ) 
        # elif os.environ.get("OPENAI_API_KEY", None):
        #     self.client = openai.AsyncClient(
        #         api_key=api_key or os.getenv("OPENAI_API_KEY")
        #     )
        # else:
        #     raise ValueError("OpenAI API Key not found in environment variables")

    def preprocess(self, msgs):
        return [msg.to_openai() for msg in msgs]

    async def complete(self, msgs, tools=None, **kwargs):
        # tools = tools or []
        msgs = self.preprocess(msgs)
        openai_completion = await self.client.chat.completions.create(
            messages=msgs,
            tools=tools,
            **kwargs
        )
        return openai_completion
        # output = openai_completion.choices[0].message
        # return AIMessage(content=output.content or '', tool_calls=output.tool_calls)
        # return output



class LLM(BaseModel):

    # model="gpt-3.5-turbo-0125"
    # name="OpenAiLLM"
    model: str
    name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    logit_bias: Optional[Dict[str, int]] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    is_traceable: Optional[bool] = True
    seed: Optional[int] = None
    client: Union[OpenAiLlmClient, PhiLlmClient]    

    class Config:
        arbitrary_types_allowed = True


    def get_llm(
            self, 
            **kwargs            
        ):

        model_kwargs={}
        stop_sequences = kwargs.get("stop_sequences", self.stop_sequences)
        if stop_sequences:
            model_kwargs['stop'] = stop_sequences
        logit_bias = kwargs.get("logit_bias", self.logit_bias)
        if logit_bias:            
            model_kwargs['logit_bias'] = logit_bias
        top_p = kwargs.get("top_p", self.top_p)
        if top_p:
            if top_p > 1.0 or top_p < 0.0:
                raise ValueError("top_p must be between 0.0 and 1.0")
            model_kwargs['top_p'] = top_p
        presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
        if presence_penalty:
            if presence_penalty > 2.0 or presence_penalty < -2.0:
                raise ValueError("presence_penalty must be between -2.0 and 2.0")
            model_kwargs['presence_penalty'] = presence_penalty
        frequency_penalty = kwargs.get("frequency_penalty", self.frequency_penalty)
        if frequency_penalty:
            if frequency_penalty > 2.0 or frequency_penalty < -2.0:
                raise ValueError("frequency_penalty must be between -2.0 and 2.0")
            model_kwargs['frequency_penalty'] = frequency_penalty
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            model_kwargs['temperature'] = temperature
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            model_kwargs['max_tokens'] = max_tokens

        model = kwargs.get("model", self.model)
        if model is not None:
            model_kwargs['model'] = model        
        else:
            model_kwargs['model'] = DEFAULT_MODEL
        
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            model_kwargs['seed'] = seed
        return model_kwargs


    async def complete(
            self, 
            msgs: List[SystemMessage | HumanMessage | AIMessage], 
            tools: List[BaseModel] | None=None, 
            tool_choice: BaseModel | None=None,
            response_model: BaseModel | None=None,
            tracer_run=None, 
            metadata={}, 
            completion=None, 
            **kwargs):

        llm_kwargs = self.get_llm(**kwargs)

        extra = metadata.copy()
        extra.update(llm_kwargs)  

        if completion:
            return custom_completion(completion)


        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            run_type="llm",
            name=self.name,
            inputs={"messages": [msg.to_openai() for msg in msgs]},
            extra=extra,
        ) as llm_run:
        
            # output = await self.llm.ainvoke(messages)
            # openai_completion = await self.client.chat.completions.create(
            #     messages=openai_messages,
            #     **llm_kwargs
            # )

            tool_schema = [convert_to_openai_tool(t) for t in tools] if tools else None
            tool_choice_schema = convert_to_openai_tool(tool_choice) if tool_choice else None

            completion = await self.client.complete(
                msgs, 
                tools=tool_schema, 
                tool_choice=tool_choice_schema, 
                **llm_kwargs)

            llm_run.end(outputs=completion)
            # return completion
            output = completion.choices[0].message
            finish_reason = completion.choices[0].finish_reason

            if response_model:
                parser = OutputParser(response_model)
                try:
                    return parser.parse(output.content)
                except Exception as e:
                    print("########## ERROR PARSING OUTPUT ##########") 
            if finish_reason == "tool_calls":
                tool_lookup = {t.__name__: t for t in tools}
                output_tools = []
                for tool_call in output.tool_calls:
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_cls = tool_lookup.get(tool_call.function.name, None)
                    if tool_cls:
                        output_tools.append(tool_cls(**tool_args))
                    else:
                        raise ValueError(f"Tool {tool_call.function.name} not found in tools")                    
                    print(tool_call.function.name, tool_args)
                    tool_call.function.name
                    # tool_cls = tools[tool_call.function.name]['info'].default.__class__
                    
                return output_tools


            return AIMessage(content=output.content or '', tool_calls=output.tool_calls)
        

    async def send_stream(self, openai_messages, tracer_run, metadata={}, completion=None, **kwargs):

        llm_kwargs = self.get_llm(**kwargs)

        extra = metadata.copy() if metadata else {}
        extra.update(llm_kwargs)  

        # if completion:
            # yield custom_completion(completion)


        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            run_type="llm",
            name=self.name,
            inputs={"messages": openai_messages},
            extra=extra,
        ) as llm_run:
        
            # output = await self.llm.ainvoke(messages)
            stream = await self.client.chat.completions.create(
                messages=openai_messages,
                stream=True,
                **llm_kwargs,
            )
            openai_completion = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    openai_completion += chunk.choices[0].delta.content
                    yield LlmChunk(
                        content=chunk.choices[0].delta.content,
                    )
                    # print(chunk.choices[0].delta.content, end="")

            # llm_run.end(outputs=openai_completion)
            llm_run.end(outputs={
                "messages": [AIMessage(content=openai_completion).to_openai()]
            })
            # llm_run.end(outputs=AIMessage(content=openai_completion).to_openai())
            yield LlmChunk(
                content=openai_completion,
                finish=True
            )



    async def send_with_tools(self, openai_messages, openai_tools, tracer_run=None, tool_choice=None, metadata={}, **kwargs):

        llm_kwargs = self.get_llm(**kwargs)

        extra = metadata.copy()
        extra.update(llm_kwargs)      

        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            run_type="llm",
            name=self.name,
            inputs={
                "messages": openai_messages,
                "tools": openai_tools
            },
            extra=extra,
        ) as llm_run:
        
        
            openai_completion = await self.client.chat.completions.create(
                messages=openai_messages,
                tools=openai_tools,
                tool_choice=tool_choice,
                **llm_kwargs
            )
            
            llm_run.end(outputs=openai_completion)
            
            return openai_completion

        


class OpenAiLLM(LLM):
    name: str = "OpenAiLLM"    
    client: Union[OpenAiLlmClient, PhiLlmClient] = Field(default_factory=OpenAiLlmClient)
    model: str = "gpt-3.5-turbo-0125"
    api_key: Optional[str] = None

    # def __init__(self, **data):    
    #     client = OpenAiLlmClient()
    #     OpenAiLlmClient(api_key=self.api_key)
    #     super().__init__(**data)


class PhiLLM(LLM):
    name: str = "PhiLLM"
    stop_sequences: List[str]=["Instruct"]
    client: Union[OpenAiLlmClient, PhiLlmClient] = Field(default_factory=PhiLlmClient)
    model: str = "microsoft/phi-2"


class AzureOpenAiLLM(LLM):
    name: str = "AzureOpenAiLLM"
    # client: Union[OpenAiLlmClient, PhiLlmClient] = Field(default_factory=OpenAiLlmClient)
    client: Union[OpenAiLlmClient, PhiLlmClient] = None
    model: str = "gpt-3.5-turbo-0125"
    api_key: Optional[str] = None
    api_version: Optional[str] = "2023-12-01-preview"
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    

    def __init__(self, **data):        
        super().__init__(**data)
        self.client = OpenAiLlmClient(
            api_key=data.get("api_key", self.api_key),
            api_version=data.get("api_version", self.api_version),
            azure_endpoint=data.get("azure_endpoint", self.azure_endpoint),
            azure_deployment=data.get("azure_deployment", self.azure_deployment),
        )

class CustomMessage(BaseModel):
    content: str
    role: Optional[str] = "assistant"
    function_call: Optional[Dict[str, Any]] = None


class CustomChoice:

    def __init__(self, content) -> ChatOpenAI:
        self.message = CustomMessage(content=content)



class CustomCompletion:
    
    def __init__(self, content):
        self.id = "chatcmpl-custom"
        self.choices = [
            CustomChoice(content)
        ]
        self.created = time()

def custom_completion(completion):
    return CustomCompletion(completion)
    