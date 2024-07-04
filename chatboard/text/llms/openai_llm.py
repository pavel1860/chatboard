
from time import time
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar, Generic, Union

from langchain.chat_models import ChatOpenAI

from chatboard.clients.openai_client import build_async_openai_client


# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from .system_conversation import AIMessage, Conversation, HumanMessage, SystemMessage, from_langchain_message
from pydantic import BaseModel
from .tracer import Tracer
import tiktoken
import yaml
import re

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
    "gpt-3.5-turbo-0125",
    'gpt-3.5-turbo',
    'gpt-4-1106-preview',
    'gpt-3.5-turbo-1106',
    "gpt-4-0125-preview",
]





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


class OpenAiLLM:


    def __init__(
            self, 
            model=DEFAULT_MODEL,
            # model='gpt-3.5-turbo-1106',
            name="OpenAiLLM",
            temperature=None, 
            max_tokens=None, 
            stop_sequences=None,
            stream=False,            
            logit_bias=None,            
            top_p=None,
            presence_penalty=None,
            frequency_penalty=None,
            suffix=None, 
            is_traceable=True, 
            seed=None          
        ) -> None:
        if model is not None and model not in chat_models:
            raise ValueError(f"model ({model}) must be one of {chat_models}")
        self.name = name
        self.client = build_async_openai_client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.stream = stream
        self.seed = seed
        self.logit_bias = None
        if logit_bias:
            self.logit_bias = encode_logits_dict(logit_bias, tiktoken.get_encoding("cl100k_base"))
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.suffix = suffix
        self.is_traceable = is_traceable
        # self.encoding_model = tiktoken.get_model(model)



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
            # model_kwargs['logit_bias'] = encode_logits_dict(logit_bias, self.encoding_model)
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
        suffix = kwargs.get("suffix", self.suffix)
        if suffix:
            model_kwargs['suffix'] = suffix
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


    async def send(self, openai_messages, tracer_run, metadata={}, completion=None, **kwargs):

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
            inputs={"messages": openai_messages},
            extra=extra,
        ) as llm_run:
        
            # output = await self.llm.ainvoke(messages)
            openai_completion = await self.client.chat.completions.create(
                messages=openai_messages,
                **llm_kwargs
            )
            llm_run.end(outputs=openai_completion)
            return openai_completion
        

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
    return {
        "id": "chatcmpl-custom",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": completion,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None
                }
            }
        ],
        "created": time(),
        "model": "gpt-3.5-turbo-0125",
        "object": "chat.completion",
        "system_fingerprint": "fp_2b778c6b35",
        "usage": {
            "completion_tokens": 182,
            "prompt_tokens": 1094,
            "total_tokens": 1276
        }
    }