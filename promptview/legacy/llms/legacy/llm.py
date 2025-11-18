
import asyncio
import json
import os
import re
from time import time
from typing import (Any, Callable, Coroutine, Dict, Generic, Iterable, List,
                    Literal, Optional, Tuple, Type, TypeVar, Union)

import tiktoken
from promptview.llms.clients.azure_client import AzureOpenAiLlmClient
from promptview.llms.clients.openai_client import OpenAiLlmClient
from promptview.llms.clients.phi_llm_client import PhiLlmClient
from promptview.llms.exceptions import LLMToolNotFound
from promptview.llms.utils.completion_parsing import PromptParsingException
from promptview.llms.utils.completion_parsing2 import OutputParser
from promptview.utils.function_utils import call_function
from promptview.utils.model_utils import schema_to_function
from pydantic import BaseModel, Field, ValidationError, validator

from ..messages import AIMessage, HumanMessage, SystemMessage
from ..tracer import Tracer

ToolChoice = Literal['auto', 'required', 'none']







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




def encode_logits(string: str, bias_value: int, encoding_model: tiktoken.Encoding) -> dict:
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
    





class LLM(BaseModel):

    model: str
    name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    logit_bias: Optional[Dict[str, int]] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: bool | None = None
    frequency_penalty: Optional[float] = None
    is_traceable: Optional[bool] = True
    seed: Optional[int] = None
    client: Union[OpenAiLlmClient, PhiLlmClient, AzureOpenAiLlmClient]    
    parallel_tool_calls: Optional[bool] = True

    class Config:
        arbitrary_types_allowed = True


    def get_llm(
            self, 
            tools=None,
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
            
        parallel_tool_calls = kwargs.get("parallel_tool_calls", self.parallel_tool_calls)
        if parallel_tool_calls is not None and tools:
            model_kwargs['parallel_tool_calls'] = parallel_tool_calls            
            
        logprobs = kwargs.get("logprobs", self.logprobs)
        if logprobs is not None:
            model_kwargs['logprobs'] = logprobs
        
        return model_kwargs


    async def complete(
            self, 
            msgs: List[SystemMessage | HumanMessage | AIMessage], 
            tools: List[Type[BaseModel]] | None=None, 
            tool_choice: ToolChoice | BaseModel | None = None,
            response_model: type[BaseModel] | None=None,
            tracer_run=None, 
            metadata={},
            completion=None,
            retries=3, 
            smart_retry=True,
            output_parser: Callable | None=None,
            # logprobs: bool=False,
            **kwargs):

        llm_kwargs = self.get_llm(tools=tools, **kwargs)

        extra = metadata.copy()
        extra.update(llm_kwargs)  

        # if completion:
            # return custom_completion(completion)


        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            run_type="llm",
            name=self.name,
            inputs={"messages": [msg.to_openai() for msg in msgs]},
            extra=extra,
        ) as llm_run:
        
            tool_schema = [schema_to_function(t) for t in tools] if tools else None
            
            if isinstance(tool_choice, BaseModel):
                tool_choice_schema =  {"type": "function", "function": {"name": tool_choice.__class__.__name__}}
            else:
                if tool_choice and tool_choice not in ["auto", "required", "none"]:
                    raise ValueError("tool_choice must be one of 'auto', 'required', 'none' or an BaseModel")
                tool_choice_schema = tool_choice
            
            for try_num in range(retries):
                try:
                    completion = await self.client.complete(
                        msgs, 
                        tools=tool_schema, 
                        tool_choice=tool_choice_schema, 
                        run_id=str(llm_run.id),
                        # logprobs=logprobs,
                        **llm_kwargs)                    
                    ai_message = self.parse_output(completion, tools, tool_choice, response_model)
                    ai_message.run_id = str(llm_run.id)
                    if output_parser:                        
                        ai_message = await call_function(
                            output_parser,
                            ai_message,                            
                            llm_response=completion, 
                        )                    
                    
                    llm_run.end(outputs=completion)
                    return ai_message
                except LLMToolNotFound as e:
                    print(f"try {try_num} tool not found error")
                    if try_num == retries - 1:
                        llm_run.end(errors=str(e))
                        raise e
                    if smart_retry:
                        msgs.append(HumanMessage(content=f"there is no such tool:\n{str(e)}"))
                except PromptParsingException as e:
                    print("prompt parsing exception")
                    if try_num == retries - 1:
                        llm_run.end(errors=str(e))
                        raise e
                    if smart_retry:
                        msgs.append(HumanMessage(content=f"could not parse output:\n{str(e)}"))
                except json.JSONDecodeError as e:
                    if try_num == retries - 1:
                        llm_run.end(errors=str(e))
                        raise e
                    if smart_retry:
                        msgs.append(HumanMessage(content=f"bad json output:\n{str(e)}"))
                except ValidationError as e:
                    print(f"try {try_num} validation error")
                    if try_num == retries - 1:
                        llm_run.end(errors=str(e))
                        raise e
                    msgs.append(HumanMessage(content=f"something is wrong with the parameters:\n{str(e)}"))
                except Exception as e:
                    llm_run.end(errors=str(e))
                    # import pickle
                    # from datetime import datetime
                    # date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    # with open(f"msgs_{date_str}.pkl", "wb") as f:
                    #     pickle.dump(msgs, f)
                    raise e
                    
        
    def parse_output(self, completion, tools, tool_choice: ToolChoice | BaseModel | None, response_model):
        output = completion.choices[0].message
        finish_reason = completion.choices[0].finish_reason
        tool_calls = completion.choices[0].message.tool_calls

        if response_model:
            parser = OutputParser(response_model)
            try:
                parsing_output = parser.parse(output.content)
                return AIMessage(content=output.content, actions=[parsing_output])
            except Exception as e:
                print("########## ERROR PARSING OUTPUT ##########") 
        actions = None
        if finish_reason == "stop":
            if tool_choice == "required" and not tool_calls:
                raise LLMToolNotFound("Tool call is required")            
        if tool_calls:
            if completion.choices[0].message.content is not None:
                print("### tool call with message: ", completion.choices[0].message.content)
            tool_lookup = {t.__name__: t for t in tools}
            actions = []
            for tool_call in output.tool_calls:
                tool_args = json.loads(tool_call.function.arguments)
                tool_cls = tool_lookup.get(tool_call.function.name, None)
                if tool_cls:
                    actions.append(tool_cls(**tool_args))
                else:
                    raise LLMToolNotFound(f"Tool {tool_call.function.name} not found in tools")
                tool_call.function.name                
        # if finish_reason == "stop":
        #     if tool_choice == "required":
        #         raise LLMToolNotFound("Tool call is required")            
        # elif finish_reason == "tool_calls":
        #     if completion.choices[0].message.content is not None:
        #         print("### tool call with message: ", completion.choices[0].message.content)
        #     tool_lookup = {t.__name__: t for t in tools}
        #     actions = []
        #     for tool_call in output.tool_calls:
        #         tool_args = json.loads(tool_call.function.arguments)
        #         tool_cls = tool_lookup.get(tool_call.function.name, None)
        #         if tool_cls:
        #             actions.append(tool_cls(**tool_args))
        #         else:
        #             raise LLMToolNotFound(f"Tool {tool_call.function.name} not found in tools")
        #         tool_call.function.name                
            # return actions
        return AIMessage(content=output.content, tool_calls=output.tool_calls, actions=actions)
        

    # async def send_stream(self, openai_messages, tracer_run, metadata={}, completion=None, **kwargs):

    #     llm_kwargs = self.get_llm(**kwargs)

    #     extra = metadata.copy() if metadata else {}
    #     extra.update(llm_kwargs)  


    #     with Tracer(
    #         is_traceable=self.is_traceable,
    #         tracer_run=tracer_run,
    #         run_type="llm",
    #         name=self.name,
    #         inputs={"messages": openai_messages},
    #         extra=extra,
    #     ) as llm_run:
        
    #         stream = await self.client.chat.completions.create(
    #             messages=openai_messages,
    #             stream=True,
    #             **llm_kwargs,
    #         )
    #         openai_completion = ""
    #         async for chunk in stream:
    #             if chunk.choices[0].delta.content is not None:
    #                 openai_completion += chunk.choices[0].delta.content
    #                 yield LlmChunk(
    #                     content=chunk.choices[0].delta.content,
    #                 )
                    
            
    #         llm_run.end(outputs={
    #             "messages": [AIMessage(content=openai_completion).to_openai()]
    #         })
    #         yield LlmChunk(
    #             content=openai_completion,
    #             finish=True
    #         )


    





class CustomMessage(BaseModel):
    content: str
    role: Optional[str] = "assistant"
    function_call: Optional[Dict[str, Any]] = None


class CustomChoice:

    def __init__(self, content):
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
    