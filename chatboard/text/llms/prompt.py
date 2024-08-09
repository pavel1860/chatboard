

import asyncio
import inspect
import json
import os
import pathlib
import random
import textwrap
import traceback
from enum import Enum
from typing import (Any, Coroutine, Dict, Generic, List, Optional, Tuple,
                    TypeVar, Union)

import tiktoken
from chatboard.text.llms.views import ViewModel
from chatboard.text.vectors.rag_documents2 import RagDocuments
from langchain_core.pydantic_v1 import BaseModel, ConfigDict, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith.run_trees import RunTree
from pydantic import ValidationError
# from pydantic import BaseModel, ConfigDict, Field
from pydantic.generics import GenericModel

from .completion_parsing import (CompletionParser, PromptParsingException,
                                 auto_split_completion,
                                 auto_split_list_completion, is_list_model,
                                 parse_completion, parse_model_list,
                                 unpack_list_model)
from .conversation import (AIMessage, ChatMessageType, Conversation,
                           ConversationRag, HumanMessage, SystemConversation,
                           SystemMessage, from_langchain_message)
# from .openai_llm import OpenAiLLM
from .llm import AzureOpenAiLLM, OpenAiLLM
from .prompt_manager import PromptManager
from .rag_manager import RagVectorSpace
from .tracer import Tracer

# from pydantic.errors import ValidationError


# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
# from components.etl.completion_parsing import auto_split_completion, auto_split_list_completion, is_list_model, parse_completion, parse_model_list, unpack_list_model
# from components.etl.conversation import SystemConversation, AIMessage, Conversation, ConversationRag, HumanMessage, SystemMessage, from_langchain_message


PROMPT_REPO_HOME = os.getenv("PROMPT_REPO_HOME", "/app/prompts")


def get_tool_scheduler_prompt(tool_dict):
    tool_function = tool_dict['function']
    prompt = f"""{tool_function["name"]}: {tool_function["description"]}\n\tparameters:"""
    for prop, value in tool_function["parameters"]['properties'].items():
        prompt += f"\n\t\t{prop}: {value['description']}"
    return prompt


# def encode_logits(string: str, bias_value: int, encoding_model: str) -> int:
#     """Returns the number of tokens in a text string."""
    
#     return {en: bias_value for en in encoding_model.encode(string)}



# def encode_logits_dict(logits, encoding_model):
#     encoded_logits = {}
#     for key, value in logits.items():
#         item_logits = encode_logits(key, value, encoding_model)
#         encoded_logits.update(item_logits)
#     return encoded_logits


class PromptErrorTypes(Enum):
    PYDANTIC_ERROR = "pydantic_error"


class PromptError(BaseModel):
    error: str
    error_type: PromptErrorTypes


def to_dict(pydantic_model):
    d = {}
    for field_name, field_info in pydantic_model.__fields__.items():
        d[field_name] = None
    return d    

def sanitize_content(content):
    content = content.strip().strip('"').strip("'").strip('\n')
    return content


class ChatResponse(BaseModel):
    value: Any
    run_id: str
    conversation: Conversation
    error: Optional[PromptError] = None
    tools: Optional[List[BaseModel]] = None
    examples: Optional[List[Any]] = None

    def __repr__(self) -> str:
        return f"{self.value!r}"
    
    def __str__(self) -> str:
        return f"{self.value!r}"
    
    def to_dict(self):
        if hasattr(self.value, 'to_dict') and callable(getattr(self.value, 'to_dict')):
            value = self.value.to_dict()
        else:
            value = self.value

        return {
            "value": value,
            "run_id": self.run_id,
            # "completion": self.completion,
            # "messages": [m.to_dict() for m in self.messages],
            # "costs": self.costs,
            # "model": self.model
        }
    
    class Config:
        arbitrary_types_allowed = True




class PromptChunkTypes(Enum):
    PROMPT_START = "prompt_start"
    PROMPT_UPDATE = "prompt_update"
    PROMPT_FINISH = "prompt_finish"
    PROMPT_ERROR = "prompt_error"
    PROMPT_MILESTONE = "prompt_milestone"


class ChatChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    msg_type: PromptChunkTypes
    func: Optional[str] = None
    content: Optional[Union[str, BaseModel]]
    output: Optional[Union[BaseModel]]
    prompt_response: Optional[ChatResponse] = None
    field: Optional[str]


    

    def to_dict(self):
        return {
            "msg_type": self.msg_type.value,
            "func": self.func,
            "content": self.content,
            "prompt_response": self.prompt_response.to_dict() if self.prompt_response is not None else None,
            "field": self.field
        }



class PromptException(Exception):
    pass


def validate_input_variables(kwargs, input_variables):
    # for key, value in params.items():
    not_found = []
    for invar in input_variables:
        if invar not in kwargs:
            not_found.append(invar)
    if not_found:
        raise ValueError(f"params {not_found} not found in passed parameters")
    return True


def filter_kwargs(kwargs, func_):
    signature = inspect.signature(func_)
    parameters = signature.parameters
    argument_names = [param for param in parameters if parameters[param].default == inspect.Parameter.empty]
    return {k: v for k, v in kwargs.items() if k in argument_names}



class FewShotRag(BaseModel):
    key: str
    value: str


class Prompt(BaseModel):
    system_prompt: str | None=None
    model: str= "gpt-3.5-turbo-0125"
    name: Optional[str] = None
    llm: Union[OpenAiLLM, AzureOpenAiLLM] = Field(default_factory=OpenAiLLM)
    output_class: Optional[BaseModel] = None
    rag_space: Optional[RagDocuments] = None
    rag_namespace: Optional[str] = None
    tools: Optional[List[ViewModel]] = None    
    function_calling: bool = False
    is_traceable: bool=True,
    # seed: int = None                 
    class Config:
        arbitrary_types_allowed = True


    def __init__(self, **data):
        super().__init__(**data)
        if self.name is None:
            self.name = self.__class__.__name__
        if self.rag_namespace:
            self.rag_space = RagDocuments(self.rag_namespace, metadata_class=FewShotRag)

        

    def get_tracer(self, inputs, metadata):
        pipeline = RunTree(
            name=self.name,
            run_type="prompt",
            inputs=inputs,
            extra=metadata,
        )
        return pipeline
    

    async def build_conversation(self, prompt, conversation=None, input_postfix=None, system_postfix=None, **kwargs):
        
        if self.system_prompt:
            system_message = SystemMessage(content=self.system_prompt)
        elif self.prompt_manager is not None:
            prompt_system_template, prompt_metadata = self.prompt_manager.get_template(
                is_chat_template=True, 
                template_type="system", 
                filename=kwargs.get("system_filename", self.system_filename)
            )
            validate_input_variables(kwargs, prompt_metadata['input_variables'])
            langchain_system_message = prompt_system_template.format(**kwargs)
            system_message = from_langchain_message(langchain_system_message)
        else:
            system_message=None

        if system_postfix:
            system_message.content += "\n" + system_postfix

        if prompt is not None:            
            user_message = HumanMessage(
                content=prompt,
            ) 
            prompt_metadata = None                     
        else:
            prompt_user_template, prompt_metadata = self.prompt_manager.get_template(
                is_chat_template=True, 
                template_type="user", 
                filename=kwargs.get("user_filename", self.user_filename)
            )
            validate_input_variables(kwargs, prompt_metadata['input_variables'])
            langchain_user_message = prompt_user_template.format(**kwargs)
            user_message = from_langchain_message(langchain_user_message)
            if input_postfix:
                user_message.content += "\n" + input_postfix
            # conversation.append(from_langchain_message(user_message))
            # conversation.user_metadata = prompt_metadata
        
        system_conversation = SystemConversation(
            system_message=system_message, 
            conversation=conversation,
            system_metadata=prompt_metadata,
            user_metadata=prompt_metadata
        )
        system_conversation.append(user_message)
        
        return system_conversation
    
    def get_extra(self, **kwargs):
        return {
            "system_filename": self.system_filename, 
            "user_filename": self.user_filename,
            "rag_index": self.rag_index,
        }
    
    # async def system_prompt(self, **kwargs):
    #     return None

    # async def prompt(self, **kwargs):
    #     return "Say something"
    async def render_system_prompt(self, **kwargs):
        return None

    async def render_prompt(self, **kwargs):
        return "Say something"
    
    async def _get_prompt_msg(self, **kwargs):
        postfix = kwargs.get('prompt_postfix', '')
        if kwargs.get('prompt'):
            prompt = kwargs['prompt']
            return HumanMessage(content=prompt + postfix, name="user")
        prompt_kwargs = filter_kwargs(kwargs, self.render_prompt)
        prompt = await self.render_prompt(**prompt_kwargs)
        prompt = textwrap.dedent(prompt).strip()
        return HumanMessage(content=prompt + postfix, name="user") 
    
    async def _get_system_msg(self, **kwargs):
        if self.system_prompt is not None:
            return SystemMessage(content=self.system_prompt)
        system_prompt = await self.render_system_prompt(**kwargs)
        if system_prompt is None:
            return None
        system_prompt = textwrap.dedent(system_prompt).strip()
        return SystemMessage(content=system_prompt)


    async def render_prompts(self, **kwargs):
        system_msg = await self._get_system_msg(**kwargs)
        prompt_msg = await self._get_prompt_msg(**kwargs)        
        return prompt_msg, system_msg


    # async def rag(self, **kwargs):            

    async def conversation(self, **kwargs: Any):
        prompt_msg, system_msg = await self.render_prompts(**kwargs)
        msgs: List[ChatMessageType] = [system_msg] if system_msg else []
        if self.rag_namespace and self.rag_space:
            rag_results = await self.rag_space.similarity(prompt_msg.content)
            for rag_res in rag_results:
                msgs.append(HumanMessage(content=rag_res.metadata.key, name="user_example"))
                msgs.append(AIMessage(content=rag_res.metadata.value, name="assistant_example"))
        msgs.append(prompt_msg)
        return msgs
    

    async def __call__(
            self, 
            prompt=None, 
            history=[], 
            tracer_run=None, 
            output_conversation=False, 
            retries=3,
            smart_retry=True,
            **kwargs: Any,
            
        ) -> Any:
        if prompt is not None:
            kwargs['prompt'] = prompt
        log_kwargs = {}
        log_kwargs.update(kwargs)        
        # extra = self.get_extra(**kwargs)
        
        if prompt is not None:
            log_kwargs['prompt'] = prompt
        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            name=self.name,
            run_type="prompt",
            inputs={
                "input": log_kwargs,
                # "messages": conversation.messages
            },
            # extra=extra,
        ) as prompt_run:
            msgs = await self.conversation(**kwargs)
            msgs = [m for m in msgs if m is not None]
            tools = [m.to_tool() for m in self.tools] if self.tools else None
            for try_num in range(retries):
                try:
                    completion_msg = await self.llm.complete(
                            msgs=msgs,
                            tools=tools,
                            tracer_run=prompt_run,
                            tags=[f"try_{try_num}"] if try_num > 0 else None,
                            # metadata=system_conversation.get_metadata(),            
                            **kwargs
                        )
                    # output = openai_completion.choices[0].message
                    completion_msg.output = await self.parser(completion_msg.content)
                    completion_msg.run_id = str(prompt_run.id)
                    prompt_run.end(outputs={'output': completion_msg})
                    if output_conversation:
                        return msgs + [completion_msg]
                    return completion_msg
                except (KeyError, PromptParsingException) as e:
                    print(f"try {try_num} KeyError: {str(e)}")
                    if try_num == retries - 1:
                        prompt_run.end(errors= str(e))
                        raise e
                    if smart_retry:
                        msgs.append(AIMessage(content=completion_msg.content))
                        msgs.append(HumanMessage(content=f"something is wrong with the parameters:\n{str(e.__class__.__name__)}: {str(e)}.\n please try again, in the format specified without any apologies or commends."))
                        prompt_run.add_tags([f"try_{try_num + 1}"])
                except Exception as e:
                    traceback.print_exc()
                    prompt_run.end(errors= str(e))
                    raise e

                    
    

    async def parser(self, completion):
        if self.function_calling:
            completion_parser = CompletionParser(self.output_class)
            return await completion_parser.parse(completion)
        
        if self.output_class:
            # try:
            if is_list_model(self.output_class):
                list_model = unpack_list_model(self.output_class)
                if not self.delimiter:
                    raise ValueError("delimiter must be provided for list models")
                return parse_model_list(completion, list_model, self.delimiter)
            else:
                return parse_completion(completion, self.output_class)
        else:
            return completion
            

    async def __call__2(self, prompt=None, conversation= None, tracer_run=None, **kwargs: Any) -> Any:
        
        system_conversation = await self.build_conversation(prompt, conversation, **kwargs)
        # prompt_run = self.get_tracer(kwargs, prompt_metadata)
        log_kwargs = {}
        log_kwargs.update(kwargs)
        extra = self.get_extra(**kwargs)
        
        if prompt is not None:
            log_kwargs['prompt'] = prompt

        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            name=self.name,
            run_type="prompt",
            inputs={
                "input": log_kwargs,                
                # "messages": conversation.messages
            },
            extra=extra,
        ) as prompt_run:
            examples = None
            if self.rag_space and self.rag_length:
                examples = await self.rag_space.similarity(system_conversation.get_messages()[-1:], self.rag_length)
                example_messages = []
                for e in examples:
                    inputs = e.metadata.get_inputs() if hasattr(e.metadata, "get_inputs") else e.metadata.inputs
                    output = e.metadata.get_output() if hasattr(e.metadata, "get_output") else e.metadata.output
                    example_messages += inputs + [output]
                system_conversation.add_examples(example_messages)
            # openai_messages = await self.to_openai(system_conversation, examples)
            openai_messages = system_conversation.to_openai()

            openai_completion = await self.llm.send(
                openai_messages=openai_messages, 
                tracer_run=prompt_run, 
                metadata=system_conversation.get_metadata(),            
                **kwargs
            )

            output = openai_completion.choices[0].message

            if self.pydantic_model:
                try:
                    if is_list_model(self.pydantic_model):
                        list_model = unpack_list_model(self.pydantic_model)
                        if not self.delimiter:
                            raise ValueError("delimiter must be provided for list models")
                        content = parse_model_list(output.content, list_model, self.delimiter)
                    else:
                        content = parse_completion(output.content, self.pydantic_model)
                except Exception as e:
                    prompt_run.end(errors={"error": output.content})
                    return ChatResponse(
                        value=output.content,
                        run_id=str(prompt_run.id),
                        conversation=system_conversation.conversation,
                        examples=examples,
                        error=PromptError(error=str(e), error_type=PromptErrorTypes.PYDANTIC_ERROR),
                    )
            else:
                content = output.content    


            kwargs['run_id'] = str(prompt_run.id)
            parsed_output = await self._call_parser(content, context={'kwargs': kwargs})        

            
            response_message = AIMessage(content=output.content)
            # ret_conversation = conversation.copy()
            # ret_conversation.append(response_message)
            system_conversation.append(response_message)

            # prompt_run.end(outputs={"messages": [response_message], 'output': parsed_output})
            prompt_run.end(outputs={'output': parsed_output})

            return ChatResponse(
                value=parsed_output,
                run_id=str(prompt_run.id),
                conversation=system_conversation.conversation.copy(),
                examples=examples
            )
        
    
    async def call_stream(self, prompt=None, conversation= None, tracer_run=None, **kwargs: Any) -> Any:
        system_conversation = await self.build_conversation(prompt, conversation, **kwargs)

        log_kwargs = {}
        log_kwargs.update(kwargs)
        extra = self.get_extra(**kwargs)
        
        if prompt is not None:
            log_kwargs['prompt'] = prompt

        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            name=self.name,
            run_type="prompt",
            inputs={
                "input": log_kwargs,                
                # "messages": conversation.messages
            },
            extra=extra,
        ) as prompt_run:
            examples = None
            if self.rag_space and self.rag_length:
                examples = await self.rag_space.similarity(system_conversation.get_messages()[-1:], self.rag_length)
                example_messages = []
                for e in examples:
                    ex_inputs_msgs = e.metadata.get_inputs() if hasattr(e.metadata, "get_inputs") else e.metadata.inputs
                    ex_output_msg = e.metadata.get_output() if hasattr(e.metadata, "get_output") else e.metadata.output
                    example_messages += ex_inputs_msgs + [ex_output_msg]
                system_conversation.add_examples(example_messages)

            openai_messages = system_conversation.to_openai()
            

            async for chunk in self._call_llm_stream(
                    openai_messages, 
                    self.name, 
                    metadata=system_conversation.get_metadata(), 
                    pydantic_model=self.pydantic_model, 
                    tracer_run=prompt_run, 
                    **kwargs
                ):
                yield chunk
            # output = to_dict(self.pydantic_model) if self.pydantic_model else None
            
            # curr_field = None
            # curr_content = ""
            # yield ChatChunk(
            #     msg_type=PromptChunkTypes.PROMPT_START,
            #     func=self.name,
            #     content=None,
            #     field=curr_field
            # )
            # try:            
            #     async for chunk in self.llm.send_stream(
            #         openai_messages=openai_messages, 
            #         tracer_run=prompt_run, 
            #         metadata=system_conversation.get_metadata(),            
            #         **kwargs
            #     ):
                    
            #         if chunk.finish is False:
            #             yield ChatChunk(
            #                 msg_type=PromptChunkTypes.PROMPT_UPDATE,
            #                 func=self.name,
            #                 content=chunk.content,
            #                 field=curr_field
            #             )
            #             if self.pydantic_model:
            #                 output, curr_field, curr_content = auto_split_completion(
            #                     curr_content=curr_content, 
            #                     chunk=chunk.content, 
            #                     output=output, 
            #                     curr_field=curr_field, 
            #                     pydantic_model=self.pydantic_model
            #                 ) 
            #         else:
            #             output[curr_field] = sanitize_content(curr_content)
            #             content = chunk.content                   
            # except Exception as e:
            #     prompt_run.end(errors={"error": str(e)})
            #     yield ChatChunk(
            #         msg_type=PromptChunkTypes.PROMPT_ERROR,
            #         func=self.name,
            #         content=str(e),
            #     )
            #     return
            content = chunk.content
            output = chunk.output
            if self.pydantic_model:
                try:
                    # if is_list_model(self.pydantic_model):
                    #     list_model = unpack_list_model(self.pydantic_model)
                    #     if not self.delimiter:
                    #         raise ValueError("delimiter must be provided for list models")
                    #     content = parse_model_list(output, list_model, self.delimiter)
                    # else:
                    output = self.pydantic_model(**output)
                except Exception as e:
                    prompt_run.end(errors={"error": output})
                    yield ChatResponse(
                        value=output,
                        run_id=str(prompt_run.id),
                        conversation=system_conversation.conversation.copy(),
                        examples=examples,
                        error=PromptError(error=str(e), error_type=PromptErrorTypes.PYDANTIC_ERROR),
                    )

            kwargs['run_id'] = str(prompt_run.id)
            parsed_output = await self._call_parser(output or content, context={'kwargs': kwargs})        
            
            response_message = AIMessage(content=content)

            system_conversation.append(response_message)

            # prompt_run.end(outputs={"messages": [response_message], 'output': parsed_output})
            prompt_run.end(outputs={'output': parsed_output})

            yield ChatResponse(
                value=parsed_output,
                run_id=str(prompt_run.id),
                conversation=system_conversation.conversation.copy(),
                examples=examples
            )


    async def _call_llm_stream(self, openai_messages, func, metadata=None, pydantic_model=None, tracer_run=None, **kwargs):
        output = to_dict(pydantic_model) if pydantic_model else None
            
        curr_field = None
        curr_content = ""
        yield ChatChunk(
            msg_type=PromptChunkTypes.PROMPT_START,
            func=func,
            content=None,
            field=curr_field
        )   
        async for chunk in self.llm.send_stream(
            openai_messages=openai_messages, 
            tracer_run=tracer_run, 
            metadata=metadata,            
            **kwargs
        ):
            
            if chunk.finish is False:
                yield ChatChunk(
                    msg_type=PromptChunkTypes.PROMPT_UPDATE,
                    func=func,
                    content=chunk.content,
                    field=curr_field
                )
                if pydantic_model:
                    output, curr_field, curr_content = auto_split_completion(
                        curr_content=curr_content, 
                        chunk=chunk.content, 
                        output=output, 
                        curr_field=curr_field, 
                        pydantic_model=pydantic_model
                    ) 
            else:
                if output:
                    output[curr_field] = sanitize_content(curr_content)
                content = chunk.content                   

        yield ChatChunk(
            msg_type=PromptChunkTypes.PROMPT_FINISH,
            func=func,
            content=content,
            output=output
        )


    async def _call_action_llm_stream(self, openai_messages, func, metadata=None, pydantic_model=None, tracer_run=None, **kwargs):
        output = None            
        curr_field = None
        curr_content = ""
        output_list = []
        yield ChatChunk(
            msg_type=PromptChunkTypes.PROMPT_START,
            func=func,
            content=None,
            field=curr_field
        )   
        async for chunk in self.llm.send_stream(
            openai_messages=openai_messages, 
            tracer_run=tracer_run, 
            metadata=metadata,            
            **kwargs
        ):
            
            if chunk.finish is False:
                yield ChatChunk(
                    msg_type=PromptChunkTypes.PROMPT_UPDATE,
                    func=func,
                    content=chunk.content,
                    field=curr_field
                )
                if pydantic_model:                    
                    output, curr_field, curr_content, is_new_output = auto_split_list_completion(
                        curr_content=curr_content, 
                        chunk=chunk.content, 
                        output_list=output_list, 
                        curr_field=curr_field,
                        pydantic_model=pydantic_model
                    )
                    if is_new_output:
                        yield ChatChunk(
                            msg_type=PromptChunkTypes.PROMPT_MILESTONE,
                            func=func,
                            content=pydantic_model(**output_list[-1]),
                        )

            else:
                if output:
                    output[curr_field] = sanitize_content(curr_content)
                content = chunk.content                   

        yield ChatChunk(
            msg_type=PromptChunkTypes.PROMPT_FINISH,
            func=func,
            content=content,
            output=output
        )

    
    async def to_openai(self, conversation, examples=None, max_memory_length=None):
        if max_memory_length:
            conversation = conversation.trim_memory(max_memory_length)
        if examples:
            example_openai_messages = [m.to_openai() for example in examples for m in example.messages]
            if conversation.messages[0].role == "system":
                return [conversation.messages[0].to_openai()] + example_openai_messages + [m.to_openai() for m in conversation.messages[1:]]
            else:
                return example_openai_messages + [m.to_openai() for m in conversation.messages]
        else:
            return [m.to_openai() for m in conversation.messages]



    async def parse_completion(self, output, **kwargs):
        if self.pydantic_model:
            content = parse_completion(output.content, self.pydantic_model)
        else:
            content = output.content    
        content = await self._call_parser(content, context={'kwargs': kwargs})        
        return content
        
        
    async def _call_parser(self, content, context):
        signature = inspect.signature(self.parser)
        parameters = signature.parameters
        argument_names = [param for param in parameters if parameters[param].default == inspect.Parameter.empty]
        if 'context' in argument_names:
            return await self.parser(content, context=context)
        else:
            return await self.parser(content)
        

    async def with_tools(self, prompt=None, conversation=None, tools=[], tool_choice=None, tracer_run=None, **kwargs):
        system_conversation = await self.build_conversation(prompt, conversation=conversation,  **kwargs)
        # prompt_run = self.get_tracer(kwargs, prompt_metadata)

        log_kwargs = {}
        log_kwargs.update(kwargs)
        extra = self.get_extra(**kwargs)
        if prompt is not None:
            log_kwargs['prompt'] = prompt
        # prompt_run = RunTree(
        #     name=self.name,
        #     # run_type="prompt",
    #     run_type="tool",
        #     inputs=log_kwargs,
        #     # extra=prompt_metadata,
        # )
        with Tracer(
            is_traceable=self.is_traceable,
            tracer_run=tracer_run,
            name=self.name,
            run_type="tool",
            inputs=log_kwargs,
            extra=extra
        ) as prompt_run:


            openai_completion, tool_calls = await self.call_llm_with_tools(
                system_conversation=system_conversation, 
                tools=tools, 
                tool_choice=tool_choice, 
                tracer_run=prompt_run, 
                **kwargs)
            output = openai_completion.choices[0].message

            # output, tool_calls = await self.llm.send_with_tools(
            #     conversation=conversation, 
            #     tools=tools, 
            #     tool_choice=tool_choice, 
            #     tracer_run=prompt_run,
            #     **kwargs
            # )

            prompt_run.end(outputs={"function": output})

            tool_str_calls = [str(tc.function) for tc in output.tool_calls] if output.tool_calls is not None else []
            content = ''
            if len(tool_str_calls) > 1:
                content = str(tool_str_calls)
            elif len(tool_str_calls) == 1:
                content=tool_str_calls[0]

            system_conversation.append(AIMessage(content=content))

            return ChatResponse(
                value=content,
                run_id=str(prompt_run.id),
                conversation=system_conversation.conversation.copy(),
                tools=tool_calls
            )
        
            

    async def call_llm_with_tools(self, system_conversation, tools, tool_choice, tracer_run, **kwargs):
        tool_lookup = {tool.__name__: tool for tool in tools}
        if tool_choice and tool_choice in tool_lookup:
            tool_choice = convert_to_openai_tool(tool_lookup[tool_choice])

        openai_messages = system_conversation.to_openai()  
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        
        openai_completion = await self.llm.send_with_tools(
            openai_messages=openai_messages, 
            openai_tools=openai_tools, 
            tool_choice=tool_choice, 
            tracer_run=tracer_run,
            metadata=system_conversation.get_metadata(),
            **kwargs
        )

        tool_calls = []

        output = openai_completion.choices[0].message
        
        if output.tool_calls is not None:
            for tool_call in output.tool_calls:
                tool_instance = tool_lookup[tool_call.function.name](**json.loads(tool_call.function.arguments))
                tool_calls.append(tool_instance)                
        return openai_completion, tool_calls



    async def retrieve_rag_async(self, kwargs):
        examples_res = await self.rag_manager.search(
            query=kwargs[self.rag_query_key],
            top_k=self.num_examples, 
            namespace=self.rag_index
        )        
        examples = [r.metadata for r in examples_res]
        return examples
    

    
    



class prompt:

    def __init__(
            self,           
            *arg,
            **kwargs
        ):
        self.args = arg
        self.kwargs = kwargs

    def __call__(self, func):
        decorator_instance = self.PromptDecoratorInstance(func, *self.args, **self.kwargs)
        return decorator_instance
        
    
    class PromptDecoratorInstance:

        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs
            if not kwargs.get('name'):
                kwargs['name'] = func.__name__
            self.prompt = ChatPrompt(
                *args, 
                **kwargs
            )
            self.prompt.parser = func

        def __call__(self, *args, **kwargs):
            return self.prompt(*args, **kwargs)
        
        def stream(self, *args, **kwargs):
            return self.prompt.call_stream(*args, **kwargs)
        

        def with_tools(self, *args, **kwargs):
            return self.prompt.with_tools(*args, **kwargs)