from __future__ import annotations
from abc import abstractmethod
from enum import StrEnum
from functools import singledispatch
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Literal, ParamSpec, Self, Type, TypeVar, Union, TYPE_CHECKING, get_args
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pydantic.fields import FieldInfo
from ..block.util import StreamEvent
from ..llms.types import ErrorMessage
from ..block import BlockChunk, BlockRole, ToolCall, LlmUsage, BlockList
from ..tracer import Tracer
from ..parsers import XmlOutputParser
from ..utils import logger
from ..utils.function_utils import call_function
from ..utils.model_utils import get_list_type, is_list_type, is_optional_type, schema_to_ts  
import xml.etree.ElementTree as ET

class LLMToolNotFound(Exception):
    
    def __init__(self, tool_name) -> None:
        self.tool_name = tool_name
        super().__init__(f"Action {tool_name} is not found")


ToolChoice = Literal['auto', 'required', 'none']

LLM_CLIENT = TypeVar('LLM_CLIENT')
CLIENT_PARAMS = ParamSpec('CLIENT_PARAMS')
CLIENT_RESPONSE = TypeVar('CLIENT_RESPONSE')



ToolReprFormat = Literal['json', 'function', 'xml']

class BaseLlmClient(BaseModel, Generic[LLM_CLIENT, CLIENT_RESPONSE]):
    client: LLM_CLIENT
    
    @abstractmethod
    async def complete(
        self, 
        messages: List[dict], 
        tools: List[dict], 
        model: str, 
        tool_choice: str, 
        **kwargs
    ) -> CLIENT_RESPONSE:
        ...


class LlmConfig(BaseModel):
    model: str | None = Field(default=None, description="The model to use")
    temperature: float = Field(default=0, ge=0, le=1, description="The temperature of the response")
    max_tokens: int = Field(default=None, ge=0, description="The maximum number of tokens in the response")
    stop_sequences: List[str] = Field(default=None, description="The stop sequences of the response")
    stream: bool = Field(default=False, description="If the response should be streamed")
    logit_bias: Dict[str, int] | None = Field(default=None, description="The logit bias of the response")
    top_p: float = Field(default=1, ge=0, le=1, description="The top p of the response")
    presence_penalty: float | None = Field(default=None, ge=-2, le=2, description="The presence penalty of the response")
    logprobs: bool | None = Field(default=None, description="If the logprobs should be returned")
    seed: int | None = Field(default=None, description="The seed of the response")
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2, description="The frequency penalty of the response")
    tools: List[Type[BaseModel]] | None = Field(default=None, description="The tools of the response")
    retries: int = Field(default=3, ge=1, description="The number of retries")
    parallel_tool_calls: bool = Field(default=False, description="If the tool calls should be parallel")
    tool_choice: ToolChoice | None = Field(default=None, description="The tool choice of the response")
    
    
    # def get_llm_args(self, **kwargs):
    #     model_kwargs={}
    #     stop_sequences = kwargs.get("stop_sequences", self.stop_sequences)
    #     if stop_sequences:
    #         model_kwargs['stop'] = stop_sequences
    #     logit_bias = kwargs.get("logit_bias", self.logit_bias)
    #     if logit_bias:            
    #         model_kwargs['logit_bias'] = logit_bias
    #     top_p = kwargs.get("top_p", self.top_p)
    #     if top_p:
    #         if top_p > 1.0 or top_p < 0.0:
    #             raise ValueError("top_p must be between 0.0 and 1.0")
    #         model_kwargs['top_p'] = top_p
    #     presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
    #     if presence_penalty:
    #         if presence_penalty > 2.0 or presence_penalty < -2.0:
    #             raise ValueError("presence_penalty must be between -2.0 and 2.0")
    #         model_kwargs['presence_penalty'] = presence_penalty
    #     frequency_penalty = kwargs.get("frequency_penalty", self.frequency_penalty)
    #     if frequency_penalty:
    #         if frequency_penalty > 2.0 or frequency_penalty < -2.0:
    #             raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    #         model_kwargs['frequency_penalty'] = frequency_penalty
    #     temperature = kwargs.get("temperature", self.temperature)
    #     if temperature is not None:
    #         model_kwargs['temperature'] = temperature
    #     max_tokens = kwargs.get("max_tokens", self.max_tokens)
    #     if max_tokens is not None:
    #         model_kwargs['max_tokens'] = max_tokens

    #     # model = kwargs.get("model", self.model)
    #     # if model is not None:
    #     #     model_kwargs['model'] = model
        
    #     seed = kwargs.get("seed", self.seed)
    #     if seed is not None:
    #         model_kwargs['seed'] = seed
            
    #     # parallel_tool_calls = kwargs.get("parallel_tool_calls", self.parallel_tool_calls)
    #     # if parallel_tool_calls is not None and actions:
    #     #     model_kwargs['parallel_tool_calls'] = parallel_tool_calls
            
    #     logprobs = kwargs.get("logprobs", self.logprobs)
    #     if logprobs is not None:
    #         model_kwargs['logprobs'] = logprobs        
    #     return model_kwargs


        
def get_field_type(field_info) -> str:
    if get_args(field_info.annotation):
        return str(field_info.annotation)
    return field_info.annotation.__name__


class OutputModel(BaseModel):    
    tool_calls: List[ToolCall] = Field(default=[], json_schema_extra={"render": False})
    _block: BlockChunk = PrivateAttr()
    
    
    
    @classmethod
    def user_suffix(cls) -> str:
        return ""
    
    @classmethod
    def response_prefix(cls) -> str:
        return ""
    
    @classmethod
    def extra_rules(cls) -> BlockChunk | None:
        return None
    
    def output_fields(self) -> List[tuple[str, FieldInfo]]:
        return [(field, field_info) for field, field_info in self.model_fields.items() if field != "tool_calls"]
    
    @classmethod
    def iter_fields(cls):
        for field, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if extra and extra.get("render") == False:
                continue
            yield field, field_info
            

    @classmethod
    def render(cls, tools: List[Type[BaseModel]] = [], config: LlmConfig | None = None) -> BlockChunk:
        with BlockChunk("Output format", tags=["output_format"]) as blk:
            blk += "you **MUST** use the following format for your output:"
            for field, field_info in cls.iter_fields():
                render_func = getattr(cls, f"render_{field}", None)                
                if render_func:
                    blk /= render_func()
                else:
                    with blk(field, style=["xml"]):
                        if field_info.description:
                            blk += field_info.description
            if tools:
                
                # elif config and config.tool_choice == "none":
                    # blk += "you should not use any tool calls"
                blk /= "the tool calls you want to use should be in the following xml format:"
                # with blk("tool_calls", tags=["tool_calls"], style=["xml"]):
                with blk("tool", tags=["tool_example"], attrs={"name": "{{the tool name}}"}, style=["xml"]):
                    blk /= "{{the tool arguments in json format}}"
                with blk("tool", tags=["tool_example"], attrs={"name": "{{the second tool name}}"}, style=["xml"]):
                    blk /= "{{the second tool arguments in json format}}"
                blk /= "<!-- (... more tool calls if needed) -->"
                    
                    
                with blk("Tool Call Rules", tags=["tool_call_rules"], style=["list"]):
                    blk /= "read carefully the tool description and the tool arguments."
                    blk /= "understand if you have a tool that can help you to complete the task or you should respond only in text."
                    blk /= "you can perform actions in response to the user input."
                    blk /= "tool calls should be in xml format with json schema inside as a child element."
                    blk /= "make sure the order of the xml tags is correct."
                    blk /= "IMPORTANT! you must use this output format!"
                    if config and config.tool_choice == "required":
                        blk /= "you have to pick the right tool to respond to the user input"
                    elif config and config.tool_choice == "auto":
                        blk /= "you can use as many tools as needed to complete the task"
                if extra_rules_block:= cls.extra_rules():
                    blk /= extra_rules_block
                blk /= "you must use the provided output format."
                blk /= "** end of output format **"
                # blk /= "don't forget to output a message when you have something to say."
                blk /= ""
                blk /= ""
                with blk("Available tools", tags=["available_tools"], style=["xml"]):
                    blk /= "you can use the following tools to complete the task:"
                    for tool in tools:
                        if not tool.__doc__:
                            raise ValueError(f"Tool {tool.__name__} has no description")
                        with blk("tool", attrs={"name": tool.__name__, "description": tool.__doc__}, tags=["tool"], style=["xml"]):                            
                            # blk /= "description:" + tool.__doc__
                            blk.model_schema(tool)
        return blk
    
    def block(self) -> BlockChunk:
        self._block.tool_calls = self.tool_calls
        return self._block
    
    
    def find_tool(self, tool_name: str) -> ToolCall | None:
        for tool in self.tool_calls:
            if tool.name == tool_name:
                return tool
        return None

    def find_except(self, tool_name: str) -> List[ToolCall]:
        tools = []
        for tool in self.tool_calls:
            if tool.name == tool_name:
                continue
            tools.append(tool)
        return tools

    # @classmethod
    # def parse(cls, completion: Block, tools: List[Type[BaseModel]]) -> Self:
    #     """parse the completion into the output model"""
    #     try:
    #         xml_parser = XmlOutputParser()
    #         fmt_res, fmt_tools = xml_parser.parse(f"<root>{cls.response_prefix()}{completion.content}</root>", tools, cls)
    #         fmt_res.tool_calls = fmt_tools
    #         fmt_res._block = completion
    #         return fmt_res
    #     except ValidationError as e:
    #         logger.exception("Output Model Validation Error")
    #         raise ErrorMessage(f"Validation error: {e}")
    
    @classmethod
    def parse_tool_calls(cls, item: ET.Element, tools: List[Type[BaseModel]]) -> List[ToolCall]:
        from ..block import ToolCall
        from uuid import uuid4
        tool_calls = []
        tool_lookup = {tool.__name__: tool for tool in tools}        
        for tool in item.findall("tool"):
            tool_cls = tool_lookup.get(tool.attrib["name"], None)
            if not tool_cls:
                from . import ErrorMessage
                raise ErrorMessage(tool.attrib["name"])
            # params = {param.attrib["name"]: param.text for param in tool.findall(param_tag)}
            tool_inst = tool_cls.model_validate_json(tool.text)
            tool_calls.append(
                ToolCall(
                    id=f"tool_call_{uuid4()}"[:40], 
                    name=tool.attrib["name"], 
                    tool=tool_inst
                )
            )
        return tool_calls
    
    @classmethod
    def parse(cls, completion: BlockChunk, tools: List[Type[BaseModel]]) -> Self:
        """parse the completion into the output model"""
        try:
            root = ET.fromstring(f"<root>{cls.response_prefix()}{completion.content}</root>")
            params = {}
            for field, field_info in cls.model_fields.items():
                item = root.find(field)                 
                if field == "tool_calls":
                    params[field] = cls.parse_tool_calls(root, tools)
                    continue
                              
                if item is None:
                    continue
                elif item.text is None:
                    raise ErrorMessage(f"Field {field} has no text")
                else:
                    parse_func = getattr(cls, f"parse_{field}", None)
                    if parse_func:
                        params[field] = parse_func(item)
                    else:
                        params[field] = item.text.strip()             
            obj = cls(**params)
            obj._block = completion
            return obj
        except ValidationError as e:
            logger.exception("Output Model Validation Error")
            raise ErrorMessage(f"Validation error: {e}")
    
    
    
    def __repr__(self) -> str:
        s = f"<{self.__class__.__name__}>"
        for field, field_info in self.model_fields.items():
            s += f"\n{field}: {getattr(self, field)}"
        # for tool in self.tool_calls:
        #     s += f"\n{tool}"
        s += f"\n</{self.__class__.__name__}>"
        return s
 
 
OUTPUT_MODEL = TypeVar('OUTPUT_MODEL', bound=OutputModel)

class LlmContext(Generic[OUTPUT_MODEL]):
    models: list[str] = []  
    
    def __init__(self, model: str):
        if model not in self.models:
            raise ValueError(f"Model {model} is not supported by {self.__class__.__name__}")
        self.config: LlmConfig = LlmConfig(model=model)
        self.blocks: BlockList = BlockList()
        self.tools: List[Type[BaseModel]] = []
        self.output_model: Type[OutputModel] | None = None
        self.is_traceable = True
        self.tracer_run: Tracer | None = None
        self.parse_response_fn: Callable[[BlockChunk], BlockChunk] | None = None
        self.parse_output_fn: Callable[[OUTPUT_MODEL], OUTPUT_MODEL] | None = None
        
        
    def __call__(self, blocks: BlockList) -> Self:
        self.blocks = blocks
        return self
    
    def __await__(self):                
        return self.run_controller_block().__await__()
    
    def set_blocks(self, blocks: BlockList):
        self.blocks = blocks
        return self
    
    def set_model(self, model: str):
        self.config.model = model
        return self

    @abstractmethod
    def to_chat(
        self, 
        blocks: BlockList, 
        tools: List[Type[BaseModel]] | None = None, 
        error_blocks: BlockList | None = None
    ) -> List[dict]:
        ...
        
    @abstractmethod
    async def client_complete(
        self, 
        blocks: BlockList, 
        tools: List[Type[BaseModel]] | None = None, 
        config: LlmConfig | None = None, 
        error_blocks: List[BlockChunk] | None = None
    ) -> BlockChunk:
        ...
    
    def pick(
        self, 
        tools: List[Type[BaseModel]], 
        tool_choice: ToolChoice="required", 
        parallel_tool_calls: bool=False
    ):
        self.config.tool_choice = tool_choice
        self.config.parallel_tool_calls = parallel_tool_calls
        self.tools = tools
        return self
    
    def pick_one(self, tools: List[Type[BaseModel]]):
        self.config.tool_choice = "required"
        self.config.parallel_tool_calls = False
        self.tools = tools
        return self
    
    def pick_many(self, tools: List[Type[BaseModel]]):
        self.config.tool_choice = "required"
        self.config.parallel_tool_calls = True
        self.tools = tools
        return self
    
    def generate_or_pick_one(self, tools: List[Type[BaseModel]]):
        self.config.tool_choice = "auto"
        self.config.parallel_tool_calls = False
        self.tools = tools
        return self
    
    def generate_or_pick_many(self, tools: List[Type[BaseModel]]):
        self.config.tool_choice = "auto"
        self.config.parallel_tool_calls = True
        self.tools = tools
        return self
    
    def parse_response(self, parse_response_fn: Callable[[BlockChunk], BlockChunk]) -> Self:
        self.parse_response_fn = parse_response_fn
        return self
    
    def parse_output(self, parse_output_fn: Callable[[OUTPUT_MODEL], OUTPUT_MODEL]) -> Self:
        self.parse_output_fn = parse_output_fn
        return self
    
    # @singledispatch
    # async def complete(self, output_model):
    #     raise NotImplementedError("Output model is not set")
    
    # @complete.register(type(None))
    # async def _(self):
    #     return await self.run_controller_block()
        
    # @complete.register    
    # async def _(self, output_model: Type[OutputModel]):
    #     self.output_model = output_model
    #     response = await self.run_controller_output_model()
    #     return response
    
    async def complete(self, output_model: Type[OUTPUT_MODEL] | None = None) -> OUTPUT_MODEL:
        if output_model is None:
            return await self.run_controller_output_model()
        self.output_model = output_model
        response = await self.run_controller_output_model()
        return response
    
    async def run_controller_output_model(self) -> OUTPUT_MODEL:
        blocks, output_model = await self.controller(self.blocks, self.tools, self.config)
        if output_model is None:
            raise ValueError("Output model is not set")
        return output_model
    
    async def run_controller_block(self) -> BlockChunk:
        blocks, output_model = await self.controller(self.blocks, self.tools, self.config)
        return blocks
    
    
    async def stream(self) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplementedError("Streaming is not implemented")
    
    
    
    async def controller(
        self, 
        blocks: BlockList,
        tools: List[Type[BaseModel]],
        config: LlmConfig    
    ) -> tuple[BlockChunk, OUTPUT_MODEL | None]:        
        response = None
        error_blocks = BlockList()
        for attempt in range(config.retries):
            try:
                response = await self.client_complete(
                    blocks=blocks,
                    tools=tools,
                    config=config,
                    error_blocks=error_blocks
                )
                if self.parse_response_fn:
                    response = await call_function(self.parse_response_fn, response)
                if self.output_model is not None:
                    parsed_response = self.output_model.parse(response, tools)
                    if self.parse_output_fn:
                        parsed_response = await call_function(self.parse_output_fn, parsed_response)
                    return response, parsed_response
                return response, None
            except ErrorMessage as e:
                
                if attempt == config.retries - 1:
                    logger.warning(f"Error Message: {e.error_content}\n too many attempts, raising error.")
                    raise
                if e.should_retry:
                    logger.warning(f"Error Message: {e.error_content}\n attempt {attempt + 1} of {config.retries}, retrying...")
                    if response is not None:
                        error_blocks.append(response)
                    error_blocks.append(e.to_block(self.output_model, role="user", tags=["generation", "error"]))
        raise Exception("Failed to complete")
    
    
    async def run_complete2(
        self, 
        retries=3,    
    ) -> BlockChunk:

        
        
        messages = [self._to_message(b) for b in self._to_chat(self.ctx_blocks)]
        for msg in messages:
            if not msg.get("role"):
                raise ValueError("role is not set")
        for message in messages:
            print(f"----------------------{message['role']}-------------------------")
            print(message['content'])            
        print(f"-----------------------------------------------------------------")
        tools = self._to_tools(self.actions)
        response = None
        for attempt in range(retries):
            try:
                response = await self._complete(
                    messages=messages,
                    tools=tools,
                    model=self.model,
                    tool_choice=self.tool_choice,
                    **self.config.get_llm_args()
                )
                parsed_response = self._parse_response(response, self.actions)
                print(f"----------------------assistant response-------------------------")
                print(parsed_response)
                if self.output_model is not None:
                    parsed_response = self.output_model.parse(parsed_response)
                return parsed_response
            except ErrorMessage as e:
                if attempt == retries - 1:
                    raise
                if e.should_retry:
                    if response:
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append(e.to_block())
        raise Exception("Failed to complete")





class LLM():
    
    _model_registry: Dict[str, Type[LlmContext]] = {}
    _default_model: str | None = None
    
    
    @classmethod
    def register(cls, model_cls: Type[LlmContext], default_model: str | None = None) -> Type[LlmContext]:
        """Decorator to register a new LLM model implementation"""
        if model_cls.__name__ in cls._model_registry:
            raise ValueError(f"Model {model_cls.__name__} is already registered")
        for model in model_cls.models:
            cls._model_registry[model] = model_cls
        if default_model:
            cls._default_model = default_model
        return model_cls
    
    @classmethod
    def _get_llm(cls, model: str | None = None) -> LlmContext:
        """Get a registered model by name"""
        if model is None:
            if cls._default_model is None:
                raise ValueError("No default model is set")
            model = cls._default_model
        if model not in cls._model_registry:
            raise KeyError(f"Model {model} is not registered")        
        llm_cls = cls._model_registry[model]
        llm = llm_cls(model)
        return llm

    
    def __call__(
        self,        
        # *args: Block | List[Block] | BlockList | str,
        *args: BlockChunk | str,
        model: str | None = None,
    ) -> LlmContext:

        # with Block() as ctx_blocks:
        #     for block in args:
        #         if isinstance(block, str):
        #             ctx_blocks.append(block, role="user", tags=["user_input"])
        #         elif isinstance(block, Block):
        #             if not block.role:
        #                 raise ValueError("Block role is not set")
        #             ctx_blocks.append(block)
        #         elif isinstance(block, BlockList):
        #             for b in block:
        #                 ctx_blocks.append(b)
        ctx_blocks = BlockList()
        for block in args:
            if isinstance(block, str):
                ctx_blocks.append(block, role="user", tags=["user_input"])
            elif isinstance(block, BlockChunk):
                ctx_blocks.append(block)
            else:
                raise ValueError("Invalid block type")
                
        
                        
        llm_ctx = self._get_llm(model)
        return llm_ctx(ctx_blocks)
