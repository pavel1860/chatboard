import copy
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Type, TypedDict, Unpack
from pydantic import BaseModel, Field

from ..block import BlockChunk, Block, BlockList
from ..prompt.flow_components import StreamController



ToolChoice = Literal['auto', 'required', 'none']

class LlmConfig(BaseModel):
    model: str | None = Field(default=None, description="The model to use")
    temperature: float = Field(default=0, ge=0, le=1, description="The temperature of the response")
    max_tokens: int | None = Field(default=None, ge=0, description="The maximum number of tokens in the response")
    stop_sequences: List[str] | None = Field(default=None, description="The stop sequences of the response")
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



    

class LLMStream(StreamController):
    blocks: BlockList
    llm_config: LlmConfig
    tools: List[Type[BaseModel]] | None = None
    model: str | None = None
    models: List[str] = []
    
    
    def __init__(self, blocks: BlockList, config: LlmConfig, tools: List[Type[BaseModel]] | None = None, model: str | None = None):
        super().__init__(self.stream, acc_factory=lambda : BlockList([], style="stream"), name="openai_llm", span_type="llm")
        self.blocks = blocks
        self.llm_config = config
        self.tools = tools
        self.model = model
        


class LlmStreamParams(TypedDict, total=False):
    blocks: BlockList
    config: LlmConfig
    tools: List[Type[BaseModel]] | None = None




# def llm_stream(
#     method: Callable[..., AsyncGenerator[Any, None]],
# ) -> Callable[..., StreamController]:
#     """
#     Decorator that wraps an async generator method to return a StreamController.
#     Provides proper typing for IntelliSense support.
#     """
#     @wraps(method)
#     def wrapper(self, *args, **kwargs) -> StreamController:
#         # If "config" not passed, inject from self.config
#         if "config" not in kwargs:
#             kwargs["config"] = getattr(self, "config", None)
#         gen = method(self, *args, **kwargs)
#         return StreamController(gen=gen, name=name or method.__name__, span_type="llm")
#     return wrapper


def pack_blocks(args: tuple[Any, ...]) -> tuple[BlockList, tuple[Any, ...]]:
    # block_list = BlockList()
    block_list = []
    extra_args = ()
    for arg in args:
        if isinstance(arg, str):
            block_list.append(Block(arg))
        elif isinstance(arg, Block):
            block_list.append(copy.copy(arg))
        elif isinstance(arg, BlockList):
            block_list.extend(copy.copy(arg))
        else:
            extra_args += (arg,)
    return block_list, extra_args
    


def llm_stream(
    name: str,
):
    def llm_stream_decorator(
        method: Callable[..., AsyncGenerator[Any, None]],
    ) -> Callable[..., StreamController]:
        """
        Decorator that wraps an async generator method to return a StreamController.
        Provides proper typing for IntelliSense support.
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs) -> StreamController:
            # If "config" not passed, inject from self.config
            if "config" not in kwargs or kwargs["config"] is None:
                kwargs["config"] = getattr(self, "config", None)
            blocks, extra_args = pack_blocks(args)
            # gen = method(self, blocks, *extra_args, **kwargs)
            # return StreamController(gen=gen, name=name or method.__name__, span_type="llm")
            return StreamController(gen_func=method, args=(self, blocks, *extra_args), kwargs=kwargs, name=name or method.__name__, span_type="llm")
        return wrapper
    return llm_stream_decorator
    
 
class BaseLLM: 
    config: LlmConfig   
    default_model: str
    models: List[str]
    
    
    def __init__(self, config: LlmConfig | None = None):        
        self.config = config or LlmConfig(model=self.default_model )
    

    @llm_stream(name="llm_call")
    async def stream(
        self,
        *blocks: BlockChunk | Block | str,
        config: LlmConfig,
        tools: List[Type[BaseModel]] | None = None
    ) -> AsyncGenerator[Any, None]:
        """
        This method is used to stream the response from the LLM.
        After decoration, this will return a StreamController instance.
        """
        pass
        yield
        
    



    
    
class LLM():
    
    _model_registry: Dict[str, Type[BaseLLM]] = {}
    _default_model: str | None = None
        
    
    
    @classmethod
    def register(cls, model_cls: Type[BaseLLM], default_model: str | None = None) -> Type[BaseLLM]:
        """Decorator to register a new LLM model implementation"""
        if model_cls.__name__ in cls._model_registry:
            raise ValueError(f"Model {model_cls.__name__} is already registered")
        for model in model_cls.models:
            cls._model_registry[model] = model_cls
        if default_model:
            cls._default_model = default_model
        return model_cls
    
    @classmethod
    def _get_llm(cls, model: str | None = None) -> Type[BaseLLM]:
        """Get a registered model by name"""
        if model is None:
            if cls._default_model is None:
                raise ValueError("No default model is set")
            model = cls._default_model
        if model not in cls._model_registry:
            raise KeyError(f"Model {model} is not registered")        
        llm_cls = cls._model_registry[model]
        return llm_cls
    
    
    def stream(
        self,
        *blocks: BlockChunk | Block | str,
        model: str | None = None,
        tools: List[Type[BaseModel]] | None = None,
        config: LlmConfig | None = None,
    ) -> StreamController:
        llm_cls = self._get_llm(model)
        llm = llm_cls(config=config)
        
        return llm.stream(*blocks, tools=tools, config=config)

    def __call__(
        self,        
        *blocks: BlockChunk | Block |BlockList | str,
        model: str | None = None,
        config: LlmConfig | None = None,
    ) -> LLMStream:                                
        
        
        llm_blocks, extra_args = pack_blocks(blocks)

        
        llm_ctx = self._get_llm(model)        
        return llm_ctx(llm_blocks, *extra_args, config=config)
    
    # def __call__(
    #     self,        
    #     blocks: BlockContext |BlockList | Block | BlockPrompt | str,
    #     model: str | None = None,
    #     config: LlmConfig | None = None,
    # ) -> LLMStream:                        
    #     if isinstance(blocks, str):
    #         llm_blocks = BlockList([Block(blocks)])
    #     elif isinstance(blocks, Block):
    #         llm_blocks = BlockList([blocks])
    #     elif isinstance(blocks, BlockPrompt):
    #         llm_blocks = BlockList([blocks.root])
    #     elif isinstance(blocks, BlockList):
    #         llm_blocks = blocks        
    #     else:
    #         raise ValueError(f"Invalid blocks type: {type(blocks)}")
        
    #     llm_ctx = self._get_llm(model)
    #     config = config or LlmConfig(model=llm_ctx.model)
    #     return llm_ctx(llm_blocks, config)
