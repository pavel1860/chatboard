import copy
import os
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Type, TypedDict, Unpack, TYPE_CHECKING
from pydantic import BaseModel, Field

from ..block import BlockChunk, Block, BlockList, BlockSchema
# from ..prompt.flow_components import StreamController
from ..prompt.fbp_process import StreamController, Stream, compute_stream_cache_key, Accumulator, Process
from dataclasses import dataclass
from .types import LlmConfig, LLMResponse, LLMUsage
if TYPE_CHECKING:
    from ..prompt import Context



class LLMStream(Stream):
    response: LLMResponse | None = None
    usage: LLMUsage | None = None
    
    def __init__(self, gen: AsyncGenerator, name: str = "stream", response: LLMResponse | None = None, usage: LLMUsage | None = None):
        super().__init__(gen, name=name)
        self.response = response
        self.usage = usage
    
    async def __anext__(self):
        """
        Receive next IP from wrapped generator.

        If persistence is enabled, collects IP for later save.
        Only saves to JSONL on successful stream completion (not on errors).
        """
        from ..block import BlockChunk
        for i in range(10):
            # ip = await super().__anext__()
            ip = await Process.__anext__(self)
            
            if isinstance(ip, LLMResponse):
                self.response = ip
                continue
            elif isinstance(ip, LLMUsage):
                self.usage = ip
                continue
            # Collect for save if persistence enabled
            if self._save_stream_path is not None:
                if hasattr(ip, "model_dump"):
                    data = ip.model_dump()
                else:
                    data = ip
                self._collected_chunks.append(data)

            return ip
        else:
            raise ValueError("more than 10 tries to get chunks")
        
    @classmethod    
    async def load_llm_call(cls, llm_call_id: int, delay: float = 0.05):
        from ..versioning.dataflow_models import LlmCall
        import asyncio
        llm_call = await LlmCall.get_or_none(id=llm_call_id)
        if llm_call is None:
            raise ValueError(f"LLM call {llm_call_id} not found")
        
        response = LLMResponse(id=llm_call.request_id, item_id=llm_call.message_id)
        usage = llm_call.usage
        
        async def load_llm_call_stream(llm_call: LlmCall):
            from ..block import BlockChunk
            for chunk in llm_call.chunks:
                if delay > 0:
                    await asyncio.sleep(delay)
                yield chunk
        # self.upstream = load_llm_call_stream(llm_call)
        return cls(load_llm_call_stream(llm_call), name=f"{cls.__name__}_stream", response=response, usage=usage)
        
    

class LLMStreamController(StreamController):
    llm: "LLM"
    blocks: BlockList
    llm_config: LlmConfig
    tools: List[Type[BaseModel]] | None = None
    model: str | None = None
    models: List[str] = []
    
    
    def __init__(
        self, 
        # gen_func: Callable[..., AsyncGenerator], 
        llm: "LLM",
        blocks: BlockList, 
        config: LlmConfig, 
        tools: List[Type[BaseModel]] | None = None, 
        args: tuple = (), 
        kwargs: dict = {}
    ):
        # super().__init__(self.stream, acc_factory=lambda : BlockList([], style="stream"), name="openai_llm", span_type="llm")
        super().__init__(gen_func=None, args=args, kwargs=kwargs, name="openai_llm", span_type="llm")
        self.blocks = blocks
        self.llm_config = config
        self.tools = tools
        self.model = config.model
        self.llm = llm
        
    @property
    def inputs(self) -> list[Any]:
        """Get the inputs."""
        return self.blocks

    def _init_stream(self, args: tuple, kwargs: dict):
        gen_instance = self._gen_func(*args, **kwargs)
        stream = LLMStream(gen_instance, name=f"{self._name}_stream")
        return stream
    
    
    
    # async def _load_llm_call(self):
    #     from ..versioning.dataflow_models import LlmCall
    #     if self.ctx is not None:
    #         load_llm_calls = self.ctx.load_llm_calls.get(self._name)
    #         if load_llm_calls is not None:
    #             llm_call = await LlmCall.get_or_none(id=load_llm_calls)
    #             if llm_call is not None:
    #                 return llm_call
    #             raise ValueError(f"LLM call {load_llm_calls} not found")
    
    
    async def on_start(self):
        """
        Build subnetwork and register with context.

        Called automatically on first __anext__() call.
        Creates span, resolves dependencies, logs kwargs as inputs, and builds the internal stream.

        In replay mode (when span has saved outputs), sets up replay buffer instead of
        executing the generator function.

        Auto-cache behavior:
        - If ctx.cache_dir is set and no explicit load/save path is configured:
          - Computes cache key from stream name + bound arguments
          - If cache file exists: loads from cache (no re-execution)
          - If cache file doesn't exist: executes and saves to cache
        """
        from ..block import Block
        bound, kwargs = await self._resolve_dependencies()
        
        load_filepath = self._load_cache(bound)
        llm_call_id = None
        if not self.dry_run and self.ctx is not None:
            llm_call_id = self.ctx.load_llm_calls.get(self._name)
        # stream = self._init_stream(bound.args, bound.kwargs)
        
        if llm_call_id is not None:
            stream = await LLMStream.load_llm_call(llm_call_id, delay=self._load_delay or 0.05)
        elif load_filepath is not None:
            stream = LLMStream.load(load_filepath, delay=self._load_delay or 0.05)
        else:
            gen_instance = self._gen_func(*bound.args, **bound.kwargs)
            stream = LLMStream(gen_instance, name=f"{self._name}_stream")

        self._handle_save_cache(stream)

        self._stream = stream
        self._gen = stream
        if self._parser is not None:
            self._gen |= self._parser
        else:
            self._accumulator = Accumulator(Block())
            self._gen |= self._accumulator  
    
    
    async def _store_llm_call(self):
        from ..versioning.dataflow_models import LlmCall
        if self._parser:
            chunks = self._parser.get_chunks()
        else:
            chunks = []
        llm_call = await LlmCall(
            config=self.llm_config,
            usage=self._stream.usage,
            chunks=chunks,
            request_id=self._stream.response.id,
            message_id=self._stream.response.item_id,
            span_id=self.span.id
        ).save()
        await self.span.add(llm_call)

    
    
    async def on_stop(self):
        """
        Mark span as completed when process exhausts.
        """
        
        if self.span:
            self.span.status = "completed"
            self.span.end_time = __import__('datetime').datetime.now()
            await self.span.save()
            if hasattr(self._stream, "usage") and self._stream.response is not None:
                await self._store_llm_call()

        # Pop from context execution stack
        if self.ctx:
            await self.ctx.end_span()

    
    async def on_error(self, error: Exception):
        """
        Mark span as failed on error.
        """
        if self.span:
            self.span.status = "failed"
            self.span.end_time = __import__('datetime').datetime.now()
            await self.span.save()
            if hasattr(self._stream, "usage") and self._stream.response is not None:
                await self._store_llm_call()

        # Pop from context execution stack
        if self.ctx:
            await self.ctx.end_span()

            
    def set_stream(self):
        self._gen_func = self.llm.stream
    # def stream(self, event_level=None, ctx: "Context | None" = None, load_eval_args: bool = False) -> "FlowRunner":
    #     """
    #     Return a FlowRunner that streams events from this process.

    #     This enables event streaming directly from decorated functions:
    #         @stream()
    #         async def my_stream():
    #             yield "hello"

    #         async for event in my_stream().stream():
    #             print(event)

    #     Args:
    #         event_level: EventLogLevel.chunk, .span, or .turn

    #     Returns:
    #         FlowRunner configured to emit events
    #     """
    #     from ..prompt.flow_components import EventLogLevel
    #     from ..prompt.fbp_process import FlowRunner
    #     if event_level is None:
    #         event_level = EventLogLevel.chunk
    #     self._ctx = ctx
    #     self._load_eval_args = load_eval_args
    #     return FlowRunner(self, ctx=ctx, event_level=event_level).stream_events()
    
    def print_inputs(self):
        sep = "─" * 50
        for i,block in enumerate(self.blocks):
            print(sep)
            print(f"{i}: {block.role.title()} Message")
            print(sep)
            block.print()
            print(" ")
        for key, value in self._kwargs.items():
            print(sep)
    
    def print(self, inputs: bool = False):
        sep = "─" * 50
        
        if inputs:            
            self.print_inputs()
            # for i,block in enumerate(self.blocks):
            #     print(sep)
            #     print(f"{i}: {block.role.title()} Message")
            #     print(sep)
            #     block.print()
            #     print(" ")
            # for key, value in self._kwargs.items():
            #     print(sep)
            #     print(key, value)
            print("######################## Output ########################")
        self.span.llm_calls[0].print()
        # self.get_response().print()
        
        


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




# def pack_blocks(args: tuple[Any, ...]) -> tuple[BlockList, tuple[Any, ...]]:
#     # block_list = BlockList()
#     block_list = BlockList()
#     extra_args = ()
#     for arg in args:
#         if isinstance(arg, str):
#             block_list.append(Block(arg))
#         elif isinstance(arg, Block):
#             block_list.append_child(arg)
#         elif isinstance(arg, BlockList):
#             block_list.extend(arg)
#         # elif isinstance(arg, Block):
#         #     block_list.append(copy.copy(arg))
#         # elif isinstance(arg, BlockList):
#         #     block_list.extend(copy.copy(arg))
#         else:
#             extra_args += (arg,)
#     return block_list, extra_args

def pack_blocks(args: tuple[Any, ...]) -> tuple[BlockList, tuple[Any, ...]]:    
    block_list = []
    extra_args = ()
    for arg in args:
        if isinstance(arg, str):
            block_list.append(Block(arg))
        elif isinstance(arg, Block):
            block_list.append(arg)
        elif isinstance(arg, BlockList):
            block_list.extend(arg)
        # elif isinstance(arg, Block):
        #     block_list.append(copy.copy(arg))
        # elif isinstance(arg, BlockList):
        #     block_list.extend(copy.copy(arg))
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
            # return StreamController(gen_func=method, args=(self, blocks, *extra_args), kwargs=kwargs, name=name or method.__name__, span_type="llm")
            # return StreamController(gen_func=method, args=(self, blocks, *extra_args), kwargs=kwargs, name=name, span_type="llm")
            return LLMStreamController(gen_func=method, blocks=blocks, config=kwargs["config"], tools=kwargs["tools"], args=(self, blocks, *extra_args), kwargs=kwargs)
        return wrapper
    return llm_stream_decorator
    
 
class LLM: 
    config: LlmConfig   
    default_model: str
    models: List[str]
    
    
    def __init__(self, config: LlmConfig | None = None):        
        self.config = config or LlmConfig(model=self.default_model )
        
        
        
    def __call__(
        self, 
        *blocks: BlockChunk | Block | str, 
        config: LlmConfig | None = None, 
        tools: List[Type[BaseModel]] | None = None,
        schema: BlockSchema | None = None
    ) -> LLMStreamController:
        if config is None:
            config = LlmConfig(model=self.default_model)
        llm_blocks, extra_args = pack_blocks(blocks)
        llm = LLMStreamController(llm=self, blocks=llm_blocks, config=config, tools=tools, args=(llm_blocks, config, tools, *extra_args))
        if schema:
            llm.parse(schema)
        return llm
    
    
    async def stream(
        self,
        blocks: BlockList,
        config: LlmConfig,
        tools: List[Type[BaseModel]] | None = None
    ) -> AsyncGenerator[Any, None]:
        """
        This method is used to stream the response from the LLM.
        After decoration, this will return a StreamController instance.
        """        
        raise NotImplementedError("stream is not implemented")
        yield
    
    
    async def complete(
        self,
        blocks: BlockList,
        config: LlmConfig,
        tools: List[Type[BaseModel]] | None = None
    ) -> LLMResponse:
        """
        This method is used to complete the response from the LLM.
        After decoration, this will return a LLMResponse instance.
        """
        raise NotImplementedError("complete is not implemented")
        
    



    
    
class LLMRegistry():
    
    _model_registry: Dict[str, Type[LLM]] = {}
    _default_model: str | None = None
        
    
    
    @classmethod
    def register(cls, model_cls: Type[LLM], default_model: str | None = None) -> Type[LLM]:
        """Decorator to register a new LLM model implementation"""
        if model_cls.__name__ in cls._model_registry:
            raise ValueError(f"Model {model_cls.__name__} is already registered")
        for model in model_cls.models:
            cls._model_registry[model] = model_cls
        if default_model:
            cls._default_model = default_model
        return model_cls
    
    @classmethod
    def get_llm(cls, model: str | None = None) -> Type[LLM]:
        """Get a registered model by name"""
        if model is None:
            if cls._default_model is None:
                raise ValueError("No default model is set")
            model = cls._default_model
        if model not in cls._model_registry:
            raise KeyError(f"Model {model} is not registered")        
        llm_cls = cls._model_registry[model]
        return llm_cls
    
    
    @classmethod
    def build_llm(cls, model: str | None = None) -> LLM:
        llm_cls = cls.get_llm(model)
        return llm_cls()
    
    
    def stream(
        self,
        *blocks: BlockChunk | Block | str,
        model: str | None = None,
        tools: List[Type[BaseModel]] | None = None,
        config: LlmConfig | None = None,
    ) -> StreamController:
        llm_cls = self.get_llm(model)
        llm = llm_cls(config=config)
        
        return llm.stream(*blocks, tools=tools, config=config)

    def __call__(
        self,        
        *blocks: BlockChunk | Block |BlockList | str,
        model: str | None = None,
        config: LlmConfig | None = None,
    ) -> LLMStreamController:                                
        
        
        llm_blocks, extra_args = pack_blocks(blocks)

        
        llm_ctx = self.get_llm(model)        
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
