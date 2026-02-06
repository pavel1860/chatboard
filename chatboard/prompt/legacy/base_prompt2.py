import inspect
from functools import wraps
from typing import (Any, Awaitable, Callable, Generic, List, Literal, Type,
                    TypedDict, TypeVar)

from chatboard.llms.legacy.anthropic_llm import AnthropicLLM
from chatboard.llms.exceptions import LLMToolNotFound
from chatboard.llms.legacy.llm2 import LLM
from chatboard.llms.messages import AIMessage, BaseMessage, HumanMessage
from chatboard.llms.legacy.openai_llm import OpenAiLLM
from chatboard.llms.tracer import Tracer
from chatboard.llms.utils.action_manager import Actions
from chatboard.prompt.legacy.mvc import ViewBlock, create_view_block
from chatboard.state.context import Context
from chatboard.utils.function_utils import call_function
from pydantic import BaseModel, Field

T = TypeVar('T')

ToolChoiceParam = Literal['auto', 'required', 'none'] | BaseModel | None


class PromptInputs(BaseModel):
    message: BaseMessage | None = None
    view_blocks: List[ViewBlock] | ViewBlock | None = None
    context: Context | None = None
    actions: List[Type[BaseModel]] | None = None
    tool_choice: ToolChoiceParam = None
    tracer_run: Tracer | None = None
    kwargs: dict[str, Any] | None = None
    
    class Config:
        arbitrary_types_allowed = True
    

class PromptExecutionContext(BaseModel):  
    name: str
    inputs: PromptInputs
    messages: List[BaseMessage] | None = None
    actions: Actions | None = None
    error: str | None = None
    root_block: List[ViewBlock] | None = None    
    prompt_run: Tracer | None = None
    response: AIMessage | None = None
    output: Any | None = None
    llm: LLM
    
    class Config:
        arbitrary_types_allowed = True
        
        
        
    async def process_render_output(self, views: Any, **kwargs: Any):
        if isinstance(views, str) or isinstance(views, BaseModel):
            return create_view_block(views, view_name=self.__class__.__name__.lower(), role='user')            
        elif isinstance(views, list):
            valid_views = []
            for view in views:
                if isinstance(view, list):
                    raise ValueError("Nested lists are not supported")
                view = view.to_view_block() if hasattr(view, "to_view_block") else view
                if not isinstance(view, ViewBlock):
                    valid_views.append(create_view_block(view, view_name="render_" + (self.name or self.__class__.__name__), role='user'))
                else:
                    valid_views.append(view)
            return valid_views
            
        elif isinstance(views, tuple):
            return create_view_block(views, view_name="render_" + (self.name or self.__class__.__name__), role='user')            
        return views
    
    
    async def transform(self, views: List[ViewBlock] | ViewBlock | None = None, **kwargs: Any):
        template_views = []
        # for field_name, field_info in cls.model_fields.items():
        #     if field_name in ["llm", "model", "actions", "is_traceable", "tool_choice"]:
        #         continue
        #     view = await cls.property_to_view(field_name, field_info, **kwargs)
        #     if view is not None:
        #         template_views.append(view)
        if not views:
            views = []
        elif not isinstance(views, list):
            views = [views]
        if template_views:
            system_view = create_view_block(
                    template_views,
                    view_name=self.name.lower()+"_template",
                    role='system',                
                )
            views = [system_view] + views
        if not views:
            raise ValueError("No views to transform")
        return create_view_block(views, "root")
        

    def messages_step(self):
        self.messages, self.actions = self.llm.transform(
                root_block=self.root_block, 
                actions=self.inputs.actions, 
                context=self.inputs.context, 
                **self.inputs.kwargs
            )
        return self
    
    
    async def run_steps(self, views: List[ViewBlock] | ViewBlock | None = None):
        
        views = await self.process_render_output(views, context=self.inputs.context, **self.inputs.kwargs)
        self.root_block = await self.transform(views, context=self.inputs.context, **self.inputs.kwargs)
        if not self.messages:
            self.messages_step()
        response = await self.complete_step()
        return response
    
    
    async def complete_step(self):
        extra_messages = []
        if self.error:
            if self.output:
                extra_messages = [
                    self.output,
                    HumanMessage(content=self.error)
                ]
        response = await self.llm.complete(
            messages=self.messages + extra_messages, 
            actions=self.actions, 
            tool_choice=self.inputs.tool_choice, 
            tracer_run=self.prompt_run, 
            **self.inputs.kwargs
        )
        self.response = response
        return response
    
    def copy_ctx(self, with_views=False, with_messages=False, with_tracer=False):
        ctx = PromptExecutionContext(
            name=self.name,
            llm=self.llm,
            inputs=self.inputs.model_copy(),
        )
        if with_views:
            ctx.root_block = self.root_block.model_copy()
        if with_messages:
            ctx.messages = [m.model_copy() for m in self.messages]
        if with_tracer:
            ctx.prompt_run = self.prompt_run
        return ctx
        
    
    def extend_views(self, views: List[ViewBlock]):
        self.root_block.extend(views)
        return self.root_block
        
        


class Prompt(BaseModel, Generic[T]):    
    is_traceable: bool = True
    llm: LLM = Field(default_factory=OpenAiLLM)
    tool_choice: ToolChoiceParam = None
    actions: List[Type[BaseModel]] | None = []
    
    _render_method: Callable | None = None
    _output_parser_method: Callable | None = None
    
    _name: str | None = None
    
    
    def render(self):
        raise NotImplementedError("render method is not set")
    
    
    async def process_render_output(self, views: Any, **kwargs: Any) -> T:
        if isinstance(views, str) or isinstance(views, BaseModel):
            return create_view_block(views, view_name=self.__class__.__name__.lower(), role='user')            
        elif isinstance(views, list):
            valid_views = []
            for view in views:
                if isinstance(view, list):
                    raise ValueError("Nested lists are not supported")
                if not isinstance(view, ViewBlock):
                    valid_views.append(create_view_block(view, view_name="render_" + (self._name or self.__class__.__name__), role='user'))
                else:
                    valid_views.append(view)
            return valid_views
            
        elif isinstance(views, tuple):
            return create_view_block(views, view_name="render_" + self._name or self.__class__.__name__, role='user')            
        return views
        
    
    async def handle_render(self, views: List[ViewBlock] | ViewBlock | None= None, context=None, **kwargs: Any) -> List[ViewBlock] | ViewBlock:
        if not views:
            views = await call_function(
                    self.render if self._render_method is None else self._render_method, 
                    context=context, 
                    **kwargs
                )
        # views = await self.process_render_output(views, context=context, **kwargs)
        # view_block = await self.transform(views, context=context, **kwargs)
        return view_block
    
        
    
    
    async def property_to_view(self, property_name: str, field_info, **kwargs: Any) -> ViewBlock | None:
        view = getattr(self, property_name)
        title = property_name.title()
        if not view:
            return None
        if inspect.isfunction(view):
            view = await call_function(view, **kwargs)
            if not view:
                return None
        if not isinstance(view, ViewBlock):
            extra = field_info.json_schema_extra
            if extra is not None:
                if "title" in extra:
                    title = extra["title"]
            view = create_view_block(view, property_name, title=title, role='system', tag=property_name)
        return view
    
    
    async def transform(self, views: List[ViewBlock] | ViewBlock | None = None, **kwargs: Any) -> T:
        template_views = []
        for field_name, field_info in self.model_fields.items():
            if field_name in ["llm", "model", "actions", "is_traceable", "tool_choice"]:
                continue
            view = await self.property_to_view(field_name, field_info, **kwargs)
            if view is not None:
                template_views.append(view)
        if not views:
            views = []
        elif not isinstance(views, list):
            views = [views]
        if template_views:
            system_view = create_view_block(
                    template_views,
                    view_name=self.__class__.__name__.lower()+"_template",
                    role='system',                
                )
            views = [system_view] + views
        if not views:
            raise ValueError("No views to transform")
        return create_view_block(views, "root")
                
    
    # async def complete(
    #         self,
    #         message: str | BaseMessage | None = None,
    #         views: List[ViewBlock] | ViewBlock | None = None, 
    #         context: Context | None = None, 
    #         actions: List[Type[BaseModel]] | None = None,
    #         tool_choice: ToolChoiceParam = None,
    #         tracer_run: Tracer | None=None,
    #         output_messages: bool = False,
    #         **kwargs: Any
    #     ) -> T:
        
    #     if message is not None:
    #         message = HumanMessage(content=message) if isinstance(message, str) else message        
    #     with Tracer(
    #         is_traceable=self.is_traceable if not output_messages else False,
    #         tracer_run=tracer_run,
    #         name=self._name or self.__class__.__name__,
    #         run_type="prompt",
    #         inputs={
    #             "input": kwargs,
    #         },
    #     ) as prompt_run:
    #         try:
    #             actions = actions or self.actions
    #             view_block = await self.handle_render(views=views, context=context, message=message, **kwargs)
    #             messages, actions = self.llm.transform(view_block, actions=actions, context=context, **kwargs)
    #             if output_messages:
    #                 return messages
                
    #             response = await self.llm.complete(
    #                 messages, 
    #                 actions=actions, 
    #                 tool_choice=tool_choice or self.tool_choice, 
    #                 tracer_run=prompt_run, 
    #                 **kwargs
    #             )
    #             prompt_run.end(outputs={'output': response})
    #             return response
    #         except Exception as e:
    #             prompt_run.end(errors=str(e))
    #             raise e
    
    
    async def complete(
        self,
        args: Any,
        context: Context | None = None,
        **kwargs: Any
    ):
        pass
        
    
    
    async def __call__(self, *args, **kwargs):        
        ex_ctx = self.build_execution_context(*args, **kwargs)        
        for i in range(3):
            try:
                ex_ctx = await self.run_steps(ex_ctx)                      
                # output = await call_function(self.parse_output, response=ex_ctx.output, messages=ex_ctx.messages, actions=ex_ctx.actions, **kwargs)
                return ex_ctx.output
            except LLMToolNotFound as e:
                print(f"Tool not found: {e}")
                ex_ctx.error = f"Tool not found:{str(e)}"
        else:
            raise e

        
        
    async def run_steps(self, ex_ctx: PromptExecutionContext)-> PromptExecutionContext:
        with self.build_tracer(ex_ctx) as prompt_run:
            try:
                ex_ctx.prompt_run = prompt_run
                if not ex_ctx.root_block:
                    ex_ctx = await self.view_step(ex_ctx)
                # if not ex_ctx.messages:
                #     ex_ctx = self.messages_step(ex_ctx)
                # ex_ctx = await self.complete_step(ex_ctx)
                prompt_run.end(outputs={'output': ex_ctx.response.raw})
                # prompt_run.end(outputs={'output': ex_ctx.output.to_langsmith()})
                return ex_ctx
            except Exception as e:
                prompt_run.end(errors=str(e))
                raise e
        
    async def parse_output(self, response: AIMessage, messages: List[BaseMessage], actions: Actions, **kwargs: Any):
        if self._output_parser_method:
            return await call_function(self._output_parser_method, response=response, messages=messages, actions=actions, **kwargs)
        return response
    
        
    async def to_ex_ctx(
        self,
        *args: Any,
        **kwargs: Any
    ):
        ex_ctx = self.build_execution_context(*args, **kwargs)
        ex_ctx = await self.view_step(ex_ctx)
        ex_ctx = self.messages_step(ex_ctx)
        return ex_ctx
        
        
    def build_tracer(self, ex_ctx: PromptExecutionContext):
        inputs = {}
        if ex_ctx.inputs.message:
            inputs["message"] = ex_ctx.inputs.message.content
        if ex_ctx.inputs.kwargs:
            inputs["input"] = ex_ctx.inputs.kwargs
        
        return Tracer(
            is_traceable=self.is_traceable,
            tracer_run=ex_ctx.inputs.tracer_run,
            name=self._name or self.__class__.__name__,
            run_type="prompt",
            inputs=inputs
        )
    
    
    def build_execution_context(
            self,
            # message: str | BaseMessage | None = None,
            views: List[ViewBlock] | ViewBlock | None = None, 
            context: Context | None = None, 
            actions: List[Type[BaseModel]] | None = None,
            tool_choice: ToolChoiceParam = None,
            tracer_run: Tracer | None=None,
            **kwargs: Any
        ) -> PromptExecutionContext:
        # if message is not None:
            # message = HumanMessage(content=message) if isinstance(message, str) else message        
        prompt_inputs = PromptInputs(
            # message=message,
            view_blocks=views,
            context=context,
            actions=actions or self.actions,
            tool_choice=tool_choice or self.tool_choice,
            tracer_run=tracer_run,
            kwargs=kwargs
        )
        return PromptExecutionContext(
                name=self._name or self.__class__.__name__, 
                inputs=prompt_inputs, 
                llm=self.llm
            )
    
    async def view_step(self, ex_ctx: PromptExecutionContext):
        # root_block = await self.handle_render(
        #     views=ex_ctx.inputs.view_blocks, 
        #     context=ex_ctx.inputs.context, 
        #     message=ex_ctx.inputs.message, 
        #     **ex_ctx.inputs.kwargs
        # )
        # ex_ctx.root_block = root_block
        # return ex_ctx
        
        output = await call_function(
                self.render if self._render_method is None else self._render_method, 
                llm=ex_ctx.run_steps,
                context=ex_ctx.inputs.context, 
                **ex_ctx.inputs.kwargs
            )
        ex_ctx.output = output
        return ex_ctx
    # def messages_step(self, ex_ctx: PromptExecutionContext):
    #     messages, actions = self.llm.transform(
    #             root_block=ex_ctx.root_block, 
    #             actions=ex_ctx.inputs.actions, 
    #             context=ex_ctx.inputs.context, 
    #             **ex_ctx.inputs.kwargs
    #         )
    #     ex_ctx.messages = messages
    #     ex_ctx.actions = actions
    #     return ex_ctx
    
    # async def complete_step(self, ex_ctx: PromptExecutionContext):
    #     extra_messages = []
    #     if ex_ctx.error:
    #         if ex_ctx.output:
    #             extra_messages = [
    #                 ex_ctx.output,
    #                 HumanMessage(content=ex_ctx.error)
    #             ]
    #     response = await self.llm.complete(
    #         messages=ex_ctx.messages + extra_messages, 
    #         actions=ex_ctx.actions, 
    #         tool_choice=ex_ctx.inputs.tool_choice, 
    #         tracer_run=ex_ctx.prompt_run, 
    #         **ex_ctx.inputs.kwargs
    #     )
    #     ex_ctx.output = response
    #     return ex_ctx
            
    def set_methods(self, render_func: Callable | None = None, output_parser: Callable | None = None) -> None:
        self._render_method = render_func
        self._output_parser_method = output_parser
    
        
    

    @classmethod
    def decorator_factory(cls):
        def prompt_decorator( 
            self=None,   
            model: str = "gpt-4o",
            llm: LLM | None = None,            
            parallel_actions: bool = True,
            is_traceable: bool = True,
            output_parser: Callable[[AIMessage], T] | None = None,
            tool_choice: ToolChoiceParam = None,
            actions: List[Type[BaseModel]] | None = None,
            max_tokens: int = 4096,
            **kwargs: Any
        ):
            if llm is None:
                if model.startswith("gpt"):
                    llm = OpenAiLLM(
                        model=model, 
                        parallel_tool_calls=parallel_actions
                    )
                elif model.startswith("claude"):
                    llm = AnthropicLLM(
                        model=model, 
                        parallel_tool_calls=parallel_actions
                    )
            def decorator(func) -> Callable[..., Awaitable[T]]:
                prompt = cls[T](
                        model=model,
                        llm=llm,                        
                        tool_choice=tool_choice or cls.model_fields.get("tool_choice").default,
                        actions=actions,
                        is_traceable=is_traceable,
                        **kwargs
                    )
                prompt._name=func.__name__
                prompt.set_methods(func, output_parser)
                if self:
                    self.router_prompt = prompt
                # @wraps(func)
                # async def wrapper(**kwargs) -> T:            
                #     return await prompt(**kwargs)
                return prompt
                    
                
                # wrapper.__signature__ = sig
                return wrapper
            
            return decorator
        return prompt_decorator







prompt = Prompt.decorator_factory()