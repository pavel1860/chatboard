
import inspect
from functools import wraps
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union, Callable
from pydantic import BaseModel
from chatboard.prompt.legacy.base_prompt import Prompt
from chatboard.state.context import Context
from chatboard.llms.messages import AIMessage, ActionCall, ActionMessage, HumanMessage
from chatboard.utils.function_utils import call_function, filter_func_args
from chatboard.prompt.legacy.chat_prompt import ChatPrompt
from chatboard.llms.tracer import Tracer
from chatboard.prompt.legacy.chat_prompt import ChatPrompt
from chatboard.prompt.legacy.decorator import prompt
from chatboard.state.context import Context
from chatboard.utils.function_utils import call_function, filter_func_args
from pydantic import BaseModel






class AgentRouter(BaseModel):
    name: str
    model: str = "gpt-4o"    
    iterations: int = 1
    router_prompt: Type[ChatPrompt] | None = None
    add_input_history: bool = False 
    is_traceable: bool = True 
    is_router: bool = False

    _action_handlers: Dict[str, callable] = {}    
       
    def handle(self, action: BaseModel, handler: Callable):
        self._action_handlers[action.__name__] =  handler
        
    
    def get_action_handler(self, action_call: ActionCall):
        handler = self._action_handlers[action_call.action.__class__.__name__]
        if not handler:
            raise ValueError(f"Action handler not found for {action_call.action.__class__.__name__}")
        return handler
    
    
    def process_action_output(self, action_call: ActionCall, action_output):          
        if type(action_output) == str:
            action_output_str = action_output
        elif isinstance(action_output, BaseModel):
            action_output_str = action_output.model_dump_json()
        else:
            raise ValueError(f"Invalid action output ({type(action_output)}): {action_output}")
        message = ActionMessage(
                id=action_call.id,
                content=action_output_str,
                # tool_call=response.tool_calls[i]
            )
        return message
        
        
    
    async def __call__(self, context: Context, message: HumanMessage | str, iterations: int | None = None, tracer_run=None, **kwargs):
        iterations = iterations or self.iterations
        message = HumanMessage(content=message) if isinstance(message, str) else message
        with Tracer(
            name=self.name,
            is_traceable=self.is_traceable,
            inputs={"message": message.content} | kwargs,
            session_id=context.session_id,
            tracer_run=tracer_run
        ) as tracer_run:
            
            action_output = None
            for i in range(iterations):           
                response = await call_function(
                        self.router_prompt.__call__, 
                        context=context, 
                        message=message, 
                        tracer_run=tracer_run, 
                        action_output=action_output, 
                        **kwargs
                    )
                await context.history.add(context, message, str(tracer_run.id), self.name)
                if not self.is_router or (response.output and hasattr(response.output,"_history") and response.output._history):                    
                    await context.history.add(context, response, str(tracer_run.id), self.name)                
                if response.content:
                    tracer_run.add_outputs(response)
                    yield response
                if response.action_calls:
                    for action_call in response.action_calls:
                        action_handler = self.get_action_handler(action_call)
                        if inspect.isasyncgenfunction(action_handler):
                            gen_kwargs = filter_func_args(action_handler, {"context": context, "action": action_call.action, "message": message, "tracer_run": tracer_run} | kwargs)
                            async for output in action_handler(**gen_kwargs):
                                if output is None:
                                    break
                                tracer_run.add_outputs(output)
                                yield output
                        elif inspect.isfunction(action_handler):
                            action_output = await call_function(action_handler, context=context, action=action_call.action, message=message, tracer_run=tracer_run, **kwargs)
                        else:
                            raise ValueError(f"Invalid action handler: {action_handler}")
                        if action_output:
                            if isinstance(action_output, AIMessage):                                
                                action_message = self.process_action_output(action_call, action_output)
                                await context.history.add(context, action_message, str(tracer_run.id), self.name)
                                await context.history.add(context, action_output, str(tracer_run.id), self.name)
                                yield action_output
                                return 
                            message = self.process_action_output(action_call, action_output)
                        else:
                            return
                else:
                    break                    
            else:
                await context.history.add(context, message, str(tracer_run.id), self.name)
    
    

    
    async def __call__2(self, context: Context, message: HumanMessage | str, iterations: int | None = None, tracer_run=None, **kwargs):
        iterations = iterations or self.iterations
        message = HumanMessage(content=message) if isinstance(message, str) else message
        has_action_response = False
        with Tracer(
            name=self.name,
            is_traceable=self.is_traceable,
            inputs={"message": message.content} | kwargs,
            session_id=context.session_id,
            tracer_run=tracer_run
        ) as tracer_run:
            
            # if self.add_input_history:
            #     context.history.add(context, message, str(tracer_run.id), "user")
            action_output = None
            for i in range(iterations):           
                response = await call_function(
                        self.router_prompt.__call__, 
                        context=context, 
                        message=message, 
                        tracer_run=tracer_run, 
                        action_output=action_output, 
                        **kwargs
                    )
                if not self.is_router:
                    context.history.add(context, message, str(tracer_run.id), "user")
                    context.history.add(context, response, str(tracer_run.id), self.name)
                tracer_run.add_outputs(response)
                if response.content:                    
                    yield response
                if response.action_calls:
                    for action_call in response.action_calls:
                        action_handler = self._action_handlers[action_call.action.__class__.__name__]
                        if inspect.isasyncgenfunction(action_handler):
                            gen_kwargs = filter_func_args(action_handler, {"context": context, "action": action_call.action, "message": message.content, "tracer_run": tracer_run} | kwargs)
                            async for output in action_handler(**gen_kwargs):
                                if output is None:
                                    break
                                tracer_run.add_outputs(output)
                                yield output
                        elif inspect.isfunction(action_handler):
                            action_output = await call_function(action_handler, context=context, action=action_call.action, message=message.content, tracer_run=tracer_run, **kwargs)
                            # tracer_run.add_outputs({"tool_output": action_output})
                        else:
                            raise ValueError(f"Invalid action handler: {action_handler}")
                        if action_output:
                            has_action_response = True
                            
                            if type(action_output) == str:
                                action_output_str = action_output
                            elif isinstance(action_output, BaseModel):
                                action_output_str = action_output.model_dump_json()
                            else:
                                raise ValueError(f"Invalid action output ({type(action_output)}): {action_output}")
                            message = ActionMessage(
                                    id=action_call.id,
                                    content=action_output_str,
                                    # tool_call=response.tool_calls[i]
                                )
                            # response.add_action_output(action_message)
                            # context.history.add(context, message, str(tracer_run.id), self.name)
                        else:
                            # tracer_run.end()
                            return
                else:
                    break
                    
                if not has_action_response:
                    break
            else:
                context.history.add(context, message, str(tracer_run.id), "user")
                # message = None                  
            # tracer_run.end()

    # def prompt(
    #     self,
    #     model: str = "gpt-3.5-turbo-0125",
    #     llm: Union[OpenAiLLM, AzureOpenAiLLM, None] = None,
    #     background: Optional[str] = None,
    #     task: Optional[str] = None,
    #     rules: Optional[str | List[str]] = None,
    #     examples: Optional[str | List[str]] = None,
    #     actions: Optional[List[Type[BaseModel]]] = None,
    #     response_model: Optional[Type[BaseModel]] = None,
    #     parallel_actions: bool = False,
    #     is_traceable: bool = True
    # ):
    #     if llm is None:
    #         llm = OpenAiLLM(
    #             model=model,
    #             parallel_tool_calls=parallel_actions
    #         )
        
    #     def decorator(func):            
    #         prompt = ChatPrompt(
    #                 name=func.__name__,
    #                 model=model,
    #                 llm=llm,
    #                 background=background,
    #                 task=task,
    #                 rules=rules,
    #                 examples=examples,
    #                 actions=actions,
    #                 response_model=response_model,
    #             )
    #         prompt.set_render_method(func)
    #         self.router_prompt = prompt
    #         @wraps(func)
    #         async def wrapper(**kwargs):             
    #             return await prompt(**kwargs)                
                
    #         return wrapper
        
    #     return decorator


    def reducer(self, action: BaseModel):
        def decorator(func):
            self.handle(action, func)
            return func
        return decorator
    
    
    
AgentRouter.prompt = prompt # type: ignore