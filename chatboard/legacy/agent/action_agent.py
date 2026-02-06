
from functools import wraps
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from chatboard.prompt.legacy.base_prompt import Prompt
from chatboard.state.context import Context
from chatboard.llms.messages import ActionCall, ActionMessage, HumanMessage
from chatboard.utils.function_utils import call_function, filter_func_args
from chatboard.prompt.legacy.chat_prompt import ChatPrompt
from chatboard.llms.tracer import Tracer
import inspect
from chatboard.prompt.legacy.decorator import prompt






class ActionAgent(BaseModel):
    name: str
    model: str = "gpt-4o"    
    iterations: int = 1
    router_prompt: Type[ChatPrompt] | None = None
    add_input_history: bool = False 
    is_traceable: bool = True 
    is_router: bool = False

    _action_handlers: Dict[str, callable] = {}    
        

    def handle(self, action: BaseModel, handler: callable):
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
    
    async def complete(views):
        return views
    
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
            # response = await call_function(
            #         self.router_prompt.__call__, 
            #         context=context, 
            #         message=message, 
            #         tracer_run=tracer_run, 
            #         action_output=action_output, 
            #         **kwargs
            #     )  
            #! need to add prompt tracer
            # view_block = await self.router_prompt.handle_render(context=context, message=message, **kwargs)
            # actions = self.router_prompt.actions
            # messages, actions = self.router_prompt.llm.transform(view_block, actions=actions, context=context, **kwargs)
            # response = await self.router_prompt.llm.complete(
            #         messages, 
            #         actions=actions, 
            #         tool_choice=self.router_prompt.tool_choice, 
            #         tracer_run=tracer_run, 
            #         **kwargs
            #     )
            #region main loop                  
            ex_ctx = self.router_prompt.build_execution_context(context=context, message= message, tracer_run=tracer_run, **kwargs)
            ex_ctx = await self.router_prompt.run_steps(ex_ctx)
            response = ex_ctx.output
            if not self.is_router:
                await context.history.add(context, message, str(tracer_run.id), self.name)                
            #? action handling loop
            for i in range(iterations):           
                if not self.is_router:
                    await context.history.add(context, response, str(tracer_run.id), self.name)
                tracer_run.add_outputs(response)
                if response.content and not response.action_calls:
                    yield response
                if response.action_calls:
                    action_output_views = []
                    for action_call in response.action_calls:
                        action_handler = self.get_action_handler(action_call)
                        if inspect.isasyncgenfunction(action_handler):
                            gen_kwargs = filter_func_args(action_handler, {"context": context, "action": action_call.action, "message": message, "tracer_run": tracer_run} | kwargs)
                            async for output in action_handler(**gen_kwargs):
                                if output is None:
                                    break
                                # tracer_run.add_outputs(output)
                                yield output
                        elif inspect.isfunction(action_handler):
                            action_output = await call_function(action_handler, context=context, action=action_call.action, message=message, tracer_run=tracer_run, **kwargs)
                        else:
                            raise ValueError(f"Invalid action handler: {action_handler}")
                        
                        #? handle action output
                        if action_output:
                            message = self.process_action_output(action_call, action_output)
                            if not self.is_router:
                                await context.history.add(context, message, str(tracer_run.id), self.name)
                            action_output_views.append(message)
                            action_output = None
                        else:
                            return   

                    # actions_views = render([response] + action_output_views)
                    # prev_views.extend(actions_views)
                    # complete(prev_views)                    
                    action_views = await self.router_prompt.process_render_output([response] + action_output_views)
                    ex_ctx = ex_ctx.copy_ctx(with_views=True)
                    ex_ctx.extend_views(action_views)
                    ex_ctx = await self.router_prompt.run_steps(ex_ctx)
                    response = ex_ctx.output

                    if i == iterations - 1:
                        yield response
                    
                    # ex_ctx = self.router_prompt.build_execution_context(
                    #         context=context, 
                    #         views=action_views, 
                    #         message= message, 
                    #         tracer_run=tracer_run, 
                    #         **kwargs
                    #     )
                    
                    
                    # ex_ctx = ex_ctx.to_new_execution()
                    # ex_ctx.root_block.extend(action_views)
                    # view_block.extend(action_views)
                    # messages, actions = self.router_prompt.llm.transform(view_block, actions=actions, context=context, **kwargs)
                    # response = await self.router_prompt.llm.complete(
                    #     messages, 
                    #     actions=actions, 
                    #     tool_choice=self.router_prompt.tool_choice, 
                    #     tracer_run=tracer_run, 
                    #     **kwargs
                    # )
                else:
                    break
                           
            else:
                if message.id not in context.history.contained_id:
                    await context.history.add(context, message, str(tracer_run.id), "user")
#endregion

    def reducer(self, action: BaseModel):
        def decorator(func):
            self.handle(action, func)
            return func
        return decorator
    
    
    
ActionAgent.prompt = prompt