import inspect
from typing import (Any, Callable,
                    TypeVar, ParamSpec, AsyncGenerator)
from ..tracer import Tracer
from .controller import Controller



P = ParamSpec('P')
R = TypeVar('R')
YieldType = TypeVar('YieldType')
# class Agent(Generic[P, YieldType]):   
class Agent(Controller[P, AsyncGenerator[YieldType, None]]):

    # _complete: Callable[P, AsyncGenerator[YieldType, None]]    
        
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[YieldType, None]:
        execution_ctx = self.build_execution_ctx()
        async with execution_ctx.start_tracer(self._name, "chain", inputs={}) as ctx:
            injection_kwargs = await self._inject_dependencies(*args, **kwargs)
            # with Tracer(
            #     name=self._name,
            #     inputs=self._filter_args_for_trace(*args, **kwargs, **injection_kwargs),
            #     # session_id=str(ctx.session_id)
            # ) as tracer_run:
            kwargs.update(injection_kwargs)
            try:
                async for output in self._complete(*args, **kwargs):
                    if inspect.isasyncgen(output):
                        try:
                            async for gen_output in output:
                                yield gen_output
                        except GeneratorExit:
                            pass
                    else:
                        yield output
            except GeneratorExit:
                print(f"GeneratorExit in {ctx.tracer.id}")
                # tracer_run._reset_context()
                # tracer_run.end()
                return






def agent(
    **kwargs: Any
) -> Callable[[Callable[P, AsyncGenerator[YieldType, None]]], 'Agent[P, YieldType]']:
    """
    Decorator to create an Agent from a generator function.

    Args:
        **kwargs: Arbitrary keyword arguments to configure the Agent.

    Returns:
        A decorator that transforms a generator function into an Agent instance.
    """
    
    def decorator(func: Callable[P, AsyncGenerator[YieldType, None]]) -> 'Agent[P, YieldType]':
        # Initialize the Agent with provided keyword arguments
        agent = Agent(
            **kwargs
        )
        # Assign the generator function to the agent
        agent._name = func.__name__
        agent._complete = func
        return agent
        
    return decorator


# def agent(
#     **kwargs: Any
# )-> Callable[[Callable[P, R]], Agent[P, R]]:
    
#     def decorator(func: Callable[P, R]) -> Agent[P,R]:
#         agent = Agent(
#                 # model=model, #type: ignore                
#                 **kwargs
#             )
#         agent._name=func.__name__
#         agent._complete = func
#         return agent        
#     return decorator