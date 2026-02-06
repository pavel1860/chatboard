
import asyncio
from typing import Any, Callable, Dict, Generator, List, Tuple

from ..llms.tracer import Tracer
from ..prompt.legacy.chat_prompt import ChatPrompt
from ..utils.function_utils import call_function


async def map_prompt(
    prompt: ChatPrompt | Callable,
    kwargs_list: list[Dict[str, Any]] | Generator[Dict[str, Any], Any, Any],
    metadata: dict = {},
    parallelism: int = 10,
    is_traceable: bool = True,
    name: str | None = None,
    verbose=False,   
    logger=None,
    output_run_id: bool = False,
) -> Tuple[List[Any], str] | List[Any]:
    if not name:
        if isinstance(prompt, ChatPrompt):
            name = prompt.name
        else:
            name = prompt.__name__
    name = f"{name}_map"
    with Tracer(
        is_traceable=is_traceable,
        name=name,
        run_type="chain",
        inputs={},
        extra=metadata,
        tags=["map"]
    ) as tracer_run:
        semaphore = asyncio.Semaphore(parallelism)
        async def run_prompt(index, **kwargs):
            async with semaphore:
                # output = await prompt(tracer_run=tracer_run, index=index, **kwargs)
                try:
                    output = await call_function(prompt, tracer_run=tracer_run, index=index, **kwargs)
                    if logger and verbose:
                        logger.info(f"Completed item {index}.")
                    return output
                except Exception as e:
                    if logger:
                        logger.exception(f"Error processing item {index}: {e}")
                    return None
        
        tasks = [run_prompt(index=index, **kwargs) for index, kwargs in enumerate(kwargs_list)]
        results = await asyncio.gather(*tasks)
        tracer_run.end(outputs={'map_length': len(results)})
    if output_run_id:
        return results, str(tracer_run.id)
    else:
        return results


