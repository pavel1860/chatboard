from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Awaitable, Tuple, TypeVar

from ..block import BlockChunk
from ..testing.test_models import Evaluation, EvaluatorConfig, TurnEvaluator    

if TYPE_CHECKING:
    from ..model.versioning.models import Turn
    from ..testing.test_models import TestCase, TestRun



    
    





evaluator_store = {}



class EvalCtx:
    test_case: "TestCase"
    test_run: "TestRun"
    config: EvaluatorConfig
    
    
    def __init__(self, test_case: "TestCase", test_run: "TestRun", config: EvaluatorConfig):
        self.test_case = test_case
        self.test_run = test_run
        self.config = config




TURN_MODEL = TypeVar("TURN_MODEL", bound="Turn", covariant=True)
    


def evaluator(
    func: Callable[["EvalCtx", TURN_MODEL, TURN_MODEL], Awaitable[Tuple[float, dict]]] | Callable[["EvalCtx", TURN_MODEL, TURN_MODEL], Awaitable[float]]
) -> Callable[["TestCase", "TestRun", EvaluatorConfig], Callable[[TURN_MODEL, TURN_MODEL], Awaitable[Evaluation]]]:
    
    @wraps(func)
    def decorator(test_case: "TestCase", test_run: "TestRun", config: EvaluatorConfig) -> Callable[[TURN_MODEL, TURN_MODEL], Awaitable[Evaluation]]:        
        async def run_evaluator(ref_turn: TURN_MODEL, test_turn: TURN_MODEL) -> Evaluation:
            ctx = EvalCtx(test_case, test_run, config)
            res = await func(ctx, ref_turn, test_turn)
            if isinstance(res, tuple):
                score, metadata = res
            else:
                score = res
                metadata = {}
            return Evaluation(
                evaluator=config.name,
                score=score,
                metadata=metadata,
                turn_eval_id=test_turn.id,
            )
        return run_evaluator
    evaluator_store[func.__name__] = decorator
    return decorator
    
    
    
      

