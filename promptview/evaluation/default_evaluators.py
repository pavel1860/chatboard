from .decorators import evaluator
from .eval_context import EvalCtx
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..model.versioning.models import DataFlowNode


@evaluator(name="prompt")
async def prompt_evaluator(
    ctx: EvalCtx,
    ref_value: "DataFlowNode",
    test_value: "DataFlowNode"
):
    ctx.config.metadata.get("prompt")
    return 1.0, {"match": True}








@evaluator(name="distance")
async def distance_evaluator(
    ctx: EvalCtx,
    ref_value: "DataFlowNode",
    test_value: "DataFlowNode"
):
    return 1.0, {"match": True}