"""Decorators for defining value evaluators."""

from functools import wraps
from typing import TYPE_CHECKING, Callable, Awaitable, Tuple
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..testing.test_models import TestCase, TestRun
    from ..prompt.span_tree import Value
    from .models import EvaluatorConfig
    from ..prompt.fbp_process import EvaluatorController


@dataclass
class EvalCtx:
    """
    Context passed to evaluator functions.

    Provides access to:
    - test_case: The test case being run
    - test_run: The current test run
    - config: Evaluator configuration (with metadata)
    """
    test_case: "TestCase"
    test_run: "TestRun"
    config: "EvaluatorConfig"


# Global registry of evaluator functions
class EvaluatorRegistry:
    def __init__(self):
        self._evaluator_registry: dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._evaluator_registry[name] = func

    def get(self, name: str) -> Callable | None:
        return self._evaluator_registry.get(name)
    
    def instantiate(self, value: "Value", eval_config: "EvaluatorConfig", test_case: "TestCase", test_run: "TestRun") -> "EvaluatorController":
        from ..prompt.fbp_process import EvaluatorController
        
        gen_func = self._evaluator_registry[eval_config.name]
        ctx = EvalCtx(test_case, test_run, eval_config)
        return EvaluatorController(gen_func, eval_config.name, span_type="evaluator", args=(), kwargs={"ctx": ctx, "ref_value": value, "test_value": value})

    def list(self) -> list[str]:
        return list(self._evaluator_registry.keys())


evaluator_registry = EvaluatorRegistry()


def evaluator(
    func: Callable[
        [EvalCtx, "Value", "Value"],
        Awaitable[float | Tuple[float, dict]]
    ]
) -> Callable:
    """
    Decorator to register a value evaluator function.

    The evaluator function should have signature:
        async def evaluator(
            ctx: EvalCtx,
            ref_value: Value,
            test_value: Value
        ) -> float | tuple[float, dict]

    Args:
        ctx: Evaluation context with test case, test run, and config
        ref_value: Reference value from the reference turn
        test_value: Test value being evaluated

    Returns:
        Either a score (float) or (score, metadata dict)

    Example:
        @evaluator
        async def validate_thought(ctx, ref_value, test_value):
            ref_output = ref_value.value
            test_output = test_value.value

            if ref_output == test_output:
                return 1.0, {"match": True}
            return 0.5, {"match": False}
    """
    @wraps(func)
    async def wrapper(
        ctx: EvalCtx,
        ref_value: "Value",
        test_value: "Value"
    ) -> Tuple[float, dict]:
        """Wrapper that ensures return value is always (score, metadata)."""
        result = await func(ctx, ref_value, test_value)

        if isinstance(result, tuple):
            return result
        else:
            # If only score returned, wrap in tuple with empty metadata
            return result, {}

    # Register in global registry
    evaluator_registry.register(func.__name__, wrapper)

    return wrapper


def get_evaluator(name: str) -> Callable | None:
    """Get an evaluator function by name from the registry."""
    return evaluator_registry.get(name)


def list_evaluators() -> list[str]:
    """List all registered evaluator names."""
    return evaluator_registry.list()
