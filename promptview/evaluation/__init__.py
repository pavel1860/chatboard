"""
Evaluation system for testing and comparing agent executions.

This module provides:
- Value-level evaluators with path-based matching
- Test case and test run management
- Evaluation context for runtime evaluation
- Early stopping on evaluation failures
"""

from .models import (
    EvaluatorConfig,
    ValueEval,
    EvaluationFailure,
    TurnEval,
    TestRun,
    TestTurn,
    TestCase,
)
from .decorators import evaluator, EvalCtx
from .matching import match_value_to_evaluators, match_ltree_pattern
from .context import EvaluationContext

__all__ = [
    # Configuration and exceptions
    "EvaluatorConfig",
    "EvaluationFailure",

    # Models
    "ValueEval",
    "TurnEval",
    "TestRun",
    "TestTurn",
    "TestCase",

    # Decorators
    "evaluator",
    "EvalCtx",

    # Context
    "EvaluationContext",

    # Matching
    "match_value_to_evaluators",
    "match_ltree_pattern",
]
