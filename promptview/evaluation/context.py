"""Evaluation context for tracking evaluations during execution."""

from typing import TYPE_CHECKING, Any, Callable
from dataclasses import dataclass, field

# from .models import EvaluatorConfig, ValueEval, EvaluationFailure
from ..model import EvaluatorConfig, ValueEval, EvaluationFailure
from .decorators import get_evaluator, EvalCtx, evaluator_registry
from .matching import match_value_to_evaluators

if TYPE_CHECKING:
    from ..prompt.span_tree import DataFlow, SpanTree
    from ..prompt.fbp_process import EvaluatorController
    from ..model import TestCase, TestRun, TurnEval, TestTurn


@dataclass
class EvaluationContext:
    """
    Context for managing evaluations during agent execution.

    This context:
    - Stores evaluator configurations
    - Tracks reference values for comparison
    - Runs evaluators as values are logged
    - Saves evaluation results
    - Handles early stopping
    """

    test_case: "TestCase"
    test_run: "TestRun"
    turn_eval: "TurnEval"
    test_turn: "TestTurn"

    # Reference span trees (for looking up values by path)
    reference_span_trees: list["SpanTree"] = field(default_factory=list)

    # Track evaluation results
    results: list[dict[str, Any]] = field(default_factory=list)
    value_evals: list[ValueEval] = field(default_factory=list)

    
    def get_eval_span_tree(self, path: list[int]) -> "SpanTree":
        span_tree = self.reference_span_trees[path[0]].get(path[1:])
        if span_tree is None:
            raise ValueError(f"Span tree not found for path {path}")
        return span_tree
    
    def get_ref_value(self, path: list[int]) -> "DataFlow":
        span_tree = self.reference_span_trees[0]
        if span_tree is None:
            raise ValueError(f"Span tree not found for path {path}")
        value = span_tree.get_value_by_path(path)
        if value is None:
            raise ValueError(f"Value not found for path {path}")
        return value
    
    
    def get_evaluators(self, value: "DataFlow") -> list["EvaluatorConfig"]:
        return match_value_to_evaluators(
            value,
            self.get_eval_span_tree(value.path[:-1]),
            self.test_turn.evaluators
        )
        
        
    def get_evaluator_handlers(self, value: "DataFlow") -> list[tuple[Callable, EvalCtx]]:
        evaluator_handlers = []
        for evaluator_config in self.get_evaluators(value):
            gen_func = evaluator_registry.get(evaluator_config.name)
            if gen_func is None:
                raise ValueError(f"Evaluator function not found for {evaluator_config.name}")
            ctx = EvalCtx(
                test_case=self.test_case,
                test_run=self.test_run,
                config=evaluator_config
            )
            evaluator_handlers.append((gen_func, ctx))
        return evaluator_handlers
    
    
    async def log_value_eval(self, value: "DataFlow", score: float, metadata: dict, evaluator_config: "EvaluatorConfig"):
        value_eval = await ValueEval(
            turn_eval_id=self.turn_eval.id,
            value_id=value.id,
            path=value.str_path,
            evaluator=evaluator_config.name,
            score=score,
            metadata=metadata,
        ).save()
        self.value_evals.append(value_eval)
        self.results.append({
            "evaluator": evaluator_config.name,
            "score": score,
            "path": value.str_path,
            "metadata": metadata
        })
        return value_eval
    
    async def commit(self):
       self.turn_eval.score = self.get_average_score()
       await self.turn_eval.save()
    
    def build_evaluator_controllers(self, value: "DataFlow") -> list["EvaluatorController"]:
        from ..prompt.fbp_process import EvaluatorController
        evaluator_controllers = []
        for evaluator_config in self.get_evaluators(value):
            gen_func = evaluator_registry.get(evaluator_config.name)
            if gen_func is None:
                raise ValueError(f"Evaluator function not found for {evaluator_config.name}")
            evaluator_controller = EvaluatorController(
                gen_func, 
                evaluator_config.name, 
                span_type="evaluator", 
                args=(), 
                kwargs={
                    "ctx": self, 
                    "ref_value": value, 
                    "test_value": value
                }
            )
            evaluator_controllers.append(evaluator_controller)
        return evaluator_controllers


    async def evaluate_value(
        self,
        test_value: "DataFlow",
        test_span: "SpanTree",
    ) -> list[ValueEval]:
        """
        Evaluate a single value against configured evaluators.

        Args:
            test_value: The value being evaluated
            test_span: The span that contains this value

        Returns:
            List of ValueEval objects created

        Raises:
            EvaluationFailure: If an evaluator triggers early stopping
        """
        # Get reference value at same path by searching reference span trees
        ref_value = None
        for ref_span_tree in self.reference_span_trees:
            ref_value = ref_span_tree.get_value_by_path(test_value.path)
            if ref_value:
                break

        # Match value to evaluators
        matched_evaluators = match_value_to_evaluators(
            test_value,
            test_span,
            self.test_turn.evaluators
        )

        if not matched_evaluators:
            return []

        # If no reference value, skip evaluation
        if ref_value is None:
            return []

        # Run each matching evaluator
        evals_created = []

        for evaluator_config in matched_evaluators:
            # Get evaluator function
            evaluator_fn = get_evaluator(evaluator_config.name)
            if not evaluator_fn:
                # Evaluator not registered, skip
                continue

            # Create eval context
            eval_ctx = EvalCtx(
                test_case=self.test_case,
                test_run=self.test_run,
                config=evaluator_config
            )

            try:
                # Run evaluator
                score, metadata = await evaluator_fn(eval_ctx, ref_value, test_value)

                # Create evaluation record
                value_eval = ValueEval(
                    turn_eval_id=self.turn_eval.id,
                    value_id=test_value.id,
                    path=test_value.path,
                    evaluator=evaluator_config.name,
                    score=score,
                    metadata=metadata,
                    status="completed"
                )

                # Save to database
                await value_eval.save()

                # Track result
                evals_created.append(value_eval)
                self.value_evals.append(value_eval)
                self.results.append({
                    "evaluator": evaluator_config.name,
                    "score": score,
                    "path": test_value.path,
                    "metadata": metadata
                })

            except EvaluationFailure as e:
                # Early stopping requested
                value_eval = ValueEval(
                    turn_eval_id=self.turn_eval.id,
                    value_id=test_value.id,
                    path=test_value.path,
                    evaluator=evaluator_config.name,
                    score=None,
                    metadata={"early_stop": True},
                    status="failed",
                    error=str(e)
                )
                await value_eval.save()
                evals_created.append(value_eval)
                self.value_evals.append(value_eval)

                # Re-raise to stop execution
                raise

            except Exception as e:
                # Evaluator error - log but don't stop execution
                value_eval = ValueEval(
                    turn_eval_id=self.turn_eval.id,
                    value_id=test_value.id,
                    path=test_value.path,
                    evaluator=evaluator_config.name,
                    score=None,
                    metadata={},
                    status="failed",
                    error=str(e)
                )
                await value_eval.save()
                evals_created.append(value_eval)
                self.value_evals.append(value_eval)

        return evals_created

    def get_results(self) -> list[dict[str, Any]]:
        """Get all evaluation results collected so far."""
        return self.results

    def get_average_score(self) -> float | None:
        """Calculate average score across all evaluations."""
        scores = [r["score"] for r in self.results if r.get("score") is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)
