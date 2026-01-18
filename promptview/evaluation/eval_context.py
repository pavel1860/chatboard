"""Evaluation context for tracking evaluations during execution."""

from typing import TYPE_CHECKING, Any, Callable
from dataclasses import dataclass, field

# from .models import EvaluatorConfig, ValueEval, EvaluationFailure
from ..versioning.models import Branch, EvaluatorConfig, ValueEval, EvaluationFailure
from .decorators import get_evaluator, EvalCtx, evaluator_registry
from .matching import match_value_to_evaluators
from ..versioning.models import TestCase, TestRun, TurnEval, TestTurn, Turn, DataFlowNode
from ..versioning.artifact_log import ArtifactLog
if TYPE_CHECKING:
    from ..prompt.span_tree import DataFlow, SpanTree
    from ..prompt.fbp_process import EvaluatorController
    from ..auth import AuthModel
    # from ..model import TestCase, TestRun, TurnEval, TestTurn, Turn


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

    # test_case: "TestCase"
    # test_run: "TestRun"
    # turn_eval: "TurnEval"
    # test_turn: "TestTurn"
    # test_case_id: int
    # test_run_id: int | None

    # Reference span trees (for looking up values by path)
    auth: "AuthModel"
    reference_turns: list["Turn"] = field(default_factory=list)
    # Track evaluation results
    results: list[dict[str, Any]] = field(default_factory=list)
    value_evals: list[ValueEval] = field(default_factory=list)    
    _did_start: bool = field(default=False)
    
    # def __init__(self, test_case: "TestCase", test_run: "TestRun"):
    @property
    def current_turn_eval(self) -> "TurnEval":
        if self.test_run is None or not self.test_run.turn_evals:
            raise ValueError("No turn evals found")
        return self.test_run.turn_evals[-1]
    
    
    @property
    def did_start(self) -> bool:
        return self._did_start
    
    
    
    async def init(self, test_case_id: int, test_run_id: int | None = None):
        self.test_case = await TestCase.query().where(id=test_case_id).include(
                TestTurn.query()
                    .include(EvaluatorConfig)
                    # .include(Turn.query().include(DataFlowNode))
                    .include(Turn.query(include_executions=True))
                ).one()
        if self.test_case is None:
            raise ValueError(f"Test case not found for id {test_case_id}")
        turns = [t.turn for t in self.test_case.test_turns]
        turns = await ArtifactLog.populate_turns(turns)
        if test_run_id is not None:
            self.test_run = await TestRun.query().where(id=test_run_id).one()
        else:
            self.test_run = await TestRun(
                test_case_id=test_case_id,
                status="running"
            ).save()
        # for test_turn in self.test_case.test_turns:
        #     turn_eval = await test_run.add(
        #         TurnEval(
                    
        #         )
        #     )
        self.current_test_turn = None
        self.current_test_turn_index = 0
        self.source_branch = await Branch.get(self.test_case.branch_id)
        fork_turn = turns[0]
        self.branch = await self.source_branch.fork_branch(fork_turn)
        return self
    
    
    
    
    
    def get_next_test(self) -> "TestTurn | None":
        if not self.did_start:
            self._did_start = True            
        if self.current_test_turn is None:
            self.current_test_turn = self.test_case.test_turns[0]
            self.current_test_turn_index = 1
        else:
            if self.current_test_turn_index >= len(self.test_case.test_turns):
                return None
            self.current_test_turn = self.test_case.test_turns[self.current_test_turn_index + 1]
            self.current_test_turn_index += 1
        return self.current_test_turn
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        res = self.get_next_test()
        if res is None:
            raise StopIteration
        return res
    
    
    def get_eval_turn_inputs(self) -> list[Any]:
        turn = self.reference_turns[0]
        return [d.value for d in turn.data["1.0.*"]]

    
    def get_eval_turn(self, path: list[int]) -> "SpanTree":
        span_tree = self.reference_turns[path[0]].get(path[1:])
        if span_tree is None:
            raise ValueError(f"Span tree not found for path {path}")
        return span_tree
    
    def get_ref_value(self, path: list[int]) -> "DataFlow":
        value = self.current_test_turn.turn.data[path]
        return value[0]
        
    
    
    def get_evaluators(self, value: "DataFlow") -> list["EvaluatorConfig"]:
        if self.current_test_turn is None:
            raise ValueError("Current test turn not set")
        return self.current_test_turn.evaluators[f"{value.path}.test.*"]
        return match_value_to_evaluators(
            value,
            self.reference_turns[0],
            self.test_turn.evaluators
        )
        
        
    def get_evaluator_handlers(self, value: "DataFlowNode") -> list[tuple[Callable, EvalCtx]]:
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
    
    
    async def build_turn_eval_from_current(self):
        if not self.current_test_turn:
            raise ValueError("Current test turn not set")
        turn_eval = await self.test_run.add(TurnEval(
            test_turn_id=self.current_test_turn.id,
            ref_turn_id=self.current_test_turn.turn.id,
            score=None,
            value_evals=self.value_evals,
        ))
        return turn_eval
    
    
    async def log_eval(self, value: "DataFlowNode", score: float, metadata: dict, evaluator_config: "EvaluatorConfig"):
        value_eval = await self.current_turn_eval.add(
            ValueEval(
                value_id=value.id,
                path=value.path,
                evaluator=evaluator_config.name,
                score=score,
                metadata=metadata,
            )
        )
        self.value_evals.append(value_eval)
        self.results.append({
            "evaluator": evaluator_config.name,
            "score": score,
            "path": value.path,
            "metadata": metadata
        })
        return value_eval
    
    
    async def commit_test_turn(self):
        if not self.current_test_turn:
            raise ValueError("Current test turn not set")
        self.current_turn_eval.score = self.get_average_score()
        await self.current_turn_eval.save()
        
    
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
        for ref_span_tree in self.reference_turns:
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
        # scores = [r["score"] for r in self.results if r.get("score") is not None]
        scores = [v.score for v in self.current_turn_eval.value_evals if v.score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)
