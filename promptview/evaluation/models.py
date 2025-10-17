"""Models for evaluation system."""

from typing import TYPE_CHECKING, List, Literal
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from ..model import Model, ModelField, KeyField, RelationField, SpanValue, Turn



class EvaluationFailure(Exception):
    """Exception raised when an evaluation fails and should stop execution."""
    pass


class EvaluatorConfig(BaseModel):
    """
    Configuration for a value evaluator.

    Evaluators can match values by:
    - path_pattern: LTREE path pattern for SpanValue.path (e.g., "1.*", "*.0", "1.2.3")
    - tags: List of tags that the parent span must have
    - span_name: Exact name match for the parent span
    - value_name: Match values by their name field

    All criteria are AND-ed together (value must match all specified criteria).
    """
    name: str = Field(..., description="Evaluator function name")
    path_pattern: str | None = Field(None, description="LTREE path pattern for SpanValue.path (e.g., '1.*', '*.0', '1.2.3')")
    tags: list[str] = Field(default=[], description="Match values whose parent span has these tags")
    # span_name: str | None = Field(None, description="Match values whose parent span has this name")
    # value_name: str | None = Field(None, description="Match values by their name field")
    metadata: dict = Field(default={}, description="Additional evaluator configuration")


class ValueEval(Model):
    """
    Result of evaluating a single value.

    Each value evaluation records:
    - Which value was evaluated (value_id, path)
    - Which evaluator was used (evaluator name)
    - The score and metadata from the evaluator
    - Link to the parent turn evaluation
    """
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)

    # What was evaluated
    turn_eval_id: int = ModelField(foreign_key=True, description="Parent turn evaluation")
    value_id: int = ModelField(foreign_key=True, foreign_cls=SpanValue, description="The SpanValue that was evaluated")
    path: str = ModelField(db_type="LTREE", description="Value path for querying")

    # Evaluation results
    evaluator: str = ModelField(description="Evaluator function name")
    score: float | None = ModelField(default=None, description="Evaluation score")
    metadata: dict = ModelField(default={}, description="Additional evaluation data")

    # Status
    status: str = ModelField(default="completed", description="Status: completed, failed, skipped")
    error: str | None = ModelField(default=None, description="Error message if evaluation failed")


class TurnEval(Model):
    """
    Evaluation results for a single turn in a test run.

    Links a test turn to a reference turn and stores all value evaluations.
    """
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)

    test_turn_id: int = ModelField(description="The test turn being evaluated")
    ref_turn_id: int = ModelField(description="The reference turn to compare against")
    test_run_id: int = ModelField(foreign_key=True, description="Parent test run")

    score: float | None = ModelField(default=None, description="Average score across all value evaluations")
    value_evals: List[ValueEval] = RelationField(foreign_key="turn_eval_id")
    trace_id: str = ModelField(default="", description="Trace ID for debugging")


class TestRun(Model):
    """
    A single execution of a test case.

    Tracks the overall status and score of running a test case.
    """
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)

    test_case_id: int = ModelField(foreign_key=True, description="Parent test case")
    branch_id: int = ModelField(default=1, description="Branch this test run belongs to")

    score: float | None = ModelField(default=None, description="Overall score of the test run")
    status: Literal["running", "success", "failure"] = ModelField(
        default="running",
        description="Status of the test run"
    )

    turn_evals: List[TurnEval] = RelationField(foreign_key="test_run_id")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.status = "failure"
        else:
            self.status = "success"
        await self.save()


class TestTurn(Model):
    """
    A reference turn with evaluator configurations.

    Maps a reference turn to evaluators that should run during test execution.
    """
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)

    test_case_id: int = ModelField(foreign_key=True, description="Parent test case")
    turn_id: int = ModelField(foreign_key=True, foreign_cls=Turn, description="Reference turn ID")

    evaluators: List[EvaluatorConfig] = ModelField(
        default=[],
        description="Evaluators to run for values in this turn"
    )


class TestCase(Model):
    """
    A test case containing reference turns and evaluator configurations.

    Test cases define what to test and how to evaluate it.
    """
    id: int = KeyField(primary_key=True)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    updated_at: datetime = ModelField(default_factory=datetime.now)

    title: str = ModelField(default="", description="Test case title")
    description: str = ModelField(default="", description="Test case description")
    # branch_id: int = ModelField(default=1, foreign_key=True, description="Branch this test case belongs to", foreign_cls=Branch)
    user_id: UUID = ModelField(description="User who created this test case")

    test_turns: List[TestTurn] = RelationField(foreign_key="test_case_id")
    test_runs: List[TestRun] = RelationField(foreign_key="test_case_id")
