"""API router for test and evaluation endpoints."""

from typing import Type, List
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from .model_router import create_model_router
from ..prompt.context import Context
from ..model import Branch, TestCase, TestRun, TestTurn, TurnEval, ValueEval, Turn
from .utils import ListParams, get_list_params


def create_test_router(context_cls: Type[Context] | None = None):
    """Create router with test and evaluation endpoints."""
    context_cls = context_cls or Context

    async def get_model_ctx(request: Request):
        return await context_cls.from_request(request)

    # Create base routers for each model
    test_case_router = create_model_router(TestCase, get_model_ctx)
    test_run_router = create_model_router(TestRun, get_model_ctx)
    turn_eval_router = create_model_router(TurnEval, get_model_ctx)
    value_eval_router = create_model_router(ValueEval, get_model_ctx)

    # Main router that includes all test-related endpoints
    router = APIRouter(prefix="/tests", tags=["tests"])

    # Include sub-routers
    router.include_router(test_case_router)
    router.include_router(test_run_router)
    router.include_router(turn_eval_router)
    router.include_router(value_eval_router)


    # Additional custom endpoints

    @router.get("/TestCase/{test_case_id}")
    async def get_test_case_with_turns(
        request: Request,        
        test_case_id: int,
        include: list[str] | None = Query(default=None, alias="include"),
        ctx = Depends(get_model_ctx)
    ):
        """
        Get test case with all test turns and their evaluator configurations.

        Returns complete test case structure including:
        - Test case metadata (title, description)
        - All test turns with their reference turn IDs
        - Evaluator configurations for each turn
        """
        test_case = await (
            TestCase.query()
            .include(TestTurn)
            .where(TestCase.id == test_case_id)
            .first()
        )

        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")

        return test_case.model_dump()


    @router.get("/TestCase/{test_case_id}/runs")
    async def get_test_case_runs(
        test_case_id: int,
        list_params: ListParams = Depends(get_list_params),
        ctx = Depends(get_model_ctx)
    ):
        """
        Get all test runs for a test case.

        Returns list of test runs with:
        - Run metadata (created_at, status, score)
        - Summary of turn evaluations
        """
        test_runs = await (
            TestRun.query()
            .include(TurnEval)
            .where(TestRun.test_case_id == test_case_id)
            .order_by("-created_at")
            .limit(list_params.limit)
            .offset(list_params.offset)
        )

        return [run.model_dump() for run in test_runs]
    
    
    
    @router.post("/TestRun/create_pending")
    async def create_test_run(
        request: Request,
        ctx = Depends(get_model_ctx)
    ):
        """
        Create a new test run.
        """
        payload = await request.json()
        if not payload.get("test_case_id"):
            raise HTTPException(status_code=400, detail="test_case_id is required")
        test_case = await TestCase.query().include(TestTurn).where(TestCase.id == payload["test_case_id"]).first()
        
        turns = await Turn.query().where(Turn.id.isin([turn.turn_id for turn in test_case.test_turns])).order_by("-index")
        
        if not test_case or not turns:
            raise HTTPException(status_code=404, detail="Test case not found")
        test_case_branch = await Branch.get(test_case.branch_id)
        if not test_case_branch:
            raise HTTPException(status_code=404, detail="Branch not found")
        test_run_branch = await test_case_branch.fork_branch(turns[0])
        test_run = TestRun(
            test_case_id=test_case.id,
            branch_id=test_run_branch.id,
        )
        await test_run.save()
        return test_run.model_dump()


    @router.get("/TestRun/{test_run_id}")
    async def get_test_run_with_evaluations(
        test_run_id: int,
        ctx = Depends(get_model_ctx)
    ):
        """
        Get test run with all turn evaluations and value evaluations.

        Returns complete evaluation results including:
        - Test run metadata (status, overall score)
        - Turn evaluations with scores
        - All value evaluations for each turn
        """
        test_run = await (
            TestRun.query()
            .include(
                TurnEval.query()
                .include(ValueEval)
            )
            .where(TestRun.id == test_run_id)
            .first()
        )

        if not test_run:
            raise HTTPException(status_code=404, detail="Test run not found")

        return test_run.model_dump()


    @router.get("/TestRun/{test_run_id}/value-evals")
    async def get_test_run_value_evaluations(
        test_run_id: int,
        evaluator: str | None = Query(None, description="Filter by evaluator name"),
        min_score: float | None = Query(None, description="Filter by minimum score"),
        max_score: float | None = Query(None, description="Filter by maximum score"),
        status: str | None = Query(None, description="Filter by status (completed, failed, skipped)"),
        list_params: ListParams = Depends(get_list_params),
        ctx = Depends(get_model_ctx)
    ):
        """
        Get detailed value evaluations for a test run with filtering.

        Allows filtering by:
        - evaluator: Specific evaluator name
        - min_score/max_score: Score range
        - status: Evaluation status

        Useful for:
        - Debugging specific evaluator failures
        - Finding low-scoring values
        - Analyzing evaluation patterns
        """
        # First get all turn_eval_ids for this test run
        turn_evals = await (
            TurnEval.query()
            .where(TurnEval.test_run_id == test_run_id)
            .select("id")
        )

        turn_eval_ids = [te.id for te in turn_evals]

        if not turn_eval_ids:
            return []

        # Build value eval query with filters
        query = ValueEval.query().where(
            ValueEval.turn_eval_id.isin(turn_eval_ids)
        )

        # Apply optional filters
        if evaluator:
            query = query.where(ValueEval.evaluator == evaluator)

        if min_score is not None:
            query = query.where(ValueEval.score >= min_score)

        if max_score is not None:
            query = query.where(ValueEval.score <= max_score)

        if status:
            query = query.where(ValueEval.status == status)

        # Apply pagination and ordering
        value_evals = await (
            query
            .order_by("-created_at")
            .limit(list_params.limit)
            .offset(list_params.offset)
        )

        return [ve.model_dump() for ve in value_evals]


    @router.get("/TestCase/{test_case_id}/comparison")
    async def compare_test_runs(
        test_case_id: int,
        run_count: int = Query(default=5, ge=2, le=20, description="Number of recent runs to compare"),
        ctx = Depends(get_model_ctx)
    ):
        """
        Compare recent test runs for a test case.

        Returns:
        - List of recent runs with scores
        - Score trends over time
        - Pass/fail status for each run

        Useful for tracking test stability and performance over time.
        """
        test_runs = await (
            TestRun.query()
            .where(TestRun.test_case_id == test_case_id)
            .order_by("-created_at")
            .limit(run_count)
        )

        comparison = {
            "test_case_id": test_case_id,
            "runs": [
                {
                    "id": run.id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "status": run.status,
                    "score": run.score
                }
                for run in test_runs
            ]
        }

        # Calculate trends
        scores = [run.score for run in test_runs if run.score is not None]
        if scores:
            comparison["average_score"] = sum(scores) / len(scores)
            comparison["min_score"] = min(scores)
            comparison["max_score"] = max(scores)

        return comparison


    return router
