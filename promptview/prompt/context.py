import asyncio
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, AsyncGenerator, Iterator, Literal, Type

from fastapi import UploadFile
from pydantic import BaseModel
from ..auth.user_manager2 import AuthModel
from ..model.model3 import Model
from ..model.postgres2.pg_query_set import PgSelectQuerySet
from ..versioning.models import Branch, ExecutionSpan, SpanType, Turn, TurnStatus, ValueIOKind, VersionedModel, Artifact
from dataclasses import dataclass
from .events import StreamEvent
from ..utils.function_utils import call_function

from ..block import Block

if TYPE_CHECKING:
    from fastapi import Request
    from .span_tree import SpanTree
    from ..evaluation.eval_context import EvaluationContext

# Context variable for implicit context passing across async boundaries
_context_var: ContextVar["Context | None"] = ContextVar('context', default=None)

CACHE_DIR = os.getenv("CACHE_DIR", None)

@dataclass
class LoadTurn:
    turn_id: int

@dataclass
class LoadBranch:
    branch_id: int


@dataclass
class ForkTurn:
    turn: Turn | None = None
    turn_id: int | None = None

@dataclass
class ForkBranch:
    branch_id: int


@dataclass
class StartTurn:
    branch_id: int | None = None
    auto_commit: bool = True

@dataclass
class StartEval:
    test_case_id: int
    test_run_id: int | None = None

class ContextError(Exception):
    pass



class Context(BaseModel):
    branch_id: int | None = None
    turn_id: int | None = None
    cache_dir: str | None = None
    _request: "Request | None" = None
    _auth: AuthModel | None = None
    _ctx_models: dict[str, Model] = {}
    _tasks: list[LoadBranch | LoadTurn | ForkTurn | StartTurn | StartEval] = []
    _execution_stack: list = []
    _replay_span_trees: list["ExecutionSpan"] = []  # Top-level span trees to replay from
    _execution_path: list[int] = []  # Current execution path like [0, 1, 2]
    _top_level_span_count: int = 1  # Counter for top-level spans in current turn
    _top_level_spans: list["ExecutionSpan"] = []  # All top-level spans in this turn
    _evaluation_context: "EvaluationContext | None" = None  # Evaluation context if in eval mode
    _index: int = 0
    events: list[StreamEvent] = []
    message: "Block | None" = None
    state: dict | None = None
    
    
    def __init__(
        self,
        *models: Model,
        branch: Branch | None = None,
        turn: Turn | None = None,
        branch_id: int | None = None,
        turn_id: int | None = None,
        message: "Block | None" = None,
        state: dict | None = None,        
        request: "Request | None" = None,
        auth: AuthModel | None = None,
        eval_ctx: "EvaluationContext | None" = None,
        verbose: bool = False,  
        cache_dir: str | None = None,      
    ):
        super().__init__()
        self._ctx_models = {m.__class__.__name__:m for m in models}
        self._tasks = []
        self.branch_id = branch_id
        self.turn_id = turn_id
        self._request = request
        if branch_id is not None:
            self._tasks.append(LoadBranch(branch_id=branch_id))
        if turn_id is not None:
            self._tasks.append(LoadTurn(turn_id=turn_id))
            
        self._branch = branch
        self._turn = turn
        self._auth = auth
        self._initialized = False
        self._evaluation_context = eval_ctx
        self._index = 0
        self.events = []
        self.message = message
        self.state = state
        self._verbose = verbose
        self.cache_dir = cache_dir or CACHE_DIR
        
    @property
    def request_id(self):
        if self._request is not None:
            return self._request.state.request_id
        return None

    @classmethod
    def current_or_none(cls) -> "Context | None":
        """
        Get the current context from ContextVar.

        Returns:
            The current Context instance, or None if no context is active
        """
        return _context_var.get()
    
    
    @classmethod
    def current(cls) -> "Context":
        ctx = cls.current_or_none()
        if ctx is None:
            raise ValueError("Context not set")
        return ctx
    
    @property
    def eval_ctx(self) -> "EvaluationContext":
        if self._evaluation_context is None:
            raise ValueError("Evaluation context not set")
        return self._evaluation_context

    @property
    def current_component(self):
        """Get the current component on the execution stack."""
        return self._execution_stack[-1] if self._execution_stack else None

    @property
    def current_span(self) -> "ExecutionSpan | None":
        """Get the current span tree from the current component."""
        if self.current_component and hasattr(self.current_component, '_span_tree'):
            return self.current_component._span_tree
        return None
    
    
    

    @property
    def top_level_spans(self) -> list["ExecutionSpan"]:
        """Get all top-level spans created in this turn."""
        return self._top_level_spans

    @property
    def root_span(self) -> "ExecutionSpan | None":
        """
        Get the first top-level span (for backward compatibility).
        If you have multiple top-level spans, use top_level_spans instead.
        """
        return self._top_level_spans[0] if self._top_level_spans else None

    def _get_current_execution_path(self) -> list[int]:
        """
        Get the execution path for the NEXT component about to be created.

        This is used for replay span lookup. The path represents where the
        new component will be in the tree:
        - Root component: [0]
        - First child of root: [0, 0]
        - Second child of root: [0, 1]

        Returns:
            List of indices representing the next component's position
        """
        if not self._execution_stack:
            # Creating root component
            return [0]

        # Creating child component - build path from parent's path + child index
        parent = self._execution_stack[-1]
        if not hasattr(parent, '_child_count'):
            parent._child_count = 0

        # Parent's current path + next child index
        parent_path = self._execution_path.copy()
        child_index = parent._child_count

        return parent_path + [child_index]

    def _get_replay_span_at_path(self, path: list[int]) -> "ExecutionSpan | None":
        """
        Look up a span in the replay tree at the given path.

        Path format:
        - [0] = first top-level span
        - [1] = second top-level span
        - [0, 0] = first child of first top-level span
        - [0, 1] = second child of first top-level span

        Args:
            path: Execution path like [0], [1], [0, 0], [0, 1]

        Returns:
            SpanTree at that path, or None if not found
        """
        if not self._replay_span_trees or not path:
            return None

        # First index is the top-level span index
        if path[0] >= len(self._replay_span_trees):
            return None

        top_level_span = self._replay_span_trees[path[0]]

        # If path has only one element, return the top-level span
        if len(path) == 1:
            return top_level_span

        # Navigate down the tree using remaining path indices
        current = top_level_span
        for index in path[1:]:
            if not current.children or index >= len(current.children):
                return None
            current = current.children[index]

        return current

    def get_next_top_level_span_index(self) -> int:
        """
        Get the next index for a top-level span in the current turn.

        Returns:
            The next span index (starting from 0)
        """
        index = self._top_level_span_count
        self._top_level_span_count += 1
        return index


    async def start_span(
        self,
        component,
        name: str,
        span_type: str = "component",
        tags: list[str] | None = None
    ) -> "DataFlowNode":
        """
        Start a new span for a component and add to execution stack.

        In replay mode, checks if a span exists in the replay tree at the current
        execution path. If found and names match, uses the existing span (which
        contains saved outputs). Otherwise creates a new span.

        Args:
            component: The process/component to start
            name: Name for the span
            span_type: Type of span (component, stream, etc.)
            tags: Optional tags for the span

        Returns:
            SpanTree instance for this component (either from replay or newly created)
        """
        from ..model.versioning.artifact_log import ArtifactLog
        tags = tags or []

        # Check if we should use a span from replay tree
        replay_span = None
        if self._replay_span_trees:
            # Get current execution path
            current_path = self._get_current_execution_path()

            # Look up span at this path in replay tree
            replay_span = self._get_replay_span_at_path(current_path)

            # Verify span name matches (safety check)
            if replay_span and replay_span.name != name:
                replay_span = None  # Name mismatch, don't use replay span

        # Use replay span if available, otherwise create new span

        if replay_span:
            # Replay mode: use existing span with saved outputs
            span_tree = replay_span
            data_flow = span_tree
        elif not self._execution_stack:
            # Top-level component - create top-level span (no root span)
            # Get next top-level span index from context counter
            span_index = self.get_next_top_level_span_index()
            top_level_path = str(span_index)  # "0", "1", "2", ...

            # Create ExecutionSpan directly
            span_tree = await ExecutionSpan(
                name=name,
                span_type=span_type,
                tags=tags,
                path=top_level_path,
                parent_span_id=None  # Top-level, no parent
            ).save()
            
            data_flow = await ArtifactLog.log_value(span_tree, ctx=self)

            # Add to top-level spans list
            self._top_level_spans.append(span_tree)
        else:
            # Child component - create child span
            parent_span = self.current_span
            if parent_span is None:
                raise ValueError("Parent component has no span tree")

            # span_tree = await parent_span.add_child(
            #     name=name,
            #     span_type=span_type,
            #     tags=tags
            # )
            span_tree = await ExecutionSpan(
                name=name,
                span_type=span_type,
                tags=tags,
                path=parent_span.path + f".{len(parent_span.outputs) + 1}",
                parent_span_id=parent_span.id
            ).save()
            data_flow = await ArtifactLog.log_value(span_tree, ctx=self)

        # Attach span_tree to component
        component._span_tree = span_tree

        # Push component onto execution stack and update path
        self._execution_stack.append(component)

        # Update execution path to match SpanTree path format
        # Root: [0], First child: [0, 0], Second child: [0, 1], etc.
        if len(self._execution_stack) == 1:
            # Root component - path is [0]
            self._execution_path = [0]
            component._child_count = 0
        else:
            # Child component - append child index to parent's path
            parent = self._execution_stack[-2]
            if not hasattr(parent, '_child_count'):
                parent._child_count = 0

            # Build path: parent's path + child index
            parent_path = self._execution_path[:-1] if len(self._execution_path) > 0 else [0]
            child_index = parent._child_count
            self._execution_path = parent_path + [child_index]

            parent._child_count += 1

        return data_flow
    
    
    def push_event(self, path: str, kind: str, payload: Any, name: str | None = None, error: str | None = None) -> StreamEvent:
        event = self.build_event(path=path, kind=kind, payload=payload, error=error)        
        self.events.append(event)
        return event
    
    
    def build_event(self, path: str, kind: str, payload: Any, name: str | None = None, error: str | None = None) -> StreamEvent:        
        index = self._index
        self._index += 1
        return StreamEvent(
            type=kind,
            payload=payload,
            path=path,
            name=name,
            request_id=self.request_id,
            index=index,
            error={"message": error} if error else None,
        )

    async def end_span(self):
        """
        End the current span and pop component from execution stack.

        Also updates the execution path to reflect the new stack depth.

        Returns:
            The component that was popped
        """
        if not self._execution_stack:
            raise ValueError("No component on execution stack to complete")

        component = self._execution_stack.pop()

        # Update execution path: pop last element when leaving a child
        if self._execution_path:
            self._execution_path.pop()

        return component
    
    def get_span(self, path: list[int]) -> "ExecutionSpan | None":
        """
        Get a span by its full path from the top-level spans.
        Path format: [top_level_index, child_index, child_child_index, ...]
        Examples:
            [0] = first top-level span
            [1] = second top-level span
            [2, 1, 0] = first child of second child of third top-level span
        """
        if not path or not self._top_level_spans:
            return None

        # First index is the top-level span index
        if path[0] >= len(self._top_level_spans):
            return None

        top_level_span = self._top_level_spans[path[0]]

        # If path has more elements, navigate down the tree
        if len(path) == 1:
            return top_level_span
        else:
            # Use SpanTree.get() which expects relative path from this span
            return top_level_span.get(path[1:])

    @classmethod
    async def from_request(cls, request: "Request"):
        from ..api.utils import get_request_ctx, get_auth
        # ctx_args = request.state.get("ctx")
        # if ctx_args is None:
        #     raise ValueError("ctx is not set")
        # ctx = cls(**ctx_args)
        ctx_kwargs = get_request_ctx(request)
        auth = await get_auth(request)
        # ctx = cls(**ctx, auth=auth)
        ctx = await call_function(cls, **ctx_kwargs, auth=auth)
        return ctx
    
    @classmethod
    async def from_kwargs(cls, **kwargs):
        ctx = cls(**kwargs)    
        return ctx

        
    @property
    def branch(self) -> Branch:
        if self._branch is None:
            raise ValueError("Branch not found")
        return self._branch
    
    @property
    def turn(self) -> Turn:
        if self._turn is None:
            raise ValueError("Turn not found")
        return self._turn
    
    async def get_branch(self) -> Branch:
        return await Branch.get_main()
        
    
    async def _get_branch(self) -> Branch:
        if self._branch is None:
            self._branch = await Branch.get_main()
        return self._branch
    
        # if self._branch is not None:
        #     return self.branch
        # elif self.branch_id is not None:
        #     self._branch = await Branch.get(self.branch_id)
        #     return self.branch
        # else:
        #     self._branch = await self.get_branch()
        #     return self.branch
        
    
    def start_turn(self, auto_commit: bool = True) -> "Context":
        self._tasks.append(StartTurn(auto_commit=auto_commit))
        return self

    def start_eval(self, test_case_id: int, test_run_id: int | None = None) -> "Context":
        """
        Start evaluation mode for this context.

        This loads a test case and sets up evaluation context to automatically
        run evaluators as values are logged during execution.

        Args:
            test_case_id: The test case ID to evaluate against

        Returns:
            Self for method chaining

        Example:
            ctx = Context()
            async with ctx.start_eval(test_case_id=1):
                result = await my_agent("test input")
        """
        self._tasks.append(StartEval(test_case_id=test_case_id, test_run_id=test_run_id))
        self._tasks.append(StartTurn())
        return self

    def fork(self, turn: Turn | None = None, turn_id: int | None = None) -> "Context":
        self._tasks.append(ForkTurn(turn=turn, turn_id=turn_id))
        return self
    
    async def _setup_evaluation(self, test_case_id: int, test_run_id: int | None = None):
        from ..evaluation import EvaluationContext
        self._evaluation_context = await EvaluationContext().init(test_case_id=test_case_id, test_run_id=test_run_id)
        return self._evaluation_context
    
    
    async def iter_replay(self):
        if not self._evaluation_context:
            raise ValueError("Evaluation context not setup")
        return self._evaluation_context
    # async def _setup_evaluation(self, test_case_id: int, test_run_id: int | None = None):
    #     """
    #     Setup evaluation context for the given test case.

    #     Loads reference data and creates evaluation tracking objects.
    #     """
    #     from ..evaluation import EvaluationContext
    #     from ..model import TestCase, TestRun, TurnEval, TestTurn, Turn, DataFlowNode
    #     from .span_tree import SpanTree

    #     # Load test case
    #     test_case = await TestCase.get(test_case_id)

    #     # Get first test turn (for MVP, support single turn)
    #     test_turns = await TestTurn.query().where(test_case_id=test_case_id).execute()
    #     if not test_turns:
    #         raise ValueError(f"No test turns found for test case {test_case_id}")

    #     test_turn = test_turns[0]

    #     # Load reference turn and span trees
    #     ref_turns = await Turn.query(include_executions=True).where(Turn.id.isin([turn.turn_id for turn in test_turns]))
    #     if not isinstance(ref_turns, list):
    #         ref_turns = [ref_turns]

    #     if test_run_id is not None:
    #         test_run = await TestRun.get(test_run_id)
    #         if test_run is None:
    #             raise ValueError(f"Test run {test_run_id} not found")
    #         if test_run.status != "pending":
    #             raise ValueError(f"Test run {test_run_id} is not pending. start a new test run to evaluate this test case.")
    #     else:
    #     # Create test run
    #         test_run = await TestRun(
    #             test_case_id=test_case_id,
    #             status="running"
    #         ).save()

    #     # Create turn evaluation
    #     turn_eval = await TurnEval(
    #         test_turn_id=test_turn.id,
    #         ref_turn_id=test_turn.turn_id,
    #         test_run_id=test_run.id
    #     ).save()

    #     # Create evaluation context
    #     # Instead of building flat index, just pass the reference span trees
    #     # EvaluationContext will use get_value_by_path() on-demand
    #     eval_ctx = EvaluationContext(
    #         test_case=test_case,
    #         test_run=test_run,
    #         turn_eval=turn_eval,
    #         test_turn=test_turn,
    #         reference_turns=ref_turns
    #     )

    #     # Store in context
    #     self._evaluation_context = eval_ctx

    async def load_replay(self, turn_id: int, span_id: int | None = None, branch_id: int | None = None) -> "Context":
        """
        Load span trees for replay mode.

        When span trees are loaded, Context will use them to provide saved spans
        during execution instead of creating new ones. This enables replay of
        previous executions.

        Args:
            turn_id: The turn ID to load spans from
            span_id: Optional specific span to start from (default: all top-level spans)
            branch_id: Optional branch ID for the turn

        Returns:
            Self for method chaining

        Example:
            ctx = Context()
            async with ctx.load_replay(turn_id=42).fork().start_turn():
                # Execution will replay from turn 42's spans
                async for event in my_component().stream():
                    print(event)
        """
        from .span_tree import SpanTree
        self._replay_span_trees = await SpanTree.replay_from_turn(turn_id, span_id, branch_id)
        return self

    # def fork(self, branch: Branch | None = None)
    
    async def _handle_tasks(self) -> Branch | None:
        for task in self._tasks:
            if isinstance(task, LoadBranch):
                self._branch = await Branch.get(task.branch_id)
                self.push_event(path="0", kind="branch_load", payload=self._branch)
            elif isinstance(task, LoadTurn):
                self._turn = await Turn.get(task.turn_id)
                self.push_event(path="0", kind="turn_load", payload=self._turn)
            elif isinstance(task, ForkTurn):
                if task.turn is not None:
                    branch = await self._get_branch()
                    self._branch = await branch.fork_branch(task.turn)
                elif task.turn_id is not None:
                    branch = await self._get_branch()
                    turn = await Turn.get(task.turn_id)
                    self._branch = await branch.fork_branch(turn)
                else:
                    branch = await self._get_branch()
                    turn = await Turn.query().where(branch_id=branch.id).last()
                    self._branch = await branch.fork_branch(turn)
                self.push_event(path="0", kind="branch_fork", payload=self._branch)
            elif isinstance(task, StartTurn):
                branch = await self._get_branch()
                self._turn = await branch.create_turn(auto_commit=task.auto_commit)
                self.push_event(path="0", kind="turn_start", payload=self._turn)
            elif isinstance(task, StartEval):
                await self._setup_evaluation(task.test_case_id, task.test_run_id)
                

        if self._branch is None and len(self._tasks) > 0:
            # Only load branch if there were tasks that require it
            branch = await self._get_branch()
        # if self.turn is None:
            # raise ValueError("Turn not found")
        self._tasks = []
        return self._branch
    
    def get_models(self):
        v_models = []
        models = []
        for model in self._ctx_models.values():
            if isinstance(model, VersionedModel):
                v_models.append(model)
            else:
                models.append(model)
        return v_models, models
    
    
    def is_set(self) -> bool:
        return _context_var.get() is not None
    
    
    async def initialize(self, safe: bool = False):
        if self._initialized:
            if safe:
                return
            raise ValueError("Context already initialized")
        if self._auth is not None:
            auth = self._auth.__enter__()
        branch = await self._handle_tasks()
        v_models, models = self.get_models()
        for model in models:
            model.__enter__()
        # Only enter branch context if we have a branch
        if branch is not None:
            branch.__enter__()
        if self._turn is not None:
            await self._turn.__aenter__()
        for model in v_models:
            model.__enter__()
        self._initialized = True

                
        
    async def __aenter__(self):
        # Set this context as the current context
        _context_var.set(self)
        await self.initialize(safe=True)

        # if self._auth is not None:
        #     auth = self._auth.__enter__()
        # branch = await self._handle_tasks()
        # v_models, models = self.get_models()
        # for model in models:
        #     model.__enter__()
        # # Only enter branch context if we have a branch
        # if branch is not None:
        #     branch.__enter__()
        # if self._turn is not None:
        #     await self._turn.__aenter__()
        # for model in v_models:
        #     model.__enter__()
        return self
    
    
    async def finalize(self, exc_type, exc_value, traceback):
        if not self._initialized:
            raise ValueError("Context not initialized")
        v_models, models = self.get_models()
        for model in reversed(v_models):
            model.__exit__(exc_type, exc_value, traceback)
        if self._turn is not None:
            await self._turn.__aexit__(exc_type, exc_value, traceback)
        # Only exit branch context if we have a branch
        if self._branch is not None:
            self._branch.__exit__(exc_type, exc_value, traceback)
        for model in reversed(models):
            model.__exit__(exc_type, exc_value, traceback)
        if self._auth is not None:
            self._auth.__exit__(exc_type, exc_value, traceback)

        
        
        
        
    async def __aexit__(self, exc_type, exc_value, traceback):
        # v_models, models = self.get_models()
        # for model in reversed(v_models):
        #     model.__exit__(exc_type, exc_value, traceback)
        # if self._turn is not None:
        #     await self._turn.__aexit__(exc_type, exc_value, traceback)
        # # Only exit branch context if we have a branch
        # if self._branch is not None:
        #     self._branch.__exit__(exc_type, exc_value, traceback)
        # for model in reversed(models):
        #     model.__exit__(exc_type, exc_value, traceback)
        # if self._auth is not None:
        #     self._auth.__exit__(exc_type, exc_value, traceback)
        await self.finalize(exc_type, exc_value, traceback)

        # Clear the context
        _context_var.set(None)
              
            
            
    def select(self, target: Type[Model] | PgSelectQuerySet[Model], fields: list[str] | None = None, alias: str | None = None) -> "PgSelectQuerySet[Model]":
        turn_cte = Turn.vquery().select("*").where(status=TurnStatus.COMMITTED)
        # query = model.query().use_cte(turn_cte, "committed_turns", alias="ct")
        if isinstance(target, PgSelectQuerySet):
            query = target 
        else:
            query = target.query().select(*fields if fields else "*")
            # query = PgSelectQuerySet(target, alias=alias) \
            #     .select(*fields if fields else "*")        
        query.use_cte(
            turn_cte,
            name="committed_turns",
            alias="ct",
        )
        return query
    
    
    
    
    def get_verbosity(self, target: Literal["parser"]) -> bool:
        return self._verbose