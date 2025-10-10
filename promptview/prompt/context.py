import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, AsyncGenerator, Iterator, Type

from pydantic import BaseModel
from ..auth.user_manager2 import AuthModel
from ..model.model3 import Model
from ..model.postgres2.pg_query_set import PgSelectQuerySet
from ..model.versioning.models import Branch, ExecutionSpan, SpanTypeEnum, Turn, TurnStatus, VersionedModel
from dataclasses import dataclass
from ..utils.function_utils import call_function
if TYPE_CHECKING:
    from fastapi import Request
    from .span_tree import SpanTree

# Context variable for implicit context passing across async boundaries
_context_var: ContextVar["Context | None"] = ContextVar('context', default=None)



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


class ContextError(Exception):
    pass



class Context(BaseModel):
    branch_id: int | None = None
    turn_id: int | None = None
    _request: "Request | None" = None
    _auth: AuthModel | None = None
    _ctx_models: dict[str, Model] = {}
    _tasks: list[LoadBranch | LoadTurn | ForkTurn | StartTurn] = []
    _span_tree: "SpanTree | None" = None
    _execution_stack: list = []
    _replay_span_tree: "SpanTree | None" = None  # Span tree to replay from
    _execution_path: list[int] = []  # Current execution path like [0, 1, 2]
    
    
    def __init__(
        self,
        *models: Model,
        branch: Branch | None = None,
        turn: Turn | None = None,
        branch_id: int | None = None,
        turn_id: int | None = None,
        request: "Request | None" = None,
        auth: AuthModel | None = None,
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
        self._root_span: "SpanTree | None" = None
        
    @property
    def request_id(self):
        if self._request is not None:
            return self._request.state.request_id
        return None

    @classmethod
    def current(cls) -> "Context | None":
        """
        Get the current context from ContextVar.

        Returns:
            The current Context instance, or None if no context is active
        """
        return _context_var.get()

    @property
    def current_component(self):
        """Get the current component on the execution stack."""
        return self._execution_stack[-1] if self._execution_stack else None

    @property
    def current_span_tree(self) -> "SpanTree | None":
        """Get the current span tree from the current component."""
        if self.current_component and hasattr(self.current_component, '_span_tree'):
            return self.current_component._span_tree
        return None

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

    def _get_replay_span_at_path(self, path: list[int]) -> "SpanTree | None":
        """
        Look up a span in the replay tree at the given path.

        Path format matches SpanTree.path:
        - Root: [0]
        - First child: [0, 0]
        - Second child: [0, 1]

        Args:
            path: Execution path like [0], [0, 0], [0, 1]

        Returns:
            SpanTree at that path, or None if not found
        """
        if not self._replay_span_tree:
            return None

        # Check if looking for root
        if path == [0]:
            return self._replay_span_tree

        # Navigate down the tree using path indices
        # Start at root (path[0] = 0), then navigate children
        current = self._replay_span_tree

        # Skip first index (root is [0]), navigate using remaining indices
        for index in path[1:]:
            if not current.children or index >= len(current.children):
                return None
            current = current.children[index]

        return current

    async def start_span(
        self,
        component,
        name: str,
        span_type: str = "component",
        tags: list[str] | None = None
    ) -> "SpanTree":
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
        from .span_tree import SpanTree

        tags = tags or []

        # Check if we should use a span from replay tree
        replay_span = None
        if self._replay_span_tree:
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
        elif not self._execution_stack:
            # Root component - create root span tree
            if self._span_tree is None:
                self._span_tree = await SpanTree.init_new(
                    name=name,
                    span_type=span_type,
                    tags=tags
                )
            span_tree = self._span_tree
        else:
            # Child component - create child span
            parent_span = self.current_span_tree
            if parent_span is None:
                raise ValueError("Parent component has no span tree")

            span_tree = await parent_span.add_child(
                name=name,
                span_type=span_type,
                tags=tags
            )

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

        if self._root_span is None:
            self._root_span = span_tree
        return span_tree

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
    
    def get_span(self, path: list[int]) -> "SpanTree | None":
        if self._root_span is None:
            return None
        return self._root_span.get(path)

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
    
    def fork(self, turn: Turn | None = None, turn_id: int | None = None) -> "Context":
        self._tasks.append(ForkTurn(turn=turn, turn_id=turn_id))
        return self

    async def load_replay(self, turn_id: int, span_id: int | None = None, branch_id: int | None = None) -> "Context":
        """
        Load a span tree for replay mode.

        When a span tree is loaded, Context will use it to provide saved spans
        during execution instead of creating new ones. This enables replay of
        previous executions.

        Args:
            turn_id: The turn ID to load spans from
            span_id: Optional specific span to start from (default: root span)

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
        self._replay_span_tree = await SpanTree.replay_from_turn(turn_id, span_id, branch_id)
        return self

    # def fork(self, branch: Branch | None = None)
    
    async def _handle_tasks(self) -> Branch | None:
        for task in self._tasks:
            if isinstance(task, LoadBranch):
                self._branch = await Branch.get(task.branch_id)
            elif isinstance(task, LoadTurn):
                self._turn = await Turn.get(task.turn_id)
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
            elif isinstance(task, StartTurn):
                branch = await self._get_branch()
                self._turn = await branch.create_turn(auto_commit=task.auto_commit)


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
                
        
    async def __aenter__(self):
        # Set this context as the current context
        _context_var.set(self)

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
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
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