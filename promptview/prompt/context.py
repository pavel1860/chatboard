import asyncio
from contextlib import asynccontextmanager
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
    

class Context(BaseModel):
    branch_id: int | None = None
    turn_id: int | None = None
    _request: "Request | None" = None
    _auth: AuthModel | None = None
    _ctx_models: dict[str, Model] = {}
    _tasks: list[LoadBranch | LoadTurn | ForkTurn | StartTurn] = []
    
    
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
        
    @property
    def request_id(self):
        if self._request is not None:
            return self._request.state.request_id
        return None

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
    
    @asynccontextmanager
    async def start_span(self, name: str, index: int, span_type: SpanTypeEnum = "component", tags: list[str] | None = None) -> AsyncGenerator["ExecutionSpan", None]:
        span = await ExecutionSpan(name=name, span_type=span_type, tags=tags, index=index).save()
        try:    
            with span:
                yield span
            span.status = "completed"
            await span.save()
        except Exception as e:
            span.status = "failed"
            await span.save()
            raise e
        finally:
            await span.save()

        
        
    
    # def fork(self, branch: Branch | None = None)
    
    async def _handle_tasks(self) -> Branch:
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
            

        if self._branch is None:
            branch = await self._get_branch()
        # if self.turn is None:
            # raise ValueError("Turn not found")
                
        return self.branch
    
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
        if self._auth is not None:
            auth = self._auth.__enter__()
        branch = await self._handle_tasks()
        v_models, models = self.get_models()
        for model in models:
            model.__enter__()
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
        self.branch.__exit__(exc_type, exc_value, traceback)
        for model in reversed(models):
            model.__exit__(exc_type, exc_value, traceback)
        if self._auth is not None:
            self._auth.__exit__(exc_type, exc_value, traceback)
              
            
            
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