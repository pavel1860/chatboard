
import asyncio
import contextvars
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Literal, Type, TypeVar

from pydantic import BaseModel

from chatboard.model.namespace_manager import NamespaceManager
from chatboard.model.versioning import ArtifactLog, Partition, TurnStatus
from chatboard.tracer.langsmith_tracer import RunTypes
from ..tracer import Tracer

    

if TYPE_CHECKING:
    from chatboard.model.model import Model, ContextModel
    from chatboard.model.versioning import Branch, Turn
    from chatboard.block.block import Block, BlockList, Blockable
    
PARTITION_MODEL = TypeVar("PARTITION_MODEL", bound="Model")
CONTEXT_MODEL = TypeVar("CONTEXT_MODEL", bound="ContextModel")


CURR_CONTEXT = contextvars.ContextVar("curr_context")

class InitStrategy(StrEnum):
    START_TURN = "start_turn"    
    BRANCH_FROM = "branch_from"
    NO_PARTITION = "no_partition"
    RESUME_TURN = "resume_turn"
    
    

class Context(Generic[PARTITION_MODEL, CONTEXT_MODEL]):
    partition_model: Type[PARTITION_MODEL]
    context_model: Type[CONTEXT_MODEL]
    
    _ctx_token: contextvars.Token | None
    _branch_id: int
    _branch: "Branch | None"
    _on_exit: Literal["commit", "revert", "none"]
    _turn: "Turn | None"
    _user: PARTITION_MODEL | None
    def __init__(
        self,
        user: PARTITION_MODEL,              
        on_exit: Literal["commit", "revert", "none"], 
        branch: int | None = 1,
        filters: dict[str, Any] | None = None,       
        span_name: str | None = None,         
        state: Any | None = None
    ):
        self._user = user
        self._init_method = InitStrategy.RESUME_TURN        
        self.filters = filters or {}
        # self._init_params = {"branch_id": branch if type(branch) is int else branch.id}
        self._branch_id = branch if branch is not None else 1
        self._ctx_token = None
        self._span_name = span_name
        self._branch = None
        self._turn = None
        self._on_exit = on_exit
        self._parent_ctx = None
        self._trace_id = None
        self._state = state
        self._tracer_run = None
        
        
    async def _init(self):
        await self.get_current_head()
    
        
    @property
    def trace_id(self):  
        if self._tracer_run is not None:      
            return str(self._tracer_run.id)
        return self._trace_id
    
    @trace_id.setter
    def trace_id(self, value: str):
        self._trace_id = value
        
        
    @property
    def tracer(self):
        if self._tracer_run is None:
            raise ValueError("Tracer not set")
        return self._tracer_run
    
    @property
    def state(self):
        if self._state is None:
            raise ValueError("State not set")
        return self._state
    
    @state.setter
    def state(self, value: Any):
        self._state = value
        
    @property
    def user(self) -> PARTITION_MODEL:
        if self._user is None:
            raise ValueError("User not set")
        return self._user
        
    @classmethod
    def get_current(cls, raise_error: bool = True):
        try:            
            ctx = CURR_CONTEXT.get()
        except LookupError:
            if raise_error:
                raise ValueError("Context not set")
            else:
                return None        
        if not isinstance(ctx, Context):
            if raise_error:
                raise ValueError("Context is not a Context")
            else:
                return None
        return ctx
    
    @classmethod
    async def get_current_head(cls, turn: "int | Turn | None" = None, branch: "int | Branch | None" = None) -> tuple[int | None, int | None]: 
        ctx = Context.get_current(raise_error=False)    
        turn_id, branch_id = await asyncio.gather(
            cls.get_current_turn(turn, ctx),
            cls.get_current_branch(branch, ctx)
        )
        return turn_id, branch_id
    
    @classmethod
    async def get_current_branch(cls, branch: "int | Branch | None" = None, ctx: "Context | None" = None):
        branch_id = 1 
        if ctx is None:
            ctx = Context.get_current(raise_error=False)
        if branch:
            if isinstance(branch, int):
                branch_id = branch
            else:
                branch_id = branch.id
        else:
            if ctx:
                if ctx._branch is None:
                    branch = await ArtifactLog.get_branch(ctx._branch_id)
                    branch_id = branch.id                    
                    ctx._branch = branch
                else:
                    branch_id = ctx.branch.id
        return branch_id
        
    
    @classmethod
    async def get_current_turn(cls, turn: "int | Turn | None" = None, ctx: "Context | None" = None):
        turn_id = None
        if ctx is None:
            ctx = Context.get_current(raise_error=False)
        if turn:
            if isinstance(turn, int):
                turn_id = turn
            elif isinstance(turn, Turn):
                turn_id = turn.id
        else:
            if ctx:
                if ctx.turn is None:
                    turn = await ctx._start_new_turn()
                    turn_id = turn.id
                else:
                    turn_id = ctx.turn.id            
        return turn_id
    
    
                
    
    @property
    def is_initialized(self):
        return self._branch is not None and self._turn is not None
    
  
    @property
    def branch(self):
        # if self._branch is None:
            # raise ValueError("Branch not set")
        return self._branch
    
    @property
    def turn(self):
        # if self._turn is None:
            # raise ValueError("Turn not set")
        return self._turn
    
    @property
    def can_push(self):
        return self._init_method != InitStrategy.NO_PARTITION
    
  
    def start_turn(self, branch_id: int | None = None, auto_commit: bool = False):
        if branch_id is None:
            branch_id = 1
        self._init_method = InitStrategy.START_TURN
        self._init_params["branch_id"] = branch_id
        self._auto_commit = auto_commit
        return self
    
    
    
    async def branch_from(self, turn_id: int, name: str | None = None):
        branch = await ArtifactLog.create_branch(forked_from_turn_id=turn_id, name=name)        
        return branch

    
    def build_child(self, span_name: str | None = None):
        child = Context(user=self.user, filters=self.filters, on_exit="none", span_name=span_name)
        child._branch = self._branch
        child._turn = self._turn
        child._parent_ctx = self
        return child
    
    async def _start_new_turn(self):
        # if "branch_id" in self._init_params:
        #     branch = await NamespaceManager.get_branch(self._init_params["branch_id"])
        #     if branch is None:
        #         raise ValueError(f"Branch {self._init_params['branch_id']} not found")
        #     self._branch = branch
        self._turn = await NamespaceManager.create_turn(
            branch_id=self._branch_id,
            state=self._state
        )
        return self._turn
            
        
    def _set_context(self):
        self._ctx_token = CURR_CONTEXT.set(self)
        
    def _reset_context(self):
        if self._ctx_token is not None:
            CURR_CONTEXT.reset(self._ctx_token)
        self._ctx_token = None
        
    def start_tracer(self, name: str, run_type: RunTypes = "prompt", inputs: dict[str, Any] | None = None):
        self._tracer_run = Tracer(
            name=name,
            run_type=run_type,
            inputs=inputs,
            is_traceable=True,
            tracer_run=self._parent_ctx._tracer_run if self._parent_ctx is not None else None,
        )
        return self
        
        
    async def __aenter__(self):    
        # if self._branch is None or self._turn is None:
            # raise ValueError("Branch or turn not set")
        self._set_context()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self._reset_context()
        if self._tracer_run is not None:
            self._tracer_run.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            if self._parent_ctx is None:
                await self.revert(message=str(exc_value))
            return False
        if self._on_exit == "commit" and self._parent_ctx is None:
            await self.commit()
        elif self._on_exit == "revert" and self._parent_ctx is None:
            await self.revert()
        return True
    
    
    async def commit(self):
        if self.turn is None:
            return 
        if self.turn.status == TurnStatus.COMMITTED:
            raise ValueError("Turn already committed, cannot commit again")
        if self.turn.status == TurnStatus.REVERTED:
            raise ValueError("Turn already reverted, cannot commit")
        await ArtifactLog.commit_turn(
            turn_id=self.turn.id,             
            trace_id=self.trace_id,
        )
    
    async def revert(self, message: str | None = None):
        if self.turn is None:
            return
        await ArtifactLog.revert_turn(
            turn_id=self.turn.id,
            message=message,
            trace_id=self.trace_id,
        )
    
    async def push(self, value: "Block | CONTEXT_MODEL | Blockable") -> "Block":
        if not self.can_push:
            raise ValueError("Cannot push to context")
        if not self.is_initialized:
            await self._init()
        from chatboard.prompt import Block
        if isinstance(value, Block):
            msg = self.context_model.from_block(value)
        elif hasattr(value, "block"):
            msg = self.context_model.from_block(value.block())
        else:
            msg = value
        saved_value = await msg.save(turn=self.turn.id, branch=self.branch.id)
        return saved_value.to_block(self)
        
    
    async def last(self, limit=5) -> "BlockList":
        # if not self.can_push:
            # raise ValueError("Cannot load last messages from context")
        if not self.is_initialized:
            await self._init()
        from chatboard.block.block import Block, BlockList
        records = await self.context_model.query(
            self.partition_id, 
            self.branch
        ).turn_limit(limit).order_by("created_at", direction="desc")
        # ).limit(limit).order_by("created_at", direction="desc")
        # with Block() as blocks:
        #     for r in reversed(records):
        #         blocks /= r.to_block()
        return BlockList([r.to_block(self) for r in reversed(records)])
    
    
    
    
    
    
    
    
    

