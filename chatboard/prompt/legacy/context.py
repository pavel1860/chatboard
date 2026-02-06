from abc import abstractmethod
import contextvars
from enum import Enum
from typing import Any, Callable, Generic, List, Protocol, Type, TypeVar, Union, Dict

from chatboard.artifact_log.artifact_log3 import ArtifactLog
from chatboard.model.model import Model
from collections import defaultdict
from chatboard.block.block import BlockRole, ToolCall, LlmUsage, Block
import datetime as dt

from typing import TypedDict, List
from ..local_state import TurnHooks

class MessageView(TypedDict):
    content: List[str]
    role: str



CURR_CONTEXT = contextvars.ContextVar("curr_context")






MODEL = TypeVar("MODEL", bound=Model)



def sanitize_block(block: Block):
    # if not isinstance(block, Block):        
    #     block = LLMBlock.from_block(block)        
    return block


class BlockStream(Generic[MODEL]):
    
    def __init__(
        self,         
        blocks: List[Block] | None = None,        
        run_id: str | None = None, 
        prompt_name: str | None = None
    ):
        self._blocks = []
        self._block_lookup = defaultdict(list)        
        if blocks:
            for b in blocks:
                self.append(sanitize_block(b))        
        self._dirty = True
        self.response = None
        self.run_id = run_id
        self.prompt_name = prompt_name  
        
    
                
    
    def __add__(self, other: Union[list[Block] , "BlockStream"]):
        if isinstance(other, list):
            for i, b in enumerate(other):
                if not isinstance(b, Block):
                    raise ValueError(f"Invalid block type: {type(b)} at index {i}")
            return BlockStream(self._blocks + other, run_id=self.run_id, prompt_name=self.prompt_name)
        elif isinstance(other, BlockStream):
            return BlockStream(self._blocks + other._blocks, run_id=self.run_id, prompt_name=self.prompt_name)
        else:       
            raise ValueError(f"Invalid type: {type(other)}")
        
    def __radd__(self, other: Union[list[Block], "BlockStream"]):
        if isinstance(other, list):
            for i, b in enumerate(other):
                if not isinstance(b, Block):
                    raise ValueError(f"Invalid block type: {type(b)} at index {i}")
            return BlockStream(other + self._blocks, run_id=self.run_id, prompt_name=self.prompt_name)
        elif isinstance(other, BlockStream):
            return BlockStream(other._blocks + self._blocks, run_id=self.run_id, prompt_name=self.prompt_name)
        else:       
            raise ValueError(f"Invalid type: {type(other)}")        
    
        
    def _update_lookup(self, block: Block):
        block = sanitize_block(block)
        if isinstance(block, Block):
            if block.id:
                self._block_lookup[block.id].append(block)
            for b in block.items:
                self._update_lookup(b)            
    

    def append(self, block):
        self._blocks.append(block)
        # self._block_lookup[block.id].append(block)
        self._update_lookup(block)
        
    def prepend(self, block):
        self._blocks.insert(0, block)
        # self._block_lookup[block.id].insert(0, block)
        self._update_lookup(block)
        
    def insert(self, index: int, block):
        self._blocks.insert(index, block)
        # self._block_lookup[block.id].insert(index, block)
        self._update_lookup(block)
        
    def get(self, key: str | MessageView | list[str | MessageView] ):
        
                
        return chat_blocks
        
        
    
    async def push(self, block: Block | str | dict, tool: ToolCall | None = None, role: BlockRole | None = None, id: str | None="history"):
        platform_id = None
        tool_calls = None        
        
        if isinstance(block, Block):
            block = block.root
        
        
        if isinstance(block, Block):
            content = block.render()
            _role = role or block.role or "user"
            if tool:
                platform_id = tool.id
            if isinstance(block, Block):
                tool_calls = [a.model_dump() for a in block.tool_calls]
        elif isinstance(block, Block):
            content = block.render()
            _role = role or "user"
        elif isinstance(block, str) or isinstance(block, dict):
            content = block    
            _role = role or "user"
            
        else:
            raise ValueError(f"Invalid block type: {type(block)}")
        
        block = Block(content, role=_role, id=platform_id, tool_calls=tool_calls)
        ctx = CURR_CONTEXT.get()
        model = ctx.from_blocks(block)
        await model.save()
        # model = self.from_blocks(block)
        # message = Message(
        #     content=content,
        #     role=_role,
        #     name=id,
        #     platform_id=platform_id,
        #     extra={"action_calls": action_calls}
        # )        
        # message = await self.message_log.append(message)
        # block.db_msg_id = message.id
        self.append(block)
        return model
    
    
        
    # async def commit_turn(self):
        # await self.artifact_log.commit_turn()

    # async def delete(self, block: BaseBlock):
        # if block.db_msg_id:
            # await self.message_log.delete_message(id=block.db_msg_id)
        # self._blocks.remove(block)
        # if block.id in self._block_lookup:
            # self._block_lookup[block.id].remove(block)
        
    
    # def blocks_to_messages(self, blocks: List[BaseBlock]):
    #     messages = []
    #     for block in blocks:
    #         if isinstance(block, ResponseBlock):
    #             messages.append(Message(role="assistant", content=block.render(), platform_uuid=block.platform_id, action_calls=[a.model_dump() for a in block.action_calls]))
    #         elif block.role == "tool":
    #             messages.append(Message(role="tool", content=block.render(), platform_uuid=block.platform_id))
    #         else:
    #             messages.append(Message(role="user", content=block.render(), platform_uuid=block.platform_id))
    #     return messages
    
    
    def __getitem__(self, key):
        return self._blocks[key]
    
    def __len__(self):
        return len(self._blocks)
    
    def display(self):
        from chatboard.prompt.util.block_visualization import display_block_stream
        display_block_stream(self)







class InitMethod(Enum):
    START = "start"
    RESUME = "resume"
    RESUME_SESSION = "resume_session"



class BaseContext():
    _ctx_token: contextvars.Token | None
    run_id: str | None
    
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        self._ctx_token = None
        
    def set_context(self):        
        self._ctx_token = CURR_CONTEXT.set(self)
    
    def reset_context(self):
        if self._ctx_token:
            try: 
                CURR_CONTEXT.reset(self._ctx_token)
            except ValueError as e:
                print(f"Warning: failed to reset context, probably because generator was closed improperly:")
            self._ctx_token = None
    
    async def __aenter__(self):
        self.set_context()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self.reset_context()
        # return sel
    
    
 
    

class Context(Generic[MODEL], BaseContext):
    _model: Type[MODEL]
    _artifact_log: ArtifactLog
    _head_id: int
    _branch_id: int | None
    _initialized: bool
    inputs: dict
    _ctx_token: contextvars.Token | None
    _hooks: TurnHooks | None
    run_id: str | None
    prompt_name: str
    
    def __init__(
        self, 
        head_id: int,
        branch_id: int | None = None,
        inputs: dict | None = None, 
        artifact_log: ArtifactLog | None = None,
        prompt_name: str = "global",
        run_id: str | None = None,
        commit_on_exit: bool = False,
    ):
        self._head_id = head_id
        self._branch_id = branch_id
        self._artifact_log = artifact_log or ArtifactLog(head_id=head_id, branch_id=branch_id)
        self._initialized = False
        self.inputs = inputs or {}
        self._ctx_token = None
        self._hooks = None
        self.tracer = None
        self._parent_ctx = None
        self.prompt_name = prompt_name
        self.run_id = run_id
        self._init_method: Dict[str, Any] | None = None
        self._commit_on_exit = commit_on_exit
    
    @property
    def artifact_log(self):
        if self._artifact_log is None:
            raise ValueError("Artifact log not set")
        return self._artifact_log
        
    def build_child(self, prompt_name: str):
        ctx = self.__class__(
            head_id=self.head_id,
            branch_id=self.branch_id,
            inputs=self.inputs, 
            artifact_log=self._artifact_log,
            prompt_name=prompt_name,
            run_id=self.run_id
        )
        ctx._parent_ctx = self        
        ctx._initialized = self._initialized
        # ctx._hooks = TurnHooks(self._artifact_log, prompt_name=prompt_name)
        return ctx
    

    @staticmethod
    def get_current():
        try:
            return CURR_CONTEXT.get()
        except LookupError:
            return None
    
    @property
    def session(self):
        if self._session is None:
            raise ValueError("Session not set")
        return self._session
    
    # def set_context(self):        
    #     self._ctx_token = CURR_CONTEXT.set(self)
    
    # def reset_context(self):
    #     if self._ctx_token:
    #         try: 
    #             CURR_CONTEXT.reset(self._ctx_token)
    #         except ValueError as e:
    #             print(f"Warning: failed to reset context, probably because generator was closed improperly:")
    #         self._ctx_token = None
    
    
    @property
    def session_id(self):
        if self.session is None:
            raise ValueError("Session not set")
        return self.session.id
    
    @property
    def head(self):
        return self._artifact_log.head
    
    @property
    def head_id(self):
        return self._artifact_log.head["id"]
    
    @property
    def branch_id(self):
        return self._artifact_log.head["branch_id"]
    
    @property
    def turn_id(self):
        return self._artifact_log.head["id"]
    
    @property
    def turn(self):
        return self._artifact_log.head["id"]
    
    @property
    def branch(self):
        return self._artifact_log.head["branch_id"]
    
    
    @property
    def is_initialized(self):
        return self._initialized
    
        
    async def get_last_messages(self, limit: int = 10):
        return await self.message_log.get_messages(limit)
    
    
    def resume_session(self, session_id: int, branch_id: int | None = None, new_turn: bool = True):
        self._init_method = {
            "type": InitMethod.RESUME_SESSION,
            "kwargs": {"session_id": session_id, "branch_id": branch_id, "new_turn": new_turn}
        }
        return self
    
    def resume(self, new_turn: bool = True):
        self._init_method = {
            "type": InitMethod.RESUME,
            "kwargs": {"new_turn": new_turn}
        }
        return self
        
    def start(self):
        self._init_method = {
            "type": InitMethod.START,
            "kwargs": {}
        }
        return self
    
    
    async def __aenter__(self):
        if not self._artifact_log.is_initialized:
            await self._artifact_log.init_head(head_id=self._head_id, branch_id=self._branch_id)
            self._artifact_log.set_context()
        self.set_context()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        # await self.commit()
        if self._commit_on_exit:
            await self.commit()
        self.reset_context()
        
    def use_hooks(self, prompt_name: str | None = None) -> TurnHooks:
        return TurnHooks(self.message_log, prompt_name or self.prompt_name)
    
    async def use_state(self, key: str, initial_value: Any = None):
        hooks = self.use_hooks()
        state, set_state = await hooks.use_state(key, initial_value)
        return state, set_state
    
    async def use_var(self, key: str, initial_value: Any = None):
        hooks = self.use_hooks()
        return await hooks.use_var(key, initial_value)
            
    # async def delete_message(self, id: int):
        # await self.message_log.delete_message(id=id)
    
    # @abstractmethod
    # def to_blocks(self, model: MODEL) -> LLMBlock:
    #     pass
    
    # @abstractmethod
    # def from_blocks(self, block: LLMBlock) -> MODEL:
    #     pass
    
    async def commit(self):
        await self.artifact_log.commit_turn()   

    async def last_artifacts(self, limit=10):
        records = await self._model.limit(limit).order_by("created_at", ascending=False)
        return records
    
    async def push(self, value: Block | MODEL) -> Block:
        if isinstance(value, Block):
            msg = self._model.from_block(value)
        else:
            msg = value
        saved_value = await msg.save()
        return saved_value.to_block()
        
    async def last(self, limit=10) -> Block:
        records = await self._model.limit(limit).order_by("created_at", ascending=False)
        with Block() as blocks:
            for r in reversed(records):
                blocks /= r.to_block()
        return blocks
    
