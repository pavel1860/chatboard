from typing import AsyncGenerator, Generic, Literal, ParamSpec, Set, Callable, TYPE_CHECKING, Type
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ..prompt import PipeController, component, stream
from ..prompt.fbp_process import TurnController
from ..prompt.context import Context, VerbosityLevel
from ..prompt.flow_components import EventLogLevel, FlowRunner
from ..block.util import StreamEvent
from ..block import Block
from ..api.utils import get_auth, get_request_content, get_request_ctx
from ..auth.user_manager2 import AuthModel
from fastapi import APIRouter, FastAPI, Query, Request, Depends, UploadFile
import datetime as dt
from ..evaluation import EvaluationContext
if TYPE_CHECKING:   
    from ..model import Branch


async def get_ctx_from_request(request: Request):
    ctx_args = request.state.get("ctx")
    if ctx_args is None:
        raise ValueError("ctx is not set")
    ctx = Context(**ctx_args)    
    return ctx








# @asynccontextmanager
# async def agent_context(request: Request | None = None, kwargs: dict | None = None):    
    
#     if request is not None:
#         ctx = await get_ctx_from_request(request)
#     elif kwargs is not None:
#         user = kwargs["user"]
#         branch = kwargs["branch"]
#         partition = kwargs["partition"]
#         message = message = Block(kwargs["message"], role="user")
#         ex_ctx = ExecutionContext(request_id=kwargs["request_id"])
#     else:
#         raise ValueError("request or kwargs is required")
#     async with ex_ctx:
#         with user:
#             with partition:
#                 with branch:
#                     yield message, user, partition, branch
def process_state(state: dict) -> dict:    
    for key in state.keys():
        value = state[key]
        if isinstance(value, dict):
            if "_type" in value:
                if value["_type"] == "Block":
                    state[key] = Block.model_validate(value)
            else:
                state[key] = process_state(value)
    return state

P = ParamSpec("P")

class Agent(Generic[P]):
    name: str
    # agent_component: Callable[P, AsyncGenerator[StreamEvent, None]]
    agent_component: Callable[P, PipeController]
    ingress_router: APIRouter
    
    def __init__(self, agent_component, name: str | None = None, state_model: Type[BaseModel] | None = None):
        self.name = name or "default"
        self.agent_component = agent_component
        self.ingress_router = APIRouter(prefix=f"/{name}" if name else "")
        self._setup_ingress()
        self.state_model = state_model
        
    def connect_ingress(self, app: FastAPI):          
        app.include_router(self.ingress_router, prefix="/api")
        print(f"{self.name} agent conntected")
        
        
    def _block_from_content(self, content: dict | str, role: str):
        if isinstance(content, str):
            return Block(content, role=role)
        else:
            return Block(content.get("content"), role=role)

    def _setup_ingress(self):
        from ..auth.user_manager2 import AuthModel
        @self.ingress_router.post("/complete")
        async def complete(       
            request: Request,
            payload: tuple[str, dict, dict, list[UploadFile]] = Depends(get_request_content),
            ctx: dict = Depends(get_request_ctx),
            auth: AuthModel = Depends(get_auth),
            # ctx: tuple[User, Branch, Partition, Message, ExecutionContext] = Depends(get_ctx),
        ):  
            # context = await Context.from_request(request)
            content, options, state, files = payload
            message = self._block_from_content(content, options['role'])
            # state = process_state(state)
            if self.state_model is not None:
                state = self.state_model.model_validate(state)
            context = await Context.from_kwargs(**ctx, auth=auth, message=message, state=state)            
            context = context.start_turn()
            agent_gen = self.stream_agent_with_context(context, message, state=state)
            return StreamingResponse(agent_gen, media_type="text/plain")
        
        @self.ingress_router.post("/replay")
        async def replay(
            request: Request,
            payload: str = Depends(get_request_content),
            ctx: dict = Depends(get_request_ctx),
            auth: AuthModel = Depends(get_auth),
        ):
            content, options, state, files = payload
            context = await Context.from_kwargs(**ctx, auth=auth)
            turn_id = options.get('fork_from', None)
            if turn_id is None:
                raise ValueError("forkFrom is required")
            span = await SpanTree.from_turn(turn_id)
            args = span.get_input_args() 
            # context = context.fork(turn_id=turn_id).start_turn()
            context = context.start_turn()
            agent_gen = self.stream_agent_with_context(context, args[0], serialize=True)
            return StreamingResponse(agent_gen, media_type="text/plain")
            
        @self.ingress_router.post("/evaluate")
        async def evaluate(
            request: Request,
            payload: str = Depends(get_request_content),
            ctx: dict = Depends(get_request_ctx),
            auth: AuthModel = Depends(get_auth),
        ):
            content, options, state, files = payload
            if not options.get("test_case_id"):
                raise ValueError("test_case_id is required")
            agent_gen = self.run_evaluate(options["test_case_id"])
            return StreamingResponse(agent_gen, media_type="text/plain")
    
    def update_metadata(self, ctx: Context, events: list[StreamEvent],  event: StreamEvent):
        event.request_id = ctx.request_id
        event.turn_id = ctx.turn.id
        event.branch_id = ctx.branch.id
        # event.timestamp = int(datetime.now().timestamp() * 1000)
        event.created_at = dt.datetime.now()
        # event.index = index
        events.append(event)
        return event


    async def stream_agent_with_context(
        self,
        ctx: Context,       
        message: Block,
        state: dict | None = None,
        # files: list[UploadFile] | None = None,
        serialize: bool = True,
        filter_events: Set[str] | None = None,
        auto_commit: bool = True,
        metadata: dict | None = None,
    ):

        async with ctx:
            # auto_commit = user.auto_respond == "auto" and message.role == "user" or message.role == "assistant"
            # async with branch.start_turn(
            #     metadata=metadata, 
            #     auto_commit=auto_commit
            # ) as turn: 
                for event in ctx.events:
                    print("streaming event", event)
                    yield event.to_ndjson() if serialize else event
                           
            
                if message.role == "user":
                    events = []  
                    # index = 0
                    async for event in self.agent_component(message).stream():
                        print("streaming event", event)
                        # event = self.update_metadata(ctx, events, event)
                        # index += 1
                        if filter_events and event.type not in filter_events:
                            continue
                        # if ctx.user.auto_respond == "auto":
                        try:            
                            yield event.to_ndjson() if serialize else event
                        except Exception as e:
                            raise e
                            print("Error streaming event", e)
                            
                # elif message.role == "assistant":
                #     if not user.phone_number:
                #         raise ValueError("User phone number is required")
                #     async with TwilioClient(
                #         manager_phone_number=settings.twilio_phone_number,
                #         manager_phone_id=settings.twilio_account_sid,
                #         access_token=settings.twilio_auth_token,
                #     ) as twilio:
                #         message = await turn.add(message)
                #         await twilio.send_text_message(user.phone_number, message.content)
                
    # async def run_evaluate(
    #     self,
    #     test_case_id: int,
    #     test_run_id: int | None = None,
    #     auth: AuthModel | None = None,
    # ):
    #     ctx = Context(auth=auth)
    #     async with ctx.start_eval(test_case_id=test_case_id, test_run_id=test_run_id) as ctx:
    #         args = ctx.eval_ctx.get_eval_span_tree([0]).get_input_args()
    #         async for event in self.agent_component(*args).stream():
    #             yield event
    
    # def build
    
    def build_context(
        self, 
        auth: AuthModel, 
        branch_id: int,
        from_turn_id: int | None = None,
        start_turn: bool = True,
        eval_ctx: EvaluationContext | None = None,
    ) -> Context:
        ctx = Context(auth=auth, branch_id=branch_id, eval_ctx=eval_ctx)
        if from_turn_id:
            ctx.fork(turn_id=from_turn_id)
        if start_turn:
            ctx = ctx.start_turn()         
        return ctx
    
    async def build_flow_runner(
        self,
        auth,
        branch_id: int,
        load_eval_args: bool = False,
    ) -> FlowRunner:
        return self.agent_component().stream(ctx=ctx, load_eval_args=load_eval_args)
    
    async def run_evaluate(
        self,        
        eval_ctx: EvaluationContext | None = None,
        test_case_id: int | None = None,
        test_run_id: int | None = None,
        auth: AuthModel | None = None,
    ):
        if eval_ctx is None:
            if test_case_id is None:
                raise ValueError("test_case_id is required if eval_ctx is not provided")
            eval_ctx = await EvaluationContext(auth=auth).init(test_case_id=test_case_id, test_run_id=test_run_id)
        if eval_ctx.did_start:
            raise ValueError("Evaluation context already started")
        
        for test_turn in eval_ctx:
            args = test_turn.turn.get_args()
            ctx = self.build_context(auth=auth or eval_ctx.auth, branch_id=eval_ctx.branch.id, eval_ctx=eval_ctx)
            async with ctx:
                async for event in self.agent_component(*args).stream():
                    yield event
                await eval_ctx.commit_test_turn()
                
    def run_debug(
        self,
        message: Block | str,  
        state: BaseModel | None = None,
        auth: AuthModel | None = None,
        branch_id: int | None = None, 
        branch: "Branch | None" = None,         
        auto_commit: bool = True,
        level: Literal["chunk", "span", "turn"] = "chunk",      
        verbose: set[VerbosityLevel] | None = None,
        load_cache: dict[str, str] | None = None,
        **kwargs: dict,
    ):

        ctx = Context(auth=auth, branch=branch, branch_id=branch_id, load_cache=load_cache, verbose=verbose)
        if state is not None:
            ctx.state = state
        print_events = verbose and "events" in verbose
        ctx.start_turn(auto_commit=auto_commit)
        if isinstance(message, str):
            message = Block(message, role="user")
            # yield self.agent_component(message).stream(event_level=EventLogLevel[level])
        # agent_gen = self.agent_component(message)
        # return TurnController(self.agent_component, ctx, message, name=self.name)
        return self.agent_component(message).stream(event_level=EventLogLevel[level], ctx=ctx)
    @component()
    async def run_debug4(
        self,
        message: Block | str,  
        state: BaseModel | None = None,
        auth: AuthModel | None = None,
        branch_id: int | None = None, 
        branch: "Branch | None" = None,         
        auto_commit: bool = True,
        level: Literal["chunk", "span", "turn"] = "chunk",      
        verbose: set[VerbosityLevel] | None = None,
        load_cache: dict[str, str] | None = None,
        **kwargs: dict,
    ):
        ctx = await Context.from_kwargs(**kwargs, auth=auth, branch=branch, branch_id=branch_id, load_cache=load_cache, verbose=verbose)
        if state is not None:
            ctx.state = state
        print_events = verbose and "events" in verbose
        ctx.start_turn(auto_commit=auto_commit) 
        yield self.agent_component(message).stream(event_level=EventLogLevel[level], ctx=ctx)
        

        
            

    async def run_debug1(
        self,
        message: Block | str,  
        state: BaseModel | None = None,
        auth: AuthModel | None = None,
        branch_id: int | None = None, 
        branch: "Branch | None" = None,         
        auto_commit: bool = True,
        level: Literal["chunk", "span", "turn"] = "chunk",      
        verbose: set[VerbosityLevel] | None = None,
        load_cache: dict[str, str] | None = None,
        **kwargs: dict,
    ):
        ctx = await Context.from_kwargs(**kwargs, auth=auth, branch=branch, branch_id=branch_id, load_cache=load_cache, verbose=verbose)
        if state is not None:
            ctx.state = state
        print_events = verbose and "events" in verbose
        async with ctx.start_turn(auto_commit=auto_commit) as turn:            
            async for event in self.agent_component(message).stream(event_level=EventLogLevel[level]):
                if print_events:
                    print("--------------------------------")
                    if isinstance(event, Block):
                        event.print()
                    else:
                        print(event)
                yield event
                
                


    async def run_debug2(
        self,
        message: str | Block,                
        auth: AuthModel | None = None,
        branch_id: int | None = None,         
        auto_commit: bool = True,
        level: Literal["chunk", "span", "turn"] = "chunk",      
        **kwargs: dict,
    ):
        context = await Context.from_kwargs(**kwargs, auth=auth)
        message = Block(message, role="user") if isinstance(message, str) else message        
        agent_gen = self.stream_agent_with_context(context, message, serialize=True)
        async for event in agent_gen:
            print("--------------------------------")
            if isinstance(event, Block):
                event.print()
            else:
                print(event)
            yield event