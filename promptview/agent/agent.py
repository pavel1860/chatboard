from typing import Literal, Set
from fastapi.responses import StreamingResponse

from ..model.context import Context
from ..prompt.flow_components import EventLogLevel
from ..block.util import StreamEvent
from ..block import Block
from ..api.utils import get_auth, get_request_content, get_request_ctx
from ..auth.user_manager2 import AuthModel
from fastapi import APIRouter, FastAPI, Query, Request, Depends
import datetime as dt



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




class Agent():
    
    def __init__(self, agent_component, name: str | None = None):
        self.name = name or "default"
        self.agent_component = agent_component
        self.ingress_router = APIRouter(prefix=f"/{name}" if name else "")
        self._setup_ingress()

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
            payload: str = Depends(get_request_content),
            ctx: dict = Depends(get_request_ctx),
            auth: AuthModel = Depends(get_auth),
            # ctx: tuple[User, Branch, Partition, Message, ExecutionContext] = Depends(get_ctx),
        ):  
            print("ctx >>>", auth)      
            # context = await Context.from_request(request)
            content, options, state, files = payload
            message = self._block_from_content(content, options['role'])
            context = await Context.from_kwargs(**ctx, auth=auth)            
            agent_gen = self.stream_agent_with_context(context, message)
            return StreamingResponse(agent_gen, media_type="text/plain")
        
        # self.ingress_router.add_api_route(
        #     path="/complete", 
        #     endpoint=complete,
        #     methods=["POST"],        
        # )
    
    def update_metadata(self, ctx: Context, index, events: list[StreamEvent],  event: StreamEvent):
        event.request_id = ctx.request_id
        event.turn_id = ctx.turn.id
        event.branch_id = ctx.branch.id
        # event.timestamp = int(datetime.now().timestamp() * 1000)
        event.created_at = dt.datetime.now()
        event.index = index
        events.append(event)
        return event


    async def stream_agent_with_context(
        self,
        ctx: Context,       
        message: Block,
        serialize: bool = True,
        filter_events: Set[str] | None = None,
        auto_commit: bool = True,
        metadata: dict | None = None,
    ):

        async with ctx.start_turn() as turn:            
            # auto_commit = user.auto_respond == "auto" and message.role == "user" or message.role == "assistant"
            # async with branch.start_turn(
            #     metadata=metadata, 
            #     auto_commit=auto_commit
            # ) as turn:
                if message.role == "user":
                    events = []  
                    index = 0
                    async for event in self.agent_component(message).stream_events():
                        event = self.update_metadata(ctx, index, events, event)
                        index += 1
                        if filter_events and event.type not in filter_events:
                            continue
                        # if ctx.user.auto_respond == "auto":            
                        yield event.to_ndjson() if serialize else event
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
                

    async def run_debug(
        self,
        message: Block | str,                
        auth: AuthModel | None = None,
        branch_id: int | None = None,         
        auto_commit: bool = True,
        level: Literal["chunk", "span", "turn"] = "chunk",      
        **kwargs: dict,
    ):
        ctx = await Context.from_kwargs(**kwargs, auth=auth)
        async with ctx.start_turn(auto_commit=auto_commit) as turn:            
            async for event in self.agent_component(message).stream_events(event_level=EventLogLevel[level]):
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