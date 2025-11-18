
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
import json
from typing import AsyncContextManager, Awaitable, Callable, Concatenate, Dict, Generic, List, Literal, ParamSpec, Type, TypeVar
from fastapi import Depends, FastAPI, Request


from .api.tracing_router import router as tracing_router
from .api.model_router import create_model_router
from .auth.user_manager2 import AuthManager, AuthModel
from .model import ArtifactModel, Model, NamespaceManager
from .prompt import Context
from .context.model_context import CtxRequest, ModelCtx
from .api.auth_router import create_auth_router
from .api.artifact_log_api import router as artifact_log_router
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


MSG_MODEL = TypeVar('MSG_MODEL', bound=Model)
USER_MODEL = TypeVar('USER_MODEL', bound=AuthModel)
CTX_MODEL = TypeVar('CTX_MODEL', bound=Context)


P = ParamSpec('P')

EnpointType = Callable[Concatenate[CTX_MODEL, P], Awaitable[List[MSG_MODEL]]]

app_ctx = ContextVar("app_ctx")



class AddCtxMiddleware(BaseHTTPMiddleware):
    
    
    
    def is_form_request(self,request: Request) -> bool:
        content_type = request.headers.get("content-type", "")
        return content_type.startswith("application/x-www-form-urlencoded") or \
            content_type.startswith("multipart/form-data")
            
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api"):            
            ctx = None
            ctx_str = request.query_params.get("ctx")
            if ctx_str is not None:
                ctx = json.loads(ctx_str)
            request.state.ctx = ctx
            filters = request.query_params.get("filter")
            if filters:
                request.state.filters = filters
            list_params = request.query_params.get("list")
            if list_params:
                list_params = json.loads(list_params)
                request.state.list = {
                    "offset": list_params.get("offset"), 
                    "limit": list_params.get("limit")
                }
            message = None
            config = None
            if self.is_form_request(request):
                form = await request.form()
                message = form.get("message")            
                message = json.loads(message) if message else None
                config = form.get("config") 
                config = json.loads(config) if config else None
            request.state.message = message
            request.state.config = config
        
        response = await call_next(request)
        return response





def include_chatboard_routers(app: FastAPI, user_manager: AuthManager[USER_MODEL]):
    app.include_router(artifact_log_router, prefix="/api")
    app.include_router(create_auth_router(user_manager), prefix="/api")
    app.include_router(tracing_router, prefix="/api")
    # app.add_middleware(AddCtxMiddleware)


class Chatboard(Generic[MSG_MODEL, USER_MODEL, CTX_MODEL]):
    _app: FastAPI
    _entrypoints_registry: Dict[str, EnpointType]
    _message_model: Type[MSG_MODEL]
    _user_model: Type[USER_MODEL]
    _ctx_model: Type[CTX_MODEL]
    def __init__(
        self, 
        message_model: Type[MSG_MODEL], 
        user_model: Type[USER_MODEL], 
        auth_manager: Type[AuthManager[USER_MODEL]],
        ctx_model: Type[CTX_MODEL],
        app: FastAPI | None = None
    ):
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # This code runs before the server starts serving
            ns = user_model.get_namespace()
            await auth_manager.initialize_tables()
            auth_manager.register_user_model(user_model)
            await NamespaceManager.create_all_namespaces()
            # Yield to hand control back to FastAPI (start serving)
            yield
            
        self._app = app or FastAPI(lifespan=lifespan)
        self._entrypoints_registry = {}
        self._message_model = message_model
        self._user_model = user_model
        self._auth_manager = auth_manager
        self._ctx_model = ctx_model
        self.setup_apis()
        self.app_token = app_ctx.set(self)
        
    @classmethod
    def get_app_ctx(cls):
        return app_ctx.get()
    
    def get_app(self):
        return self._app
    
    def register_model_api(self, model: Type[Model], get_context: AsyncContextManager[CTX_MODEL]):
        self._app.include_router(create_model_router(model, get_context), prefix="/api/model")        
    
    def setup_apis(self):
        # self._app.include_router(create_crud_router(self._message_model), prefix="/api/model")
        self._app.include_router(artifact_log_router, prefix="/api")
        self._app.include_router(create_auth_router(self._user_model), prefix="/api")
        self._app.include_router(tracing_router, prefix="/api")
        # connect_testing_routers(self._app)

    
    def entrypoint(self, path: str, method: Literal['GET', 'POST', 'PUT', 'DELETE'] = 'POST', ctx: Type[CTX_MODEL]| None = None, commit: bool = True):
        
        async def model_context_parser(request: Request, ctx: Context = Depends(ctx.from_request if ctx else self._ctx_model.from_request)):
            ctx._request = CtxRequest(request=request)
            return ctx
        def decorator(func: EnpointType):
            self._entrypoints_registry[path] = func
            async def input_endpoint(
                ctx: CTX_MODEL = Depends(model_context_parser),
            ):                
                async with ctx:
                    responses = await func(ctx=ctx)
                return responses
                # async with self._ctx_model(
                #     user,
                #     ctx.partition, 
                #     "commit",
                #     state=state,
                #     branch=branch_id
                # ).start_tracer(
                #         name=func.__name__, 
                #         run_type="chain", 
                #         inputs={
                #             "message": message.model_dump_json(),
                #             "state": state.model_dump_json() if isinstance(state, BaseModel) else state,
                #             "branch_id": branch_id,
                #         }
                #     ) as ctx:
                #     responses = await func(ctx=ctx, message=message)
                #     print(ctx.state)
                # return [message, *responses]
            
            async def test_endpoint(
                body: dict,
            ):
                test_case_id = body.get("test_case_id")
                async with TestManager(test_case_id=test_case_id) as tm:
                        for inputs, evaluator in tm.iter_turns():
                            async with self._ctx_model(head_id=tm.head_id) as ctx:
                                message = self._message_model(content=inputs, role="user")
                                response = await func(ctx=ctx, message=message)
                                await ctx.commit()
                            await evaluator(response.content)
                return tm.test_run
            
            self._app.add_api_route(path, endpoint=input_endpoint, methods=[method])
            self._app.add_api_route("/api/testing/run", endpoint=test_endpoint, methods=["POST"])
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    
    
    
    
    