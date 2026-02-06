import asyncio
from contextlib import asynccontextmanager
import json
from typing import Annotated
from fastapi import Depends, FastAPI, Form, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from chatboard.api.auth_router import router as auth_router
from chatboard.api.model_router2 import create_crud_router
from chatboard.api.head_model_router import create_head_crud_router
from chatboard.api.artifact_log_api import router as artifact_log_router
from chatboard.api.testing_router import connect_testing_routers
from chatboard.api.user_router import connect_user_model_routers
from chatboard.artifact_log.artifact_log3 import ArtifactLog
from app.test_models import TestContext, Manager, Message
from app.test_agent import chat_prompt, run_agent
from chatboard.auth.user_manager import UserManager
from chatboard.model.resource_manager import connection_manager
from chatboard import testing
from chatboard.prompt.base_prompt import prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs before the server starts serving
    artifact_log = ArtifactLog()
    await artifact_log.initialize_tables()
    UserManager.register_user_model(Manager)
    await UserManager.initialize_tables() 
    await connection_manager.init_all_namespaces()
    
    # Yield to hand control back to FastAPI (start serving)
    yield








# Create FastAPI app
app = FastAPI(
    title="chatboard API",
    description="API for interacting with the chatboard message system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(create_crud_router(Message), prefix="/api/model")
# app.include_router(create_crud_router(testing.TestCase), prefix="/api/testing")
# app.include_router(create_crud_router(testing.Evaluator), prefix="/api/evaluators")
app.include_router(artifact_log_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
connect_user_model_routers(app, [Manager])
connect_testing_routers(app)


def unpack_int_env_header(request: Request, field: str):    
    value = request.headers.get(field)
    if value is None or value == "null":
        return None
    return int(value)


# async def env_ctx(request: Request):
#     head_id = unpack_int_env_header(request, "head_id")
#     branch_id = unpack_int_env_header(request, "branch_id")
#     # with connection_manager.set_env(env or "default"):
#         # yield env
#     if head_id is None:
#         raise HTTPException(status_code=400, detail="head_id is not supported")
#     async with Context(head_id=head_id, branch_id=branch_id) as ctx:
#         yield ctx


# async def env_ctx(request: Request):
#     head_id = unpack_int_env_header(request, "head_id")
#     branch_id = unpack_int_env_header(request, "branch_id")
#     # with connection_manager.set_env(env or "default"):
#         # yield env
#     if head_id is None:
#         raise HTTPException(status_code=400, detail="head_id is not supported")
#     async with Context(head_id=head_id, branch_id=branch_id) as ctx:
#         yield ctx





# @app.post("/api/chat")
# async def chat(
#     # message: Message,
#     # env: str = Depends(env_ctx)
#     request: Request
#     ):
#     form_data = await request.form()
#     message = json.loads(form_data.get("message"))
#     print(message)
#     return [message, {"message": "Welcome to chatboard API", "role": "assistant"}]




@app.post("/api/chat")
async def chat(
    message_json:  Annotated[str, Form(...)],
    head_id: Annotated[int, Header(alias="head_id")],
    branch_id: Annotated[int | None, Header(alias="branch_id")] = None,
    ):
    
    async with TestContext(head_id=head_id, branch_id=branch_id) as ctx:
        message = Message.model_validate_json(message_json)
        response, message = await run_agent(ctx=ctx, message=message)
        await ctx.commit()
        return [response, message]






# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to chatboard API"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    ) 