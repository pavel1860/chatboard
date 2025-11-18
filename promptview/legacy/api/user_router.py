from typing import Dict, Type, List, TypeVar, Generic
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.datastructures import QueryParams
from pydantic import BaseModel

# from app.util.dependencies import get_partitions
from promptview.api.model_router2 import create_crud_router
from promptview.auth.dependencies import get_auth_user, get_user_manager, get_user_token
from promptview.model.resource_manager import connection_manager
# from app.util.auth import varify_token
# from app.util.dependencies import unpack_request_token
from promptview.model.model import Model
from promptview.auth.user_manager import AuthManager, AuthModel
from typing import Type, Any, Dict, Optional
from fastapi import Query
from pydantic import BaseModel
from promptview.artifact_log.artifact_log3 import ArtifactLog

AUTH_MODEL = TypeVar("AUTH_MODEL", bound=AuthModel)



def unpack_int_env_header(request: Request, field: str):    
    value = request.headers.get(field)
    if value is None or value == "null":
        return None
    return int(value)
        

async def get_user(user_token: str = Depends(get_user_token), user_manager: AuthManager[AUTH_MODEL] = Depends(get_user_manager)):
    return await user_manager.get_by_session_token(user_token)

async def get_admin_user(user: AUTH_MODEL = Depends(get_user)):
    if not user.is_admin:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


# def create_user_crud_router(user_model: Type[USER_MODEL]) -> APIRouter:    
#     router = APIRouter(prefix=f"/{user_model.__name__}", tags=[user_model.__name__.lower()])
    
    
#     async def env_ctx(request: Request):
#         yield None
#         # head_id = unpack_int_env_header(request, "head_id")
#         # branch_id = unpack_int_env_header(request, "branch_id")
#         # # with connection_manager.set_env(env or "default"):
#         #     # yield env
#         # if head_id is None:
#         #     raise HTTPException(status_code=400, detail="head_id is not supported")
#         # async with ArtifactLog(head_id=head_id, branch_id=branch_id) as art_log:
#         #     yield art_log
    
#     def validate_access(instance: USER_MODEL, partitions: Dict[str, str]):
#         for key, value in partitions.items():
#             if getattr(instance, key) != value:
#                 raise HTTPException(status_code=403, detail="Forbidden")
#         return instance
    
    
#     def generate_filter_args(model: Type[BaseModel]) -> Dict[str, Any]:
#         """
#         Dynamically generate query parameters for filtering based on model fields.
#         """
#         filters = {}
#         for field_name, field_type in model.__annotations__.items():
#             filters[field_name] = Query(
#                 None, description=f"Filter by {field_name}", alias=field_name
#             )
#         return filters


#     def filter_dependency(model: Type[BaseModel]) -> Any:
#         """
#         Dependency to extract and return filter arguments.
#         """
#         def filter_query(**filters: Any):
#             return {key: value for key, value in filters.items() if value is not None}

#         def dependency():
#             return filter_query(**generate_filter_args(model))

#         return Depends(dependency)



#     @router.post("/create")
#     async def create_instance(data: dict, env: str = Depends(env_ctx)):
#         instance = user_model(**data)
#         await instance.save()  # Assuming your save method is asynchronous
#         return instance.model_dump()

#     @router.get("/id/{item_id}")
#     async def read_instance(item_id: str, env: str = Depends(env_ctx)):
#         instance = await user_model.get(item_id)
#         if not instance:
#             raise HTTPException(status_code=404, detail="Item not found")            
#         return instance.model_dump()

#     @router.post("/update/{item_id}")
#     async def update_instance(item_id: str, updates: dict, env: str = Depends(env_ctx)):
#         instance = await user_model.get(item_id)
#         if not instance:
#             raise HTTPException(status_code=404, detail="Item not found")
#         for key, value in updates.items():
#             setattr(instance, key, value)
#         await instance.save()
#         return instance.model_dump()

#     @router.post("/delete/{item_id}")
#     async def delete_instance(item_id: str, env: str = Depends(env_ctx)):
#         instance = await user_model.get(item_id)
#         if not instance:
#             raise HTTPException(status_code=404, detail="Item not found")
#         result = await user_model.delete(item_id)
#         if not result:
#             raise HTTPException(status_code=404, detail="Item not found")
#         return {"detail": "Item deleted"}

#     @router.get("/list")
#     async def list_instances(
#             limit: int = 10, 
#             offset: int = 0, 
#             # filters: Dict[str, Any] = Depends(filter_dependency(model)), 
#             # partitions: dict = Depends(get_partitions),
#             env: str = Depends(env_ctx)
#         ):
#         model_query = user_model.filter(lambda x: x.is_admin == False).limit(limit).offset(offset).order_by("created_at", False)
#         # model_query._filters = filters
#         instances = await model_query        
#         return instances
#         # return [instance.model_dump() for instance in instances]
    
#     @router.get("/last")
#     async def last_instance(request: Request, env: str = Depends(env_ctx)):
#         query_params = dict(request.query_params)
#         instance = await user_model.last(query_params)
#         if not instance:
#             return None
#         return instance.model_dump()
    
#     @router.get("/first")
#     async def first_instance(request: Request, env: str = Depends(env_ctx)):
#         query_params = dict(request.query_params)
#         instance = await user_model.first(query_params)
#         if not instance:
#             return None
#         return instance.model_dump()
    
#     class ChangeHeadRequest(BaseModel):
#         head_id: int
    
#     @router.post("/change-head")
#     async def change_head(request: ChangeHeadRequest, user= Depends(get_user), env: str = Depends(env_ctx)):
#         head = await user.check_out_head(request.head_id)
#         return head
    
    
    
#     # @router.get("/similar", response_model=List[model])
#     # async def similar_instances(query: str, limit: int = 10, offset: int = 0, partitions: dict = Depends(get_partitions)):
#     #     return await model.similar()

#     return router


class ChangeHeadRequest(BaseModel):
    head_id: int

def create_user_crud_router(user_model: Type[AUTH_MODEL]) -> APIRouter:    
    router = create_crud_router(user_model)
    
    @router.post("/change-head")
    async def change_head(request: ChangeHeadRequest, user= Depends(get_user)):
        head = await user.check_out_head(request.head_id)
        return head
    
    return router

user_model_router = APIRouter(prefix=f"/users", tags=["users"])

@user_model_router.get("/envs")
async def get_envs():    
    return connection_manager.get_env_names()

def connect_user_model_routers(app, user_model_list: List[Type[AUTH_MODEL]], envs: Dict[str, Dict[str, str]] | None = None, prefix: str = "/api"):
    for model in user_model_list:
        router = create_user_crud_router(model)
        app.include_router(router, prefix=f"{prefix}/users", tags=[model.__name__.lower()])
    if envs:
        for env_name, env in envs.items():
            connection_manager.add_env(env_name, env)
    app.include_router(user_model_router, prefix=f"{prefix}/users", tags=["manager"])
    
    