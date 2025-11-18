from typing import Dict, Type, List, TypeVar, Generic
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.datastructures import QueryParams
from pydantic import BaseModel

# from app.util.dependencies import get_partitions
from promptview.auth.dependencies import get_user_token
from promptview.model import Model
# from app.util.auth import varify_token
# from app.util.dependencies import unpack_request_token

from typing import Type, Any, Dict, Optional
from fastapi import Query
from pydantic import BaseModel
import json

from promptview.model.query_filters import QueryListType, QueryFilter, parse_query_params
MODEL = TypeVar("MODEL", bound=Model)



def unpack_int_env_header(request: Request, field: str, default: int | None = None):
    value = request.headers.get(field)
    if value is None or value == "null":
        return default
    return int(value)


def query_filters(request: Request) -> QueryFilter | None:
    filters = request.query_params.get("filter")
    if filters:
        return parse_query_params(json.loads(filters))
    return None

        
# async def get_user(user_token: str = Depends(get_user_token), user_manager: UserManager = Depends(get_user_manager)):
#     return await user_manager.get_user_by_session_token(user_token)
class Head(BaseModel):
    branch_id: int
    turn_id: int | None = None

def create_crud_router(model: Type[MODEL], IdType: Type[int] | Type[str] = int) -> APIRouter:
    router = APIRouter(prefix=f"/{model.__name__}", tags=[model.__name__.lower()])
    
    
    
    async def get_head(request: Request) -> Head | None:
        if not model._is_versioned:
            return None
        branch_id = unpack_int_env_header(request, "branch_id")
        if branch_id is None:
            branch_id = 1
        turn_id = unpack_int_env_header(request, "turn_id")
        head = Head(branch_id=branch_id, turn_id=turn_id)
        return head
        
    
    def validate_access(instance: MODEL, partitions: Dict[str, str]):
        for key, value in partitions.items():
            if getattr(instance, key) != value:
                raise HTTPException(status_code=403, detail="Forbidden")
        return instance
    
    
    def generate_filter_args(model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Dynamically generate query parameters for filtering based on model fields.
        """
        filters = {}
        for field_name, field_type in model.__annotations__.items():
            filters[field_name] = Query(
                None, description=f"Filter by {field_name}", alias=field_name
            )
        return filters


    def filter_dependency(model: Type[BaseModel]) -> Any:
        """
        Dependency to extract and return filter arguments.
        """
        def filter_query(**filters: Any):
            return {key: value for key, value in filters.items() if value is not None}

        def dependency():
            return filter_query(**generate_filter_args(model))

        return Depends(dependency)



    @router.post("/create")
    async def create_instance(req: Request, data: dict, head: Head | None = Depends(get_head)):
        print(req)
        instance = model(**data)
          # Assuming your save method is asynchronous
        head = data.get("head", None)
        if head is not None:
            await instance.save(head.turn_id, head.branch_id)
        else:
            await instance.save()
        return instance.model_dump()

    @router.get("/id/{item_id}")
    async def read_instance(item_id: IdType):
        instance = await model.get(item_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Item not found")            
        return instance.model_dump()

    @router.post("/update/{item_id}")
    async def update_instance(item_id: IdType, updates: dict):
        instance = await model.get(item_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Item not found")
        for key, value in updates.items():
            setattr(instance, key, value)
        await instance.save()
        return instance.model_dump()

    @router.post("/delete/{item_id}")
    async def delete_instance(item_id: IdType):
        instance = await model.get(item_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Item not found")
        result = await model.delete(item_id)
        if not result:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"detail": "Item deleted"}

    @router.get("/list")
    async def list_instances(
            # request: Request,
            limit: int = 10, 
            offset: int = 0, 
            filters: QueryListType | None = Depends(query_filters), 
            # filters: Dict[str, Any] = Depends(filter_dependency(model)), 
            # partitions: dict = Depends(get_partitions),
            head: Head | None = Depends(get_head)
        ):
        if head is None:
            query = model.query()
        else:
            query = model.query(branch=head.branch_id)
                
        model_query = query.limit(limit).offset(offset).order_by("created_at", "desc")
        if filters:
            model_query = model_query.set_filter(filters)
        # model_query._filters = filters
        instances = await model_query        
        return [instance.model_dump() for instance in instances]
    
    @router.get("/last")
    async def last_instance(request: Request, head: Head | None = Depends(get_head)):
        query_params = dict(request.query_params)
        if head is None:
            instance = await model.query().last()
        else:
            instance = await model.query(branch=head.branch_id).last()
        if not instance:
            return None
        return instance.model_dump()
    
    @router.get("/first")
    async def first_instance(request: Request, head: Head | None = Depends(get_head)):
        query_params = dict(request.query_params)
        if head is None:
            instance = await model.query().first()
        else:
            instance = await model.query(branch=head.branch_id).first()
        if not instance:
            return None
        return instance.model_dump()
    
    # @router.get("/similar", response_model=List[model])
    # async def similar_instances(query: str, limit: int = 10, offset: int = 0, partitions: dict = Depends(get_partitions)):
    #     return await model.similar()

    return router




model_manager_router = APIRouter(prefix=f"/manager", tags=["manager"])

@model_manager_router.get("/envs")
async def get_envs():    
    raise NotImplementedError("Not implemented")
    # return connection_manager.get_env_names()

def connect_model_routers(app, model_list: List[Type[Model]], envs: Dict[str, Dict[str, str]] | None = None, prefix: str = "/api"):
    raise NotImplementedError("Not implemented")
    for model in model_list:
        router = create_crud_router(model)
        app.include_router(router, prefix=f"{prefix}/model", tags=[model.__name__.lower()])
    if envs:
        for env_name, env in envs.items():
            connection_manager.add_env(env_name, env)
    app.include_router(model_manager_router, prefix=f"{prefix}/model", tags=["manager"])
    
    