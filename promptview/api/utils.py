

from typing import Type, TypeVar
from fastapi import Request,  Depends, HTTPException, Query, Request
import json


from ..model.query_filters import QueryListType
from pydantic import BaseModel, Field







def unpack_int_env_header(request: Request, field: str, default: int | None = None):
    value = request.headers.get(field)
    if value is None or value == "null":
        return default
    return int(value)


def query_filters(request: Request):
    # filters = request.query_params.get("filter.filter")
    filters = request.query_params.get("filter")
    if filters:
        return json.loads(filters)
    return None

# def query_filters(request: Request) -> QueryFilter | None:
#     filters = request.query_params.get("filter.filter")
#     if filters:
#         return parse_query_params(json.loads(filters))
#     return None

class ListModelQuery(BaseModel):
    offset: int = Query(default=0, ge=0, alias="filter.offset")
    limit: int = Query(default=10, ge=1, le=100, alias="filter.limit")
    filters: QueryListType | None = Depends(query_filters)


def get_list_query(
        offset: int = Query(default=0, ge=0, alias="filter.offset"),
        limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
        filters: QueryListType | None = Depends(query_filters)
    ):
        return ListModelQuery(offset=offset, limit=limit, filters=filters)
        




# async def get_head(request: Request, auth_user: AuthModel = Depends(get_auth_user)) -> Head | None:
#     partition_id = unpack_int_env_header(request, "partition_id")
#     partition = None
#     if partition_id is None:
#         if not auth_user.is_admin:
#             partition = await auth_user.last_partition("default")
#             if not partition:
#                 raise HTTPException(status_code=401, detail="No Partition specified for regular user")        
#     else:
#         partition = await ArtifactLog.get_partition(partition_id)
#         if not partition.is_participant(auth_user.id) and not auth_user.is_admin:
#             raise HTTPException(status_code=401, detail="Unauthorized partition for user")
        
#     branch_id = unpack_int_env_header(request, "branch_id")
#     if branch_id is None:
#         branch_id = 1
#     turn_id = unpack_int_env_header(request, "turn_id")
#     head = Head(branch_id=branch_id, turn_id=turn_id, partition_id=partition_id, partition=partition)
#     return head
    
    

MODEL_CONTEXT = TypeVar("MODEL_CONTEXT", bound=BaseModel)


def build_model_context_parser(model_context_cls: Type[MODEL_CONTEXT]):
    def extract_model_context(request: Request) -> MODEL_CONTEXT:
        raw = {}
        for key in request.headers.keys():
            if key.endswith("_id"):
                val = request.headers.get(key)
                if val and val != "null":
                    raw[key] = int(val) if val.isdigit() else val
        return model_context_cls(**raw)
    return extract_model_context




class ListParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)
    
    
    
def get_list_params(
    list_str: str | None = Query(None, alias="list")
):
    if list_str is None:
        return None
    try:
        list_params = ListParams.model_validate_json(list_str)
    except json.JSONDecodeError:
        return None
    return list_params





def get_request_ctx(request: Request):
    ctx = request.query_params.get("ctx")
    if ctx:
        return json.loads(ctx)
    return None

def is_form_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return content_type.startswith("application/x-www-form-urlencoded") or \
        content_type.startswith("multipart/form-data")


async def get_request_content(request: Request):
    if is_form_request(request):
        form = await request.form()
        content = form.get("content")
        content = content.decode("utf-8") if isinstance(content, bytes) else content
        content = json.loads(content) if content else None
        options = form.get("options") 
        options = options.decode("utf-8") if isinstance(options, bytes) else options
        options = json.loads(options) if options else None
        state = form.get("state") 
        state = state.decode("utf-8") if isinstance(state, bytes) else state
        state = json.loads(state) if state else None
        files = form.get("files") 
        files = files.decode("utf-8") if isinstance(files, bytes) else files
        files = json.loads(files) if files else None
        return content, options, state, files
    else:
        return None, None, None, None





def get_auth_manager(request: Request):
    return request.app.state.user_manager



async def get_auth(request: Request):
    auth_manager = get_auth_manager(request)
    return await auth_manager.get_user_from_request(request)
    

# def get_user_ref(request: Request):
#     ctx = request.query_params.get("ctx")
#     if not ctx:
#         return None
#     ctx = json.loads(ctx)
#     if ctx.get("ref_user_id"):
#         return ctx.get("ref_user_id")
#     return None

# async def get_user_from_request(request: Request):
#     from ..auth.user_manager2 import UserNotFound
#     user_ref_id = get_user_ref(request)
#     user_manager = get_user_manager(request)
#     try:
#         user = await request.app.state.user_manager.get_user_from_request(request)
#         # logger.info(f"user: {user.id}")
#     except UserNotFound as e:
#         # logger.error(f"User not found: {e}")
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     if user_ref_id:
#         if not user.is_admin:
#             # logger.error(f"Unauthorized user: {user.id} tried to access user: {user_ref_id}")
#             raise HTTPException(status_code=401, detail="Unauthorized")
#         # ref_user = await User.query().where(user_id=user_ref_id).last()
#         ref_user = await user_manager.fetch_by_auth_user_id(user_ref_id)
#         if ref_user is None:
#             # logger.error(f"admin user: {user.id} tried to fetch non existing user: {user_ref_id}")
#             raise HTTPException(status_code=401, detail="Unauthorized")
#         return ref_user
#     return user

