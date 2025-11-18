from typing import Dict, Type, List, TypeVar, Generic
from uuid import UUID
from fastapi import APIRouter, Depends, Request
from .utils import get_auth
from ..auth.user_manager2 import AuthModel
from pydantic import BaseModel
from ..model import NamespaceManager, Turn, Branch




router = APIRouter(prefix="/artifact_log", tags=["artifact_log"])


# def unpack_int_env_header(request: Request, field: str):    
#     value = request.headers.get(field)
#     if value is None or value == "null" or value == "undefined":
#         return None
#     return int(value)

# def get_artifact_log(request: Request):
#     head_id = unpack_int_env_header(request, "head_id")
#     branch_id = unpack_int_env_header(request, "branch_id")
#     if head_id is None:
#         raise HTTPException(status_code=400, detail="head_id is not supported")
#     try:
#         return ArtifactLog(head_id=head_id, branch_id=branch_id)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))




@router.get("/branches", response_model=List[Branch])
async def get_branches():
    branches = await ArtifactLog.list_branches()
    return branches


@router.get("/branches/{branch_id}", response_model=Branch)
async def get_branch(branch_id: int):
    branch = await ArtifactLog.get_branch(branch_id)
    return branch



class BranchFromTurnRequest(BaseModel):
    turn_id: int
    
@router.post("/branches")
async def branch_from_turn(request: BranchFromTurnRequest):
    raise NotImplementedError
    branch = await ArtifactLog.create_branch(forked_from_turn_id=request.turn_id)
    return branch


@router.get("/all_turns", response_model=List[Turn])
async def get_all_turns():
    turns = await ArtifactLog.list_turns()
    return turns


@router.get("/partitions", 
    # response_model=List[Partition]
    )
async def get_partitions(user: AuthModel = Depends(get_auth)):
    # partitions = await ArtifactLog.list_partitions(user.id)
    raise NotImplementedError
    return partitions


class CreatePartitionPayload(BaseModel):
    name: str
    participants: List[int]

@router.post("/partitions")
async def create_partition(payload: CreatePartitionPayload, user: AuthModel = Depends(get_auth)):
    raise NotImplementedError
    # partition = await ArtifactLog.create_partition(payload.name, [user.id] + payload.participants)
    return partition

# @router.get("/heads/{head_id}")
# async def get_head(head_id: int):
#     raise NotImplementedError

    
    
# @router.get("/turns/{branch_id}/partition/{partition_id}", response_model=List[Turn])
# async def get_branch_turns(branch_id: int, partition_id: int):    
#     turn_ns = NamespaceManager.get_namespace("turns")
#     branch_ns = NamespaceManager.get_namespace("branches")
#     branch = await (
#         branch_ns.query()
#         .where(id=branch_id)
#         .include(
#             turn_ns.query()
#             .where(partition_id=partition_id)
#             .include(branch_ns.model_class)
#             .tail(20)
#         )
#         .last()
#     )
#     return branch.turns


@router.get("/turns/{branch_id}/partition/{partition_id}")
async def get_branch_turns(branch_id: int, partition_id: UUID):    
    turn_ns = NamespaceManager.get_namespace("turns")
    turns = await turn_ns.query() \
        .select("*") \
        .where(branch_id=branch_id,partition_id=partition_id) \
        .agg("forked_branches", Branch.query(["id"]), on=("id", "forked_from_turn_id")) \
        .tail(20).json()
    return list(reversed(turns))



@router.post("/turns/update/{turn_id}")
async def update_turn(turn_id: int, request: Request):
    # body = await request.json()    
    # turn = await Turn.query().where(id=turn_id).last()
    # turn = await turn.update(**body)
    # return turn
    body = await request.json()    
    ns = Turn.get_namespace()
    res = await ns.update(turn_id, body)
    return res

@router.get("/heads")
async def get_head_list():    
    raise NotImplementedError

