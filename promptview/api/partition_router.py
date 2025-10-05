import json
from typing import List
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from ..model.sql.compiler import Compiler
from ..model.query_filters import QueryListType
from .utils import query_filters
from ..model import Branch, TurnStatus
from ..model.query_url_params import parse_query_params
from ..utils.db_connections import PGConnectionManager
from ..prompt.context import Context
from .model_router import create_model_router
from fastapi import Request
from typing import Type
from .utils import ListParams, get_list_params





# @asynccontextmanager
# async def get_partition_context(
#     user: User = Depends(get_api_user),
#     branch: Branch = Depends(get_branch),
# ):
#     with user:
#         with branch:
#             yield PartitionCtx(user=user, branch=branch)


async def get_conversations(limit: int, offset: int, where_clause: str, params: list):
    query = f"""
    SELECT
        p.id,
        p.name,
        p.avatar,
        p.phone_number,
        p.provider,
        p.platform,
        p.created_at,
        p.updated_at,
        COALESCE(
            (
            SELECT jsonb_agg(
                jsonb_build_object(
                'id', u.id,
                'auth_user_id', u.auth_user_id,
                'is_guest', u.is_guest,
                'guest_token', u.guest_token,
                'is_admin', u.is_admin,
                'created_at', u.created_at,
                'name', u.name,
                'email', u.email,
                'image', u.image,
                'role', u.role,
                'phone_number', u.phone_number,
                'provider', u.provider,
                'auto_respond', u.auto_respond
                )
            )
            FROM users AS u
            INNER JOIN partition_participants AS p1 ON p1.user_id = u.id
            WHERE p1.partition_id = p.id
            ),
            '[]'::jsonb
        ) AS participants,
        COALESCE(
            (
            SELECT jsonb_agg(turn_obj)
            FROM (
                SELECT jsonb_build_object(
                'id', t.id,
                'created_at', t.created_at,
                'ended_at', t.ended_at,
                'index', t.index,
                'status', t.status,
                'message', t.message,
                'branch_id', t.branch_id,
                'trace_id', t.trace_id,
                'metadata', t.metadata,
                'partition_id', t.partition_id,
                'messages', COALESCE(
                    (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                        'id', m.id,
                        'created_at', m.created_at,
                        'content', m.content,
                        'state', m.state,
                        'name', m.name,
                        'sender_id', m.sender_id,
                        'task_id', m.task_id,
                        'role', m.role
                        )
                        ORDER BY m.created_at ASC
                    )
                    FROM messages AS m
                    WHERE m.turn_id = t.id
                    ),
                    '[]'::jsonb
                )
                ) AS turn_obj
                FROM turns AS t
                WHERE t.partition_id = p.id
                ORDER BY t.created_at DESC
                LIMIT 1
            ) AS s
            ),
            '[]'::jsonb
        ) AS turns        
        FROM partitions AS p
        {where_clause}
"""
    res = await PGConnectionManager.fetch(query, *params)
    for row in res:
        row['turns'] = json.loads(row['turns'])
        row['participants'] = json.loads(row['participants'])
    return res



def create_partition_router(context_cls: Type[Context] | None = None):
    context_cls = context_cls or Context
    
    def get_model_ctx(request: Request):
        return context_cls.from_request(request)
    
    partition_router = create_model_router(Partition, get_model_ctx, exclude_routes={"update"})

    router = APIRouter(prefix=f"/Partition", tags=["Partition"])
        
    @router.get("/list", response_model=List[dict])
    async def list_models(
        list_params: ListParams = Depends(get_list_params),
        filters: QueryListType | None = Depends(query_filters),
        ctx = Depends(get_model_ctx)
    ):
        """List all models with pagination"""
                
        async with ctx:
            query = Partition.query().limit(10)
            where_clause = ""
            params = []
            if filters:      
                condition = parse_query_params(Partition, filters, query.table)
                print(condition)
                compiler = Compiler()
                where_clause = compiler.compile_expr(condition)
                where_clause = f"WHERE {where_clause}"
                params = compiler.params
            conversations = await get_conversations(limit=list_params.limit, offset=list_params.offset, where_clause=where_clause, params=params)
            return conversations
        



    @router.get("/record/{id}")
    async def get_artifact(
        id: UUID, 
        ctx = Depends(get_model_ctx)
        ):
        """Get a specific artifact by ID"""
        artifact = await Partition.get(id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact



    @router.post("/create")
    async def create_model(
        payload: dict = Body(...),
        ctx = Depends(get_model_ctx)
    ):
        """Create a new model"""
        try:
            partition = await user.add(Partition(**payload))
            branch = await Branch.get(1)        
            turn = await branch.create_turn(status=TurnStatus.COMMITTED, partition_id=partition.id)
                
            return partition
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=400, detail=str(e))

    @router.put("/update")
    async def update_model(
        partition: Partition, 
        ctx = Depends(get_model_ctx)
    ):
        """Update an existing model"""
        existing = await Partition.query(status=TurnStatus.COMMITTED).filter(lambda x: x.id == partition.id).first()
        if not existing:
            raise HTTPException(status_code=404, detail="Model not found")
        
        try:
            for field, value in partition.dict(exclude_unset=True).items():
                setattr(existing, field, value)
            updated = await existing.save()
            return updated
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/delete")
    async def delete_model(        
        ctx = Depends(get_model_ctx)
        ):
        """Delete an model"""
        existing = await Partition.query(status=TurnStatus.COMMITTED).filter(lambda x: x.id == partition.id).first()
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        try:
            await existing.delete()
            return {"message": "Artifact deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/last")
    async def last_model(
        skip: int = Query(default=0, ge=0, alias="filter.skip"),
        limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
        ctx = Depends(get_model_ctx)
    ):
        """Get the last model with pagination"""
        artifact = await ctx.user.partitions.query().last()
        return artifact

    @router.get("/first")
    async def first_model(
        user: User = Depends(get_user_from_request),
    ):
        """Get the first model with pagination"""
        artifact = await user.partitions.query().first()
        return artifact