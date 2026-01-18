from typing import Type
from ..model import Branch, Turn
from .model_router import create_model_router
from ..prompt.context import Context
from fastapi import Request, Depends, Query, Body
from ..model.query_url_params import parse_query_params, QueryListType
from .utils import query_filters

def create_branch_router(context_cls: Type[Context] | None = None):
    context_cls = context_cls or Context
    
    async def get_model_ctx(request: Request):
        return await context_cls.from_request(request)
    
    branch_router = create_model_router(Branch, get_model_ctx, exclude_routes={"update"})
    
    
    @branch_router.get("/turns/{branch_id}") 
    async def get_branch_turns(
        branch_id: int,
        offset: int = Query(default=0, ge=0, alias="filter.offset"),
        limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
        filters: QueryListType | None = Depends(query_filters),
        ctx: Context = Depends(get_model_ctx)
    ):
        # async with ctx:
        query = Turn.query() \
            .agg("forked_branches", Branch.query(["id"]), on=("id", "forked_from_turn_id")) \
            .where(branch_id=branch_id)
        model_query = query.limit(limit).offset(offset).order_by("-created_at")
        if filters:
            condition = parse_query_params(Branch, filters, model_query.from_table)
            model_query.query.where(condition)
        # instances = await model_query.json()
        instances = await model_query
        return [instance for instance in instances]
    
    @branch_router.post("/fork")
    async def fork_branch(
        from_branch_id: int= Body(...),
        from_turn_id: int= Body(...),
        ctx: Context = Depends(get_model_ctx)
    ):
        branch = await Branch.get(from_branch_id)
        turn = await Turn.get(from_turn_id)
        forked_branch =await branch.fork_branch(turn)        
        return forked_branch

    return branch_router



