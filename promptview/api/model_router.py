from typing import AsyncContextManager, Set, Type, List, TypeVar
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from ..model.query_url_params import parse_query_params, QueryListType
from ..model import Model, TurnStatus
from .utils import query_filters



MODEL = TypeVar("MODEL", bound=Model)
CTX_MODEL = TypeVar("CTX_MODEL", bound=BaseModel)

def create_model_router(model: Type[MODEL], get_context: AsyncContextManager[CTX_MODEL], exclude_routes: Set[str] = set()):
    
    # model_context_parser = build_model_context_parser(model_context_cls)
    # async def model_context_parser(request: Request, ctx: CTX_MODEL = Depends(get_context)):
    #     ctx._request = CtxRequest(request=request)
    #     return ctx
    
    router = APIRouter(prefix=f"/{model.__name__}", tags=[model.__name__.lower()])
    
    if "list" not in exclude_routes:
        @router.get("/list", response_model=List[dict])
        async def list_models(
            request: Request,
            offset: int = Query(default=0, ge=0, alias="filter.offset"),
            limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
            filters: QueryListType | None = Depends(query_filters),
            include: str | None = Query(default=None, alias="include"),
            ctx: CTX_MODEL = Depends(get_context),            
        ):
            """List all models with pagination"""
            # async with ctx:

                # if model.__name__ == "Turn":
                #     print("Turn")
            print(request)
            query = model.query(status=TurnStatus.COMMITTED)

            model_query = query.limit(limit).offset(offset).order_by("-created_at")
            if filters:
                condition = parse_query_params(model, filters, model_query.from_table)
                model_query.query.where(condition)

            # Parse include parameter
            include_list = []
            if include:
                import json
                try:
                    include_list = json.loads(include)
                except json.JSONDecodeError:
                    pass

            # Apply includes to query
            if include_list:
                for relation in include_list:
                    model_query = model_query.include(relation)

            # model_query._filters = filters
            instances = await model_query
            return [instance.model_dump() for instance in instances]       
    
    if "record" not in exclude_routes:
        @router.get("/record/{record_id}")
        async def get_artifact(
            record_id: int | UUID,
            include: str | None = Query(default=None)
        ):
            """Get a specific artifact by ID"""
            query = model.query(status=TurnStatus.COMMITTED).filter(lambda x: x.id == record_id)

            # Parse include parameter
            include_list = []
            if include:
                import json
                try:
                    include_list = json.loads(include)
                except json.JSONDecodeError:
                    pass

            # Apply includes to query
            if include_list:
                for relation in include_list:
                    query = query.include(relation)

            artifact = await query.first()
            if not artifact:
                raise HTTPException(status_code=404, detail="Model not found")
            return artifact
    
    
    # @router.get("/{artifact_id}/version/{version}")
    # async def get_artifact_version(artifact_id: UUID, version: int):
    #     """Get a specific artifact by ID and version"""
    #     artifact = await model.get_artifact(artifact_id, version)
    #     if not artifact:
    #         raise HTTPException(status_code=404, detail="Artifact not found")
    #     return artifact

    if "create" not in exclude_routes:
        @router.post("/create")
        async def create_model(
            payload: dict = Body(...),
            ctx: CTX_MODEL = Depends(get_context)
        ):
            """Create a new model"""
            try:
                relation_payload = {}
                ns = model.get_namespace()
                for name, rel_info in ns._relations.items():
                    if name in payload:
                        relation_payload[name] = (rel_info, payload[name])
                        del payload[name]
                        
                model_payload = model(**payload)
                created_model = await model_payload.save()
                
                for name, (rel_info, data) in relation_payload.items():                    
                    cls = rel_info.foreign_cls
                    if not isinstance(data, list):
                        data = [data]
                    for item in data:
                        await created_model.add(cls(**item))
                return created_model
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    if "update" not in exclude_routes:
        @router.put("/update/{model_id}")
        async def update_model(
            model_id: int | UUID,
            request: Request,
            ctx: CTX_MODEL = Depends(get_context)
        ):
            """Update an existing model"""
            payload = await request.json()
            # existing = await model.query(status=TurnStatus.COMMITTED).filter(lambda x: x.id == model.id).first()
            # existing = await model.get(model_id)
            updated = await model.update_query(model_id, payload)
            if not updated:
                raise HTTPException(status_code=404, detail="Model not found")
            return updated
    
    if "delete" not in exclude_routes:
        @router.delete("/delete")
        async def delete_model(request: Request, ctx: CTX_MODEL = Depends(get_context)):
            """Delete an model"""
            payload = await request.json()
            model_id = payload.get("id")
            existing = await model.query(status=TurnStatus.COMMITTED).filter(lambda x: x.id == model_id).first()
            if not existing:
                raise HTTPException(status_code=404, detail="Artifact not found")
            
            try:
                await existing.delete()
                return {"message": "Artifact deleted successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    if "last" not in exclude_routes:
        @router.get("/last")
        async def last_model(
            skip: int = Query(default=0, ge=0, alias="filter.skip"),
            limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
            ctx: CTX_MODEL = Depends(get_context)
        ):
            """Get the last model with pagination"""
            artifact = await model.query(**ctx).last()
            return artifact
    
    if "first" not in exclude_routes:
        @router.get("/first")
        async def first_model(
            skip: int = Query(default=0, ge=0, alias="filter.skip"),
            limit: int = Query(default=10, ge=1, le=100, alias="filter.limit"),
            ctx: CTX_MODEL = Depends(get_context)
        ):
            """Get the first model with pagination"""
            artifact = await model.query(**ctx).first()
            return artifact
    
    
    return router
    
    
    
    
 