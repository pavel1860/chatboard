from typing import Type, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from ..auth.dependencies import get_auth_user
from ..auth.user_manager import AuthModel
from ..model import ArtifactModel
from ..context.model_context import ModelCtx
from ..model.query_filters import QueryFilter, QueryListType
from .utils import build_model_context_parser, query_filters


def create_artifact_router(model: Type[ArtifactModel], model_context_cls: Type[ModelCtx] | None = None):
    
    model_context_parser = build_model_context_parser(model_context_cls or ModelCtx)
    
    router = APIRouter(prefix=f"/{model.__name__}", tags=[model.__name__.lower()])
    
    @router.get("/list", response_model=List[dict])
    async def list_artifacts(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
        filters: QueryListType | None = Depends(query_filters),         
        auth_user: AuthModel = Depends(get_auth_user),
        ctx_params: dict = Depends(model_context_parser)
    ):
        """List all artifacts with pagination"""
        with auth_user:
            query = model.query(**ctx_params)
                    
            model_query = query.limit(limit).offset(offset).order_by("-created_at")
            if filters:
                model_query = model_query.set_filter(filters)
            # model_query._filters = filters
            instances = await model_query
            return [instance.model_dump() for instance in instances]       
    
    @router.get("/{artifact_id}")
    async def get_artifact(artifact_id: UUID):
        """Get a specific artifact by ID"""
        artifact = await model.get_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact
    
    
    @router.get("/{artifact_id}/version/{version}")
    async def get_artifact_version(artifact_id: UUID, version: int):
        """Get a specific artifact by ID and version"""
        artifact = await model.get_artifact(artifact_id, version)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return artifact

    
    @router.post("/")
    async def create_artifact(
        artifact: model,
        ctx_params: dict = Depends(model_context_parser)
    ):
        """Create a new artifact"""
        try:
            created_artifact = await artifact.save()
            return created_artifact
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.put("/{artifact_id}")
    async def update_artifact(artifact_id: UUID, updated_artifact: model):
        """Update an existing artifact"""
        existing = await model.query().filter(lambda x: x.id == artifact_id).first()
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        try:
            for field, value in updated_artifact.dict(exclude_unset=True).items():
                setattr(existing, field, value)
            updated = await existing.save()
            return updated
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.delete("/{artifact_id}")
    async def delete_artifact(artifact_id: UUID):
        """Delete an artifact"""
        existing = await model.query().filter(lambda x: x.id == artifact_id).first()
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        try:
            await existing.delete()
            return {"message": "Artifact deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/last")
    async def last_artifact(
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
        ctx_params: dict = Depends(model_context_parser)
    ):
        """Get the last artifact with pagination"""
        artifact = await model.query(**ctx_params).last()
        return artifact
    
    @router.get("/first")
    async def first_artifact(
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
        ctx_params: dict = Depends(model_context_parser)
    ):
        """Get the first artifact with pagination"""
        artifact = await model.query(**ctx_params).first()
        return artifact
    
    
    return router
    
    
    
    
    
    
    