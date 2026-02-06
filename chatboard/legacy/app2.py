import asyncio
import os
from typing import Any, List, Literal, Optional

from .app_manager import app_manager
from .llms.utils.completion_parsing import (is_list_model,
                                                    unpack_list_model)
from .llms.prompt_tracer import PromptTracer
from .llms.tracer_api import get_run_messages
from .vectors.rag_documents import RagDocuments
from .vectors.stores.base import OrderBy
from .model.resource_manager import connection_manager
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel

LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")

class GetRagParams(BaseModel):
    namespace: str


class GetAssetDocumentsParams(BaseModel):
    asset: str


class UpsertRagParams(BaseModel):
    namespace: str
    input: Any
    output: Any
    id: str | int | None = None


class DeleteRagParams(BaseModel):
    id: str | int




def add_promptboard(app, rag_namespaces=None, assets=None, profiles=None, prompts=None):

    if rag_namespaces:
        for space in rag_namespaces:            
            app_manager.register_rag_space(space["namespace"], space["metadata_class"], space["prompt"])

    if assets:
        for asset in assets:
            app_manager.register_asset(asset)


    if profiles:
        for profile in profiles:
            app_manager.register_profile(profile)

    if prompts:
        for prompt in prompts:
            app_manager.register_prompt(prompt)


    @asynccontextmanager
    async def init_promptboard(app: FastAPI):
        print("Initializing promptboard...")
        await app_manager.verify_rag_spaces()
        yield


    app.router.lifespan_context = init_promptboard

    
    @app.get('/promptboard/metadata')
    def get_promptboard_metadata():
        app_metadata = app_manager.get_metadata()
        # return {"metadata": app_metadata}
        return app_metadata
    
    print("promptboard added to app.")


    @app.get("/promptboard/get_asset_documents")
    async def get_asset_documents(asset: str):
        asset_cls = app_manager.assets[asset]
        asset_instance = asset_cls()
        res = await asset_instance.get_assets()
        return [r.to_dict() for r in res]
    
    @app.get("/promptboard/rag_documents/{namespace}")
    # async def get_rag_documents(namespace: str, offset: int = 0, limit: int = 10):
    async def get_rag_documents(namespace: str, page: int = 0, pageSize: int = 10, sortField: str | None=None, sortOrder: Literal["asc", "desc"] | None=None):
        rag_cls = app_manager.rag_spaces[namespace]["metadata_class"]
        ns = app_manager.rag_spaces[namespace]["namespace"]
        rag_space = RagDocuments(ns, metadata_class=rag_cls)

        order_by = OrderBy(
            key= sortField,
            direction= sortOrder,
            start_from= page*pageSize
        ) if sortOrder and sortField else None
        
        res = await rag_space.get_documents(top_k=pageSize, offset=page*pageSize, order_by=order_by)
        return [r.to_dict() for r in res]
    

    @app.post("/promptboard/get_rag_document")
    async def get_rag_document(body: GetRagParams):
        print(body.namespace)
        rag_cls = app_manager.rag_spaces[body.namespace]["metadata_class"]
        ns = app_manager.rag_spaces[body.namespace]["namespace"]
        rag_space = RagDocuments(ns, metadata_class=rag_cls)
        res = await rag_space.get_many(top_k=10)
        return res


    @app.post("/promptboard/rag_documents/upsert_rag_document")
    async def upsert_rag_document(body: UpsertRagParams):
        rag_cls = app_manager.rag_spaces[body.namespace]["metadata_class"]
        ns = app_manager.rag_spaces[body.namespace]["namespace"]
        prompt_cls = app_manager.rag_spaces[body.namespace].get("prompt", None)
        if prompt_cls is not None:
            prompt = prompt_cls()
            user_msg_content = await prompt.render_prompt(**body.input)
            
        rag_space = RagDocuments(ns, metadata_class=rag_cls)
        # doc_id = [body.id] if body.id is not None else None
        doc_id = [body.id] if body.id is not None else None
        key = user_msg_content
        if is_list_model(rag_cls):
            list_model = unpack_list_model(rag_cls)
            if type(body.output) == list:
                value = [list_model(**item) for item in body.output]
            else:
                raise ValueError("Output must be a list.")
        else:
            value = rag_cls(key=key, value=body.output)
        #     value = rag_cls(**body.output)
        res = await rag_space.add_documents([key], [value], doc_id) #type: ignore
        return [r.to_dict() for r in res]
    
    @app.get('/promptboard/get_asset_partition')
    async def get_asset_partition(asset: str, field: str, partition: str):
        # asset_cls = app_manager.assets[asset]
        # asset_instance = asset_cls()
        # assets = await asset_instance.get_assets(filters={ field: partition })        
        # return [a.to_json() for a in assets]
        model_cls = connection_manager.get_model(asset)
        if not model_cls:
            raise ValueError("Model class not found for asset")
        recs = await model_cls.partition({field: partition}).limit(30)
        return recs
        


    @app.get('/promptboard/get_profile_partition')
    async def get_profile_partition(profile: str, partition: str):
        profile_cls = app_manager.profiles[profile]
        profile_list = await profile_cls.get_many()        
        return [p.to_dict() for p in profile_list]
    

    @app.post("/promptboard/edit_document")
    def edit_rag_document():
        return {}
    

    @app.get("/promptboard/get_runs")
    async def get_runs(limit: int = 10, offset: int = 0, runNames: Optional[List[str]] = None):
        LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")
        tracer = PromptTracer()
        runs = await tracer.aget_runs(name=runNames, limit=limit, project_name=LANGCHAIN_PROJECT)
        return [r.run for r in runs]
    

    @app.get("/promptboard/get_run_tree")
    async def get_run_tree(run_id: str):
        tracer = PromptTracer()
        run = await tracer.aget_run(run_id)
        return run
    


    @app.get("/promptboard/get_trace")
    async def get_trace(run_id: str):
        tracer = PromptTracer()
        run = await tracer.aget_run(run_id)
        trace = get_run_messages(run)
        return trace.model_dump()