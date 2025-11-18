
import os
from typing import List
from fastapi import APIRouter
from ..llms.prompt_tracer import PromptTracer
from ..llms.tracer_api import get_run_messages


router = APIRouter(prefix="/tracing", tags=["tracing"])



@router.get("/get_runs")
async def get_runs(limit: int = 10, offset: int = 0, runNames: List[str] | None = None):
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")
    tracer = PromptTracer()
    runs = await tracer.aget_runs(name=runNames, limit=limit, project_name=LANGCHAIN_PROJECT)
    return [r.run for r in runs]


@router.get("/get_run_tree")
async def get_run_tree(run_id: str):
    tracer = PromptTracer()
    run = await tracer.aget_run(run_id)
    return run



@router.get("/get_trace")
async def get_trace(run_id: str):
    tracer = PromptTracer()
    run = await tracer.aget_run(run_id)
    trace = get_run_messages(run)
    return trace.model_dump()