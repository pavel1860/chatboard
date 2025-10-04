from typing import AsyncContextManager, Type, List
from fastapi import Depends, Request
from .model_router import create_model_router
from ..model.context import Context
from ..model.query_url_params import parse_query_params, QueryListType
from ..model import TurnStatus
from .utils import query_filters
from ..model.block_models.block_log import get_blocks
from ..model.versioning.models import Turn, ExecutionSpan, SpanValue, Log, Artifact
from ..model.versioning.models import Branch
from .utils import ListParams, get_list_params







def create_turn_router(context_cls: Type[Context] | None = None):
    context_cls = context_cls or Context
    
    async def get_model_ctx(request: Request):
        return await context_cls.from_request(request)
    
    turn_router = create_model_router(
        Turn, 
        get_model_ctx, 
        # exclude_routes={"update"}
    )
    
    
    @turn_router.get("/spans")   
    async def get_turn_spans(
        list_params: ListParams = Depends(get_list_params),
        filters: QueryListType | None = Depends(query_filters),
        ctx = Depends(get_model_ctx)
    ):
        async with ctx:
            turns = await (
                Turn.query() 
                .include(Artifact)
                .agg("forked_branches", Branch.query(["id"]), on=("id", "forked_from_turn_id"))
                .where(status = TurnStatus.COMMITTED)
                .limit(10) 
                .offset(0) 
                .order_by("-created_at") 
                .json()
            )
                
                
            # filter_artifacts = lambda kind: [a for turn in turns for a in turn["artifacts"] if a["kind"] == kind]
            filter_art_ids = lambda kind: [a["id"] for turn in turns for a in turn["artifacts"] if a["kind"] == kind]

            async def get_spans(art_ids: list[int]):
                if not art_ids:
                    return {}
                spans = await ExecutionSpan.query().where(lambda t: t.artifact_id.isin(art_ids)).include(SpanValue).json()
                return {s["artifact_id"]: s for s in spans}

            async def get_logs(art_ids: list[int]):
                if not art_ids:
                    return {}
                logs = await Log.query().where(lambda t: t.artifact_id.isin(art_ids)).json()
                return {l["artifact_id"]: l for l in logs}


            blocks_lookup = await get_blocks(filter_art_ids('block'))
            log_lookup = await get_logs(filter_art_ids('log'))
            spans_lookup = await get_spans(filter_art_ids('span'))

            visited = set()


            root_spans = []
            visited = set()

            for turn in turns:
                root_span = None
                for art in turn['artifacts']:
                    if art['kind'] == 'span':
                        span = spans_lookup[art['id']]
                        if span['parent_span_id'] is None:
                            if root_span is not None:
                                raise ValueError("Multiple root spans found")                
                            root_span = span
                            root_spans.append(span['id'])
                turn['span'] = root_span
                del turn['artifacts']                

            def populate_span_values(span, depth=0, max_depth=2):
                if span['id'] in visited:
                    raise ValueError(f"Circular reference detected: {span['id']}")
                visited.add(span['id'])    
                for value in span['values']:
                    if value['kind'] == "block":
                        value['artifact'] = blocks_lookup[value['artifact_id']]
                    elif value['kind'] == "log":
                        value['artifact'] = log_lookup[value['artifact_id']]
                    elif value['kind'] == "span":
                        value['artifact'] = spans_lookup[value['artifact_id']]
                        vs = value['artifact']
                        print(depth*" ", "populating value:", value["id"], "span:", vs["id"], f"(parent {vs["parent_span_id"]})")
                        populate_span_values(vs, depth+1, max_depth)
                
            for turn in turns:
                if turn['span']:
                    populate_span_values(turn['span'])

            return turns



    @turn_router.get("/spans2")   
    async def get_turn_blocks(
        list_params: ListParams = Depends(get_list_params),
        filters: QueryListType | None = Depends(query_filters),
        ctx = Depends(get_model_ctx)
    ):
        async with ctx:
            turns = await Turn.query().include(
                        ExecutionSpan.query(alias="es").select("*").include(
                            SpanValue
                        )
                ) \
                .agg("forked_branches", Branch.query(["id"]), on=("id", "forked_from_turn_id")) \
                .where(status = TurnStatus.COMMITTED) \
                .limit(10) \
                .offset(0) \
                .order_by("-created_at") \
                .json()

            block_ids = []
            log_ids = []
            spans_lookup = {}
            for turn in turns:
                for span in turn['spans']:
                    if span['parent_span_id'] is None:
                        root_span = span
                    spans_lookup[str(span['id'])] = span
                    for event in span['events']:
                        if event['event_type'] == 'block':
                            block_ids.append(event['event_id'])
                        elif event['event_type'] == 'log':
                            log_ids.append(int(event['event_id']))

            blocks_lookup = {}
            if block_ids:
                blocks_lookup = await get_blocks(block_ids)
                
                
            log_lookup = {}
            if log_ids:
                logs = await Log.query().where(lambda l: l.id.isin(log_ids)).json()
                log_lookup = {l['id']: l for l in logs}



            for turn in turns:
                root_span = None
                for span in turn['spans']:
                    if span['parent_span_id'] is None:
                        root_span = span
                    for event in span['events']:
                        if event['event_type'] == 'block':
                            event['data'] = blocks_lookup[event['event_id']]
                        elif event['event_type'] == 'log':
                            event['data'] = log_lookup[int(event['event_id'])]
                        elif event['event_type'] == 'span':
                            event['data'] = spans_lookup[event['event_id']]
                
                if root_span is None and len(turn['spans']) > 0:
                    raise ValueError("No root span found")
                turn['span'] = root_span
                del turn['spans']
                

            return turns    
        
        
        
        
    return turn_router