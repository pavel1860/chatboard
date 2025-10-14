from typing import AsyncContextManager, Type, List
from fastapi import Depends, Request
from .model_router import create_model_router
from ..prompt.context import Context
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
        """
        Get turns with their span trees using the new SpanTree architecture.

        This endpoint uses SpanTree.from_turn() which handles:
        - Loading ExecutionSpans with their hierarchy
        - Loading SpanValues with the ValueArtifact junction table
        - Instantiating actual model instances for values
        - Building the complete tree structure
        """
        from ..prompt.span_tree import SpanTree

        async with ctx:
            # Get recent committed turns
            turns = await (
                Turn.query()
                .include(Artifact)
                .agg("forked_branches", Branch.query(["id"]), on=("id", "forked_from_turn_id"))
                .where(status = TurnStatus.COMMITTED)
                .limit(10)
                .offset(0)
                .order_by("-created_at")
            )

            # Load span tree for each turn
            result = []
            for turn in turns:
                turn_data = {
                    "id": turn.id,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None,
                    "status": turn.status,
                    "branch_id": turn.branch_id,
                    "forked_branches": [{"id": b.id} for b in getattr(turn, "forked_branches", [])],
                    "span": None
                }

                try:
                    # Load span tree using SpanTree.from_turn()
                    span_tree = await SpanTree.from_turn(turn.id, branch_id=turn.branch_id)

                    # Convert SpanTree to JSON-serializable format
                    turn_data["span"] = span_tree.to_dict()
                except Exception as e:
                    # If turn has no spans, just skip
                    print(f"Error loading span tree for turn {turn.id}: {e}")
                    turn_data["span"] = None

                result.append(turn_data)

            return result



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