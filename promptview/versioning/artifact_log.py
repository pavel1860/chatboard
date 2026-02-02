



from typing import List, TYPE_CHECKING


from codecs import lookup
import asyncio
from typing import Any, Iterator, TYPE_CHECKING

from ..utils.type_utils import SerializableType, serialize_value, type_to_str_or_none
from . import ArtifactKindEnum, Turn, Branch, ExecutionSpan, SpanType, DataFlowNode, Artifact, DataArtifact, ValueIOKind, Parameter, Log, VersionedModel
from ..block import BlockList, Block
# from ..block_models.block_log import store_block, get_blocks
from .block_storage import store_block, BlockModel as StoredBlockModel

from collections import defaultdict
from ..prompt.context import Context



INPUT_TAG = "0"

def is_artifact_list(target_list: Any) -> bool:
    for item in target_list:
        if isinstance(item, VersionedModel):
            return True
        elif isinstance(item, Block):
            return True

    return False




def _get_target_meta(target: Any) -> tuple[ArtifactKindEnum, int | None]:
    from ..block import Block
    if isinstance(target, Block):
        return "block", None
    elif isinstance(target, Log):
        return "log", target.artifact_id
    elif isinstance(target, ExecutionSpan):
        # Handle SpanTree (extract ExecutionSpan for artifact_id)
        return "span", target.artifact_id
    # elif isinstance(target, ExecutionSpan):
    #     if target == self:
    #         print(f"target == self {target.id} {self.id}")
    #     return "span", target.artifact_id
    elif isinstance(target, Parameter):
        return "parameter", target.artifact_id
    elif isinstance(target, VersionedModel):
        return target.get_namespace_name(), target.artifact_id
    else:
        return "parameter", None

def _build_parameter(value: SerializableType) -> Parameter | None:
    if isinstance(value, Parameter):
        return value
    else:     
        kind = type_to_str_or_none(type(value))
        if kind is None:
            return None
        return Parameter(data={"value": serialize_value(value)}, kind=kind)


def _sanitize_target_value(target: Any) -> tuple[VersionedModel, ArtifactKindEnum, int | None]:
    kind, artifact_id = _get_target_meta(target)
    if kind == "block":
        return target, kind, artifact_id
    elif kind == "parameter":
        param = _build_parameter(target)
        if param is None:
            raise ValueError(f"Target '{target}' cannot be logged as a parameter")
        return param, kind, artifact_id
    elif kind == "span":
        # Keep SpanTree as-is (don't convert to ExecutionSpan)
        return target, kind, artifact_id
    else:
        return target, kind, artifact_id
    
# def _sanitize_target_list_value(target: Any, branch_id: int, turn_id: int) -> list[VersionedModel]:
#     if isinstance(target, list) and is_artifact_list(target):
#         container_artifact = Artifact(
#             branch_id=branch_id,
#             turn_id=turn_id,
#             kind="list",
#             model_name=target[0].__class__.__name__,  # Model type of items
#         )

class ArtifactLog:
    

    
        
    @classmethod
    async def populate_turns(cls, turns: List[Turn]):
        from collections import defaultdict
        from ..model import NamespaceManager
        from .models import Artifact

        def kind2table(k: str):
            if k == "parameter":
                return "parameters"
            elif k == "block":
                return "blocks"  # New table name
            elif k == "span":
                return "execution_spans"
            return k

        models_to_load = defaultdict(list)

        for turn in turns:
            for value in turn.data:
                for da in value.artifact_data:
                    models_to_load[kind2table(da.kind)].append(da.artifact_id)

        model_lookup = {}

        for k in models_to_load:
            if k == "list":
                models = await Artifact.query(include_branch_turn=True).where(Artifact.id.isin(models_to_load[k]))
                model_lookup["list"] = {m.id: m for m in models}
            elif k == "blocks":
                # Use new StoredBlockModel and convert to Block
                block_models = await StoredBlockModel.query(include_branch_turn=True).where(
                    StoredBlockModel.artifact_id.isin(models_to_load[k])
                )
                model_lookup[k] = {bm.artifact_id: bm.to_block() for bm in block_models}
            else:
                ns = NamespaceManager.get_namespace(k)
                models = await ns._model_cls.query(include_branch_turn=True).where(ns._model_cls.artifact_id.isin(models_to_load[k]))
                model_lookup[k] = {m.artifact_id: m for m in models}

        for turn in turns:
            for value in turn.data:
                if value.kind == "list":
                    value._value = []
                    for da in value.artifact_data:
                        if da.kind == "list":
                            value._container_value = model_lookup[kind2table(da.kind)][da.artifact_id]
                        else:
                            value._value.append(model_lookup[kind2table(da.kind)][da.artifact_id])
                else:
                    value._value = model_lookup[kind2table(value.kind)][value.artifact_id]
        return turns    
    
    


    
    
    # async def log_value(self, value: Any, io_kind: ValueIOKind = "input", name: str | None = None):
    #     """
    #     Log a value to the current span and add it to the values list

    #     Args:
    #         value: The value to log (can be single artifact or list of artifacts)
    #         io_kind: Whether this is an input or output
    #         name: Optional parameter name for function kwargs
    #     """
    #     value = value.root if isinstance(value, SpanTree) else value
    #     value = await self.root.log_value(value, io_kind=io_kind, name=name)
    #     return value
    
    @classmethod
    def build_data_flow_node(cls, execution_span: ExecutionSpan, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None, ctx: Context | None = None):
        if io_kind == "output":
            value_path = f"{execution_span.path}.{len(execution_span.outputs) + 1}"
        elif io_kind == "input":
            value_path = f"{execution_span.path}.{INPUT_TAG}" 
            if name is not None:
                value_path += "." + name
        else:
            raise ValueError(f"Invalid io_kind: {io_kind}")
        
        _, kind, artifact_id = _sanitize_target_value(target)
        data_flow = DataFlowNode(
            span_id=None,
            kind=kind,
            alias=alias,
            io_kind=io_kind,
            name=name,
            path=value_path,
            artifact_id=-1,
        )
        data_flow._value = target
        return data_flow
            
    @classmethod
    async def log_value(cls, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None, ctx: Context | None = None):
        
        
        ctx = Context.current_or_none() if ctx is None else ctx
        if ctx is None:
            raise ValueError("Context is not set")
        span_id = None

        execution_span = ctx.current_span
        turn = ctx.turn
        if execution_span is None:
            # handle case when target is an ExecutionSpan and not span in context. add to root
            # value_path = str(ctx.get_next_top_level_span_index())
            value = await DataFlowNode(
                    span_id=None,
                    kind="span",
                    alias=alias,
                    io_kind=io_kind,
                    name=name,
                    path=target.path,  # NEW: Set path
                    artifact_id=target.artifact_id,
                ).save()
            value._value = target
            await value.add(target.artifact, kind="span")
            turn.data.append(value)
            return value
        span_id = execution_span.id
            
        # Compute path for this value using in-memory counter
        if io_kind == "output":
            value_path = f"{execution_span.path}.{len(execution_span.outputs) + 1}"
        elif io_kind == "input":
            value_path = f"{execution_span.path}.{INPUT_TAG}" 
            if name is not None:
                value_path += "." + name
        else:
            raise ValueError(f"Invalid io_kind: {io_kind}")

        if isinstance(target, list) and is_artifact_list(target):
            try:
                container_artifact = await Artifact(
                    branch_id=ctx.branch.id,
                    turn_id=ctx.turn.id,
                    span_id=span_id,  # NEW: Track creation context
                    kind="list",
                ).save()
            except Exception as e:
                raise e

            value = await execution_span.add(DataFlowNode(
                span_id=span_id,
                kind="list",
                alias=alias,
                io_kind=io_kind,
                name=name,
                path=value_path,  # NEW: Set path
                artifact_id=container_artifact.id,
            ))

            await value.add(container_artifact, kind="list")
            
            list_artifacts = []
            for position, item in enumerate(target):
                item, kind, artifact_id = _sanitize_target_value(item)
                if kind == "block":
                    block_item = await store_block(item, ctx.branch.id, ctx.turn.id, span_id, span=execution_span.name)
                    block_item._block = item
                    item = block_item
                    artifact_id = item.artifact_id
                elif artifact_id is None:
                    await item.save()
                    artifact_id = item.artifact_id                
                list_artifacts.append(item)
                va = await DataArtifact(
                    value_id=value.id,
                    artifact_id=artifact_id,
                    kind=kind,
                    position=position,
                ).save()
                
            value.artifacts = list_artifacts

            return value
            
        else:
            target, kind, artifact_id = _sanitize_target_value(target)
            if kind == "block":
                block_tree = await store_block(target, ctx.branch.id, ctx.turn.id, span_id, span=execution_span.name)
                artifact = block_tree.artifact
                artifact_id = block_tree.artifact_id
            elif kind == "span":
                # For spans, get artifact from SpanTree or ExecutionSpan
                artifact = target.artifact
                value = await execution_span.add(DataFlowNode(
                    span_id=span_id,
                    kind=kind,
                    alias=alias,
                    io_kind=io_kind,
                    name=name,
                    path=value_path,  # NEW: Set path
                    artifact_id=artifact_id,
                ))
                await value.add(artifact, kind=kind)
                value._value = target
                return value
            else:
                artifact = target.artifact
                artifact_id = target.artifact_id
                
            if artifact_id is None:
                await target.save()
                artifact = target.artifact
                artifact_id = target.artifact_id

            value = await execution_span.add(DataFlowNode(
                span_id=span_id,
                kind=kind,
                alias=alias,
                io_kind=io_kind,
                name=name,
                path=value_path,  # NEW: Set path
                artifact_id=artifact_id,
            ))
            value.artifacts = [artifact]
            await value.add(artifact, kind=kind)
            value._value = target
            return value
