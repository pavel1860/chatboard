from codecs import lookup
import asyncio
from typing import Any, Iterator, TYPE_CHECKING

from ..utils.type_utils import SerializableType, serialize_value, type_to_str_or_none
from ..model.versioning.models import ArtifactKindEnum, Turn, Branch, ExecutionSpan, SpanType, DataFlowNode, Artifact, DataArtifact, ValueIOKind, Parameter, Log, VersionedModel
from ..block import BlockList, Block
from ..model.block_models.block_log import insert_block, get_blocks

from collections import defaultdict

# spans = await ExecutionSpan.query().include(Artifact.query().where(turn_id = 3)).print()
# spans = await ExecutionSpan.query()
# spans[0].artifact

def is_artifact_list(target_list: Any) -> bool:
    for item in target_list:
        if isinstance(item, VersionedModel):
            return True
        elif isinstance(item, Block):
            return True

    return False
            

class DataFlow:

    def __init__(self, span_value: DataFlowNode, value: Any, container: Artifact | None = None):
        self.span_value = span_value
        self._value = value
        self._is_list = span_value.kind == "list"
        self._is_span = span_value.kind == "span"
        # For single artifacts, check if it's a parameter
        self._container = container
        self.path = [int(p) for p in span_value.path.split(".")]
            
    
    def _get_value(self, value: Any):
        if isinstance(value, Parameter):
            return value.value
        return value
    
    def get_artifact_kind(self, value: Artifact):
        if value.kind == "parameter":
            return value
        return value.kind
    
    @property
    def kind(self):
        if self._is_list:
            return [self.get_artifact_kind(v) for v in self.span_value.artifacts if v.kind != "list"]
            
        return self.span_value.artifacts[0].kind

    
    @property
    def value(self):
        if self._is_list:
            return [self._get_value(v) for v in self._value]
        return self._get_value(self._value)

    # @property
    # def value(self):
    #     if self._is_list:
    #         # Value is already a list from instantiate_values
    #         return self._value
    #     elif self._is_span:
    #         # Return SpanTree directly for child spans
    #         return self._value
    #     elif self._is_parameter:
    #         return self._value.value
    #     return self._value

    @property
    def id(self):
        return self.span_value.id

    @property
    def io_kind(self):
        return self.span_value.io_kind

    @property
    def artifact_id(self):
        # For lists, return container artifact id
        if self._is_list:
            if not self._container:
                raise ValueError("Container is not set")
            return self._container.id
        # For spans, return the span's artifact_id
        if self._is_span:
            return self._value.root.artifact_id if isinstance(self._value, SpanTree) else None
        # For single artifacts
        return self.span_value.artifacts[0].id if self.span_value.artifacts else None

    @property
    def span_tree(self) -> "SpanTree | None":
        """Get as SpanTree if this is a span value."""
        return self._value if self._is_span else None

    # @property
    # def path(self) -> list[int]:
    #     """Get the LTREE path of this value."""
    #     return [int(p) for p in self.span_value.path.split(".")]
    
    @property
    def str_path(self) -> str:
        return self.span_value.path

    @property
    def name(self) -> str | None:
        """Get the name of this value."""
        return self.span_value.name

    def __repr__(self):
        return f"DataFlow(id={self.id}, io_kind={self.io_kind}, artifact_id={self.artifact_id}, value={self.value})"


class SpanTree:
        
    # def __init__(self, span: ExecutionSpan, index: int = 0, children: list[ExecutionSpan] | None = None, parent: "SpanTree | None" = None):
    def __init__(
        self,
        target: str | ExecutionSpan,
        span_type: SpanType = "component",
        tags: list[str] = [],
        index: int = 0,
        children: "list[SpanTree] | None" = None,
        parent: "SpanTree | None" = None,
        values: list[DataFlow] | None = None
    ):
        """
        Initialize a span tree.

        Note: When creating a new span from string name, the caller must handle
        path computation and call .save() to persist the ExecutionSpan.
        This __init__ is primarily for wrapping existing ExecutionSpan instances.
        """
        self.parent = parent
        self.index = index
        if isinstance(target, ExecutionSpan):
            # Wrapping existing span
            self.root = target
        else:
            # Creating new span
            if parent is None:
                # Top-level span - path will be computed in save() using turn context
                # Use placeholder path "0" that will be replaced
                self.root = ExecutionSpan(
                    name=target,
                    span_type=span_type,
                    tags=tags,
                    path="0",  # Placeholder - will be updated in save()
                    parent_span_id=None
                )
            else:
                # Child span - compute path from parent
                child_index = len(parent.children) if parent else 0
                path = f"{parent.root.path}.{child_index}"
                self.root = ExecutionSpan(
                    name=target,
                    span_type=span_type,
                    tags=tags,
                    path=path,
                    parent_span_id=parent.id if parent else None
                )

        self._lookup = {}
        self._children = children or []
        self._values = values or []
        self._value_index = 0  # Track next value index
        self.need_to_replay = False

        
    async def save(self):
        """
        Save the span tree to the database.

        For top-level spans created from string names, the path should be
        computed by the Context before calling save().
        """
        # Check if this is a top-level span with placeholder path
        if self.parent is None and self.root.parent_span_id is None and self.root.path == "0":
            # Get turn from context and use its span counter
            from .context import Context
            ctx = Context.current()
            if ctx is None or ctx.turn is None:
                raise ValueError("Cannot create top-level span outside of turn context")

            # Get next span index from context counter
            span_index = ctx.get_next_top_level_span_index()
            self.root.path = str(span_index)  # "0", "1", "2", ...
            self.index = span_index

            # Register this top-level span with the Context
            ctx._top_level_spans.append(self)

        await self.root.save()
        return self

    @property
    def id(self):
        return self.root.id
    
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return [self.index]
        if self.index is None:
            return self.parent.path
        return self.parent.path + [self.index]
    
    
    @property
    def str_path(self) -> str:
        if self.parent is None:
            return str(self.index)
        if self.index is None:
            return self.parent.str_path
        return self.parent.str_path + "." + str(self.index)
    
    @property
    def name(self):
        return self.root.name
    
    @property
    def values(self):
        return self._values

    @property
    def children(self) -> list["SpanTree"]:
        """Computed property - extracts child spans from values list."""
        return [v.span_tree for v in self._values if v._is_span and v.span_tree is not None]

    @property
    def inputs(self):
        return [v for v in self.values if v.io_kind == "input"]

    @property
    def outputs(self):
        return [v for v in self.values if v.io_kind == "output"]
    
    @property
    def span_type(self):
        return self.root.span_type
    
    @property
    def status(self):
        return self.root.status
    
    @property
    def start_time(self):
        return self.root.start_time
    
    @property
    def end_time(self):
        return self.root.end_time
    
    @property
    def branch_id(self):
        if self.root.artifact is None:
            raise ValueError("Root artifact is not set")
        return self.root.artifact.branch_id
    
    @property
    def turn_id(self):
        if self.root.artifact is None:
            raise ValueError("Root artifact is not set")
        return self.root.artifact.turn_id
    
    def get_last(self):
        last = self
        children = self.children
        while children:
            last = children[-1]
            children = last.children
        return last
    
    def traverse(self) -> "Iterator[SpanTree]":
        yield self
        for child in self.children:
            yield from child.traverse()
            
    def get(self, path: list[int]) -> "SpanTree | None":
        if len(path) == 0:
            return self
        if len(path) == 1:
            return self.children[path[0]]
        return self.children[path[0]].get(path[1:])

    def get_value_by_path(self, path: list[int]) -> DataFlow | None:
        """
        Get a value by its LTREE path string.

        Args:
            path: LTREE path string (e.g., "1.0.2")

        Returns:
            Value at that path, or None if not found
        """
        # Check if this is a value in the current span
        for value in self.values:
            if value.path == path:
                return value

        # Check children recursively
        for child in self.children:
            result = child.get_value_by_path(path)
            if result:
                return result

        return None

    def get_input_args(self):
        return [v.value for v in self.inputs]
        
    @classmethod
    async def load_span_list(cls, span_list: list[ExecutionSpan], value_dict: dict[str, dict[int, Any]]):
        """
        Load span trees from a list of ExecutionSpan instances.
        Returns list of top-level SpanTrees (no single root).
        """
        from ..model.versioning.models import DataArtifact

        lookup = defaultdict(list)
        top_level_spans = []

        for span in span_list:
            if span.parent_span_id is None:
                top_level_spans.append(span)
            else:
                lookup[span.parent_span_id].append(span)

        async def populate_children(span: SpanTree):
            children = lookup.get(span.id)

            # Reconstruct values for this span
            span._values = []
            for v in span.root.values:
                if v.kind == "list":
                    container = value_dict["list"][v.artifact_id]
                    if container is None:
                        raise ValueError("Container is not set")
                    items = []
                    for artifact in v.artifacts:
                        if artifact.kind == "list":
                            continue
                        value = value_dict[artifact.model_name][artifact.id]
                        items.append(value)
                    span._values.append(DataFlow(v, items, container))
                elif v.kind == "span":
                    # Create SpanTree for child span (not just ExecutionSpan)
                    # Find the child ExecutionSpan by artifact_id
                    child_exec_span = value_dict.get("execution_spans", {}).get(v.artifacts[0].id if v.artifacts else None)
                    if child_exec_span:
                        # Create SpanTree wrapper for the child
                        child_span_tree = SpanTree(child_exec_span, parent=span)
                        # Recursively populate this child
                        await populate_children(child_span_tree)
                        span._values.append(DataFlow(v, child_span_tree))
                    else:
                        # Fallback: just store ExecutionSpan if not found
                        span._values.append(DataFlow(v, child_exec_span))
                else:
                    # Single artifact
                    artifact = v.artifacts[0] if v.artifacts else None
                    if artifact:
                        model = value_dict.get(artifact.model_name, {}).get(artifact.id)
                        span._values.append(DataFlow(v, model))

            # Note: We don't use the children lookup anymore - children come from span values
            # But we still need to process orphaned children for backward compatibility
            if children:
                for i, c in enumerate(children):
                    # Check if this child is already in values as a span
                    already_in_values = any(
                        v._is_span and v._value and getattr(v._value.root if isinstance(v._value, SpanTree) else v._value, 'id', None) == c.id
                        for v in span._values
                    )
                    if not already_in_values:
                        # Orphaned child - create SpanTree and recursively populate
                        child_span_tree = SpanTree(c, index=i, parent=span)
                        await populate_children(child_span_tree)
                        # Note: Not adding to values since there's no SpanValue for it

            return span

        # Populate all top-level spans
        result_spans = []
        for i, top_span in enumerate(top_level_spans):
            span_tree = SpanTree(top_span, parent=None, index=i)
            await populate_children(span_tree)
            result_spans.append(span_tree)

        if len(result_spans) == 0:
            raise ValueError("No spans found")

        # Always return list of SpanTrees (even if single span)
        return result_spans
    
    
    @classmethod
    async def gather_artifacts(cls, spans, branch_id: int | None = None, span_lookup: dict[int, ExecutionSpan] | None = None):
        from ..model import NamespaceManager
        from ..model.versioning.models import DataArtifact

        model_ids = defaultdict(list)
        value_dict = {"execution_spans": {}}

        # First, collect all artifact IDs we need to load
        for s in spans:
            for v in s.values:
                if span_lookup and v.kind == "span":
                    # Handle ExecutionSpan references
                    value_dict["execution_spans"][v.artifact_id] = span_lookup[v.artifact_id]
                # elif v.kind == "list":
                    # For list containers, load artifacts via junction table
                    # value_artifacts = await ValueArtifact.query().where(value_id=v.id).order_by("position")
                    # for va in value_artifacts:
                    #     artifact = va.artifact if hasattr(va, 'artifact') else await Artifact.query().where(id=va.artifact_id).one()
                    #     model_ids[artifact.model_name].append(artifact.id)
                    # for artifact in v.artifacts:
                        # model_ids[artifact.model_name].append(artifact.id)
                else:
                    # Single artifact - load via artifacts relation
                    if v.artifacts is None:
                        raise ValueError("Artifacts are not set")
                    for artifact in v.artifacts:
                        if artifact.model_name is None:
                            if artifact.kind == "list":
                                model_ids[artifact.kind].append(artifact.id)
                            else:
                                raise ValueError("Model name is not set")
                        else:                                
                            model_ids[artifact.model_name].append(artifact.id)

        # Load all models in batches by model_name
        for k in model_ids:
            if k == "list":
                models = await Artifact.query(branch=branch_id).where(lambda a: a.id.isin(model_ids[k]))
                value_dict["list"] = {m.id: m for m in models}
            elif k == "block_trees":
                models = await get_blocks(model_ids[k], dump_models=False)
                value_dict[k] = models
            elif k == "execution_spans":
                value_dict[k] = {s.artifact_id: s for s in spans}
            else:
                ns = NamespaceManager.get_namespace(k)
                models = await ns._model_cls.query(branch=branch_id).where(lambda m: m.artifact_id.isin(model_ids[k]))
                value_dict[k] = {m.artifact_id: m for m in models}

        return value_dict

    
    
    @classmethod
    async def from_turn(cls, turn_id: int, span_id: int | None = None, branch_id: int | None = None):
        return await cls._from_turn(turn_id, span_id,branch_id, copy=False, skip_last=False)
    
    @classmethod
    async def replay_from_turn(cls, turn_id: int, span_id: int | None = None, branch_id: int | None = None):
        return await cls._from_turn(turn_id, span_id, branch_id, copy=True, skip_last=True)
        
    @classmethod
    async def _from_turn(cls, turn_id: int, span_id: int | None = None, branch_id: int | None = None, copy: bool = False, skip_last: bool = False):
        
        spans_query = (
            ExecutionSpan.query(
                turn_cte = Turn.query(branch=branch_id).where(lambda t: t.id.isin([turn_id])),                
            )
            .include(Artifact)             
            .order_by("artifact_id")
        )
        if span_id is not None:
            target_span = await ExecutionSpan.query(branch=branch_id).where(id=span_id).one()
            spans_query = (
                spans_query.where(lambda s: s.artifact_id <= target_span.artifact_id)
                .include(
                    DataFlowNode.query()
                    .where(lambda v: v.artifact_id <= target_span.artifact_id)
                    .include(Artifact)
                )
            )
        else:
            spans_query = spans_query.include(DataFlowNode.query().include(Artifact))           
        spans = await spans_query
        if not spans:
            raise ValueError(f"No spans found for turn {turn_id} branch {branch_id} span {span_id}")
        span_lookup = None
        if copy:
            parent_translation = {}
            span_lookup_artifact = {}   
            span_lookup = {}         
            for s in spans:
                prev_id = s.id
                prev_artifact_id = s.artifact_id
                s.id = None
                s.artifact_id = None
                if s.parent_span_id is not None:
                    s.parent_span_id = parent_translation[s.parent_span_id]
                s = await s.save()
                parent_translation[prev_id] = s.id
                span_lookup_artifact[prev_artifact_id] = s

            for i, s in enumerate(spans):
                span_values = []
                for v in s.values:
                    if skip_last and i == len(spans) - 1 and v.io_kind == "output":
                        continue
                    v.id = None
                    v.span_id = s.id
                    # v = await v.save()
                    if v.kind == "span":
                        v.artifact_id = span_lookup_artifact[v.artifact_id].artifact_id

                    span_values.append(v)
                span_values = await asyncio.gather(*[s.save() for s in span_values])
                span_lookup[s.artifact_id] = s
                s.values = span_values
        
        values = await cls.gather_artifacts(spans, branch_id, span_lookup)
        span_trees = await cls.load_span_list(spans, values)

        # Mark spans for replay if needed
        for span_tree in span_trees:
            last_span = span_tree.get_last()
            while last_span is not None:
                last_span.need_to_replay = True
                last_span = last_span.parent

        return span_trees
    
    
    
    def _build_parameter(self, value: SerializableType) -> Parameter | None:
        if isinstance(value, Parameter):
            return value
        else:     
            kind = type_to_str_or_none(type(value))
            if kind is None:
                return None
            return Parameter(data={"value": serialize_value(value)}, kind=kind)

    
    
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
    def _get_target_meta(self, target: Any) -> tuple[ArtifactKindEnum, int | None]:
        from ..block import Block
        if isinstance(target, Block):
            return "block", None
        elif isinstance(target, Log):
            return "log", target.artifact_id
        elif isinstance(target, SpanTree):
            # Handle SpanTree (extract ExecutionSpan for artifact_id)
            return "span", target.root.artifact_id
        elif isinstance(target, ExecutionSpan):
            if target == self:
                print(f"target == self {target.id} {self.id}")
            return "span", target.artifact_id
        elif isinstance(target, VersionedModel):
            return "model", target.artifact_id
        else:
            return "parameter", None
           
    
    
    
        
        
    def _sanitize_target_value(self, target: Any) -> tuple[VersionedModel, ArtifactKindEnum, int | None]:
        kind, artifact_id = self._get_target_meta(target)
        if kind == "block":
            return target, kind, artifact_id
        elif kind == "parameter":
            param = self._build_parameter(target)
            if param is None:
                raise ValueError(f"Target '{target}' cannot be logged as a parameter")
            return param, kind, artifact_id
        elif kind == "span":
            # Keep SpanTree as-is (don't convert to ExecutionSpan)
            return target, kind, artifact_id
        else:
            return target, kind, artifact_id
        
    def _sanitize_target_list_value(self, target: Any) -> list[VersionedModel]:
        if isinstance(target, list) and is_artifact_list(target):
            container_artifact = Artifact(
                branch_id=self.branch_id,
                turn_id=self.turn_id,
                kind="list",
                model_name=target[0].__class__.__name__,  # Model type of items
            )
            
            
    async def log_value(self, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None):
        # Compute path for this value using in-memory counter
        if io_kind == "output":
            value_path = f"{self.root.path}.{self._value_index}"
            self._value_index += 1  # Increment for next value
        elif io_kind == "input":
            value_path = self.root.path
        else:
            raise ValueError(f"Invalid io_kind: {io_kind}")

        if isinstance(target, list) and is_artifact_list(target):
            container_artifact = await Artifact(
                branch_id=self.branch_id,
                turn_id=self.turn_id,
                span_id=self.id,  # NEW: Track creation context
                kind="list",
            ).save()

            value = await self.root.add(DataFlowNode(
                span_id=self.id,
                kind="list",
                alias=alias,
                io_kind=io_kind,
                name=name,
                path=value_path,  # NEW: Set path
                artifact_id=container_artifact.id,
            ))

            await value.add(container_artifact)
            
            list_artifacts = []
            for position, item in enumerate(target):
                item, kind, artifact_id = self._sanitize_target_value(item)
                if kind == "block":
                    block_item = await insert_block(item, self.branch_id, self.turn_id, self.id)
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
                    position=position,
                ).save()
                
            value.artifacts = list_artifacts

            v = DataFlow(value, list_artifacts, container_artifact)
            self._values.append(v)
            return v

        else:
            target, kind, artifact_id = self._sanitize_target_value(target)
            if kind == "block":
                target = await insert_block(target, self.branch_id, self.turn_id, self.id)
                artifact_id = target.artifact_id
            elif kind == "span":
                # For spans, get artifact from SpanTree or ExecutionSpan
                if isinstance(target, SpanTree):
                    artifact = target.root.artifact
                else:  # ExecutionSpan
                    artifact = target.artifact
                value = await self.root.add(DataFlowNode(
                    span_id=self.id,
                    kind=kind,
                    alias=alias,
                    io_kind=io_kind,
                    name=name,
                    path=value_path,  # NEW: Set path
                    artifact_id=artifact_id,
                ))
                await value.add(artifact)
                v = DataFlow(value, target)
                return self._append_value(value, target)
            elif artifact_id is None:
                await target.save()
                artifact_id = target.artifact_id

            value = await self.root.add(DataFlowNode(
                span_id=self.id,
                kind=kind,
                alias=alias,
                io_kind=io_kind,
                name=name,
                path=value_path,  # NEW: Set path
                artifact_id=artifact_id,
            ))
            value.artifacts = [target.artifact]
            await value.add(target.artifact)
            v = DataFlow(value, target)
            return self._append_value(value, target)
            
    
    
    async def log_value2(self, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None):
        """
        Log a value to the span.

        Args:
            target: The value to log (can be a single artifact or list of artifacts)
            alias: Optional alias for the value
            io_kind: Whether this is an input or output
            name: Optional parameter name for function kwargs
        """
        # Handle list of artifacts
        if isinstance(target, list) and is_artifact_list(target):
            # Check if it's a list of versioned models/artifacts            
            # Create a container artifact for the list
            container_artifact = await Artifact(
                branch_id=self.branch_id,
                turn_id=self.turn_id,
                kind="list",
                model_name=target[0].__class__.__name__,  # Model type of items
            ).save()

            # Create container SpanValue
            value = await self.root.add(DataFlowNode(
                span_id=self.id,
                kind="list",
                alias=alias,
                io_kind=io_kind,
                name=name,
                artifact_id=container_artifact.id,
            ))

            # Create ValueArtifact entries for each item
            list_artifacts = []
            for position, item in enumerate(target):
                va = await DataArtifact(
                    value_id=value.id,
                    artifact_id=item.artifact_id,
                    position=position,
                ).save()
                list_artifacts.append(va)

            return self._append_value(value, container_artifact, list_artifacts)

        # Handle single value (existing logic)
        kind, artifact_id = self._get_target_meta(target)
        if kind == "block":
            return await self.add_block_event(target, io_kind)
        elif kind == "parameter":
            param = self._build_parameter(target)
            if param is None:
                return None
            await param.save()
            artifact_id = param.artifact.id
            target = param

        try:
            value = await self.root.add(DataFlowNode(
                span_id=self.id,
                kind=kind,
                alias=alias,
                io_kind=io_kind,
                name=name,
                artifact_id=artifact_id,
            ))
            value.artifacts = [target.artifact]
            return self._append_value(value, target)
        except Exception as e:
            print(f"Error logging value: {e}")
            raise e
    
    def _append_value(self, value: DataFlowNode, target: Any, list_artifacts: list[Any] | None = None):
        v = DataFlow(value, target, list_artifacts)
        self._values.append(v)
        return v
        
    async def add_block_event(self, block: "Block", io_kind: ValueIOKind= "output"):
        from ..model.block_models.block_log import insert_block
        # if self._should_save_to_db():
        #     tree_id = await insert_block(block, self.artifact.branch_id, self.artifact.turn_id, self.id)
        # else:
        #     tree_id = str(uuid.uuid4())
        block_tree = await insert_block(block, self.root.artifact.branch_id, self.root.artifact.turn_id, self.id)
            
        value = await self.root.add(DataFlowNode(
            span_id=self.id,
            kind="block",
            io_kind=io_kind,
            artifact_id=block_tree.artifact.id,
        ))
        return value


    
    
    async def add_child(self, name: str, span_type: SpanType = "component", tags: list[str] = []):
        """
        Add a child span to the current span by logging it as a span value.
        The child will be accessible via the computed 'children' property.
        """
        # Compute child path
        child_index = len(self.children)
        child_path = f"{self.root.path}.{child_index}"

        # Create child ExecutionSpan directly
        child_span = await ExecutionSpan(
            name=name,
            span_type=span_type,
            tags=tags,
            path=child_path,
            parent_span_id=self.id
        ).save()

        # Wrap in SpanTree
        span_tree = SpanTree(child_span, parent=self, index=child_index)

        # Log the SpanTree as a value - this adds it to _values
        await self.log_value(span_tree, io_kind="output")

        return span_tree
    
    def print_tree(self):
        for s in self.traverse():
            print(s.id, s.path, s.name)
            for v in s.inputs:
                print(">>  ", v.id, v.io_kind, v.artifact_id, v.path, ":", v.value)
            for v in s.outputs:
                print("<<  ", v.id, v.io_kind, v.artifact_id, v.path, ":", v.value)

    def to_dict(self) -> dict:
        """Convert SpanTree to JSON-serializable dict with new junction table architecture."""
        result = {
            "id": self.id,
            "name": self.name,
            "path": self.root.path,
            "span_type": self.root.span_type,
            "status": self.root.status,
            "tags": self.root.tags,
            "start_time": self.root.start_time.isoformat() if self.root.start_time else None,
            "end_time": self.root.end_time.isoformat() if self.root.end_time else None,
            "values": []
            # Note: No separate "children" field - children are serialized within values as span values
        }

        # Convert values with junction table support
        for v in self.values:
            value_data = {
                "id": v.span_value.id,
                "io_kind": v.io_kind,
                "kind": v.kind,
                "name": v.span_value.name,  # Kwarg name
                "artifact_id": v.artifact_id,  # Container or single artifact
            }

            # Serialize the actual value
            if v._is_span:
                # Child span - recursively serialize the SpanTree
                if isinstance(v.value, SpanTree):
                    value_data["value"] = v.value.to_dict()
                else:
                    # Fallback for ExecutionSpan (shouldn't happen with new architecture)
                    value_data["value"] = {"id": str(v.value.id) if v.value else None}
            elif v._is_list:
                # List of artifacts - serialize each item
                serialized_items = []
                for item in v.value:
                    if hasattr(item, 'to_dict'):
                        serialized_items.append(item.to_dict())
                    elif hasattr(item, 'model_dump'):
                        serialized_items.append(item.model_dump())
                    elif isinstance(item, (str, int, float, bool, type(None), list, dict)):
                        serialized_items.append(item)
                    else:
                        serialized_items.append(str(item))
                value_data["value"] = serialized_items
            else:
                # Single value - use consistent serialization
                raw_value = v.value
                if hasattr(raw_value, 'to_dict'):
                    value_data["value"] = raw_value.to_dict()
                elif hasattr(raw_value, 'model_dump'):
                    value_data["value"] = raw_value.model_dump()
                elif isinstance(raw_value, (str, int, float, bool, type(None), list, dict)):
                    value_data["value"] = raw_value
                else:
                    value_data["value"] = str(raw_value)

            result["values"].append(value_data)

        return result
                
                
    def __repr__(self):
        return f"SpanTree(id={self.id}, name={self.name}, path={self.path}, span_type={self.span_type}, status={self.status}, start_time={self.start_time}, end_time={self.end_time})"
                
                
                
                
                
                
                
                
