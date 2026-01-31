from contextlib import asynccontextmanager
import datetime as dt
from typing import TYPE_CHECKING, AsyncGenerator, Callable, List, Literal, Type, TypeVar, Self, Any
from promptview.model.base.types import ArtifactKind
from ..model.model3 import Model
from ..model.fields import KeyField, ModelField, RelationField
from .models import VersionedModel, Artifact, ArtifactKindEnum, Turn, SpanType
from .block_storage import BlockModel

from ..llms.types import LlmConfig, LLMUsage
from ..block.block12.chunk import BlockChunk

if TYPE_CHECKING:
    from ..prompt.context import Context
    










class Log(VersionedModel):
    _artifact_kind: ArtifactKind = "log"
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    message: str = ModelField()
    level: Literal["info", "warning", "error"] = ModelField()
    



ValueIOKind = Literal["input", "output"]


class DataArtifact(Model):
    id: int = KeyField(primary_key=True)  # Auto-increment ID
    value_id: int = ModelField(foreign_key=True)
    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    kind: ArtifactKindEnum = ModelField()
    position: int | None = ModelField(default=None)  # For lists/tuples - index in collection
    name: str | None = ModelField(default=None)  # For dicts - key name
    

class DataFlowNode(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    kind: ArtifactKindEnum = ModelField()
    io_kind: ValueIOKind = ModelField()
    path: str = ModelField(db_type="LTREE")

    # Parent: exactly one of span_id or turn_id must be set
    span_id: int | None = ModelField(foreign_key=True, default=None)
    turn_id: int | None = ModelField(foreign_key=True, foreign_cls=Turn, default=None)

    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    artifacts: list[Artifact] = RelationField(
        primary_key="id",
        junction_keys=["value_id", "artifact_id"],
        foreign_key="id",
        junction_model=DataArtifact,
    )
    artifact_data: list[DataArtifact] = RelationField(
        primary_key="id",
        foreign_key="value_id",
    )
    
    alias: str | None = ModelField(default=None)
    name: str | None = ModelField(default=None)  # Keyword argument name (e.g., "count", "items")
    
    
    _value: Any | None = None
    _container_value: Artifact | None = None
    
    
    @property
    def value(self) -> Any:
        from ..block import Block
        if self._value is None:
            raise ValueError(f"no value for DataFlowNode {self.id}")
        
        def get_value(target: Any) -> Any:
            if isinstance(target, Block):
                return target
            if target.kind == "parameter":
                return target._value.value
            elif target.kind == "block":
                return target._value
            return target._value
        if self.kind == "list":
            return [get_value(v) for v in self._value]
        return get_value(self)
    
    @property
    def list_kind(self) -> list[str] | None:
        from ..block import Block
        if self.kind != "list" or self._value is None:
            return None
        result = []
        for v in self._value:
            if isinstance(v, Block):
                result.append("block")
            elif hasattr(v, "kind"):
                result.append(v.kind)
            else:
                result.append(type(v).__name__)
        return result

    @property
    def value_or_none(self) -> Any | None:
        return self._value

    @property
    def index(self) -> int:
        """Extract index from path (last segment)"""
        return int(self.path.split('.')[-1])
    
    
    def extract(self) -> Any:
        """ turns blocks into simplified blocks """
        from ..block import Block
        if self.kind == "block":
            self._value = self._value.extract()
        elif self.kind == "list":
            new_list = []
            for t, v in zip(self.list_kind, self._value):
                if t == "block":
                    new_list.append(v.extract())
                else:
                    new_list.append(v)
            self._value = new_list
        return self._value
    
    def model_dump(self, *args, **kwargs) -> dict:
        dump = super().model_dump(*args, **kwargs)
        dump["list_kind"] = None
        if self.kind == "list":
            if self._value is None:
                dump["value"] = []
                dump["list_kind"] = []
            else:
                dump["value"] = [v.model_dump() for v in self.value]
                dump["list_kind"] = self.list_kind
        elif self._value is not None and hasattr(self._value, "model_dump"):     
            dump["value"] = self._value.model_dump()
        elif self.kind == "parameter":
            dump["value"] = self._value
        else:
            dump["value"] = None
        return dump



class LlmCall(Model):
    """a single call to an llm"""
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    config: "LlmConfig" = ModelField()
    usage: "LLMUsage" = ModelField()
    chunks: list[BlockChunk] = ModelField()
    request_id: str = ModelField()
    message_id: str = ModelField()
    span_id: int = ModelField(foreign_key=True)
    
    
    @property
    def text(self) -> str:
        return "".join([c.content for c in self.chunks])
        
    def print(self, with_metadata: bool = True):
        if with_metadata:
            sep = "─" * 50
            print(sep)
            print(f"LLM Call  |  model: {self.config.model or 'N/A'}")
            print(sep)
            # Config
            config_parts = [f"temp={self.config.temperature}"]
            if self.config.max_tokens is not None:
                config_parts.append(f"max_tokens={self.config.max_tokens}")
            if self.config.top_p != 1:
                config_parts.append(f"top_p={self.config.top_p}")
            if self.config.stream:
                config_parts.append("stream=True")
            if self.config.tools:
                tool_names = [t.__name__ for t in self.config.tools]
                config_parts.append(f"tools=[{', '.join(tool_names)}]")
            if self.config.tool_choice is not None:
                config_parts.append(f"tool_choice={self.config.tool_choice}")
            print(f"  Config   : {' | '.join(config_parts)}")
            # Usage
            usage_parts = [
                f"input={self.usage.input_tokens}",
                f"output={self.usage.output_tokens}",
                f"total={self.usage.total_tokens}",
            ]
            if self.usage.cached_tokens:
                usage_parts.append(f"cached={self.usage.cached_tokens}")
            if self.usage.reasoning_tokens:
                usage_parts.append(f"reasoning={self.usage.reasoning_tokens}")
            print(f"  Tokens   : {' | '.join(usage_parts)}")
            # IDs
            print(f"  Request  : {self.request_id}")
            print(f"  Message  : {self.message_id}")
            print(f"  Created  : {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(sep)
        print(self.text)
  

class ExecutionSpan(VersionedModel):
    """Represents a single execution unit (component call, stream, etc.)"""
    _artifact_kind: ArtifactKind = "span"
    id: int = KeyField(primary_key=True)
    name: str = ModelField()  # Function/component name
    path: str = ModelField(db_type="LTREE")
    span_type: SpanType = ModelField()
    parent_span_id: int | None = ModelField(foreign_key=True, self_ref=True)
    start_time: dt.datetime = ModelField(default_factory=dt.datetime.now)
    end_time: dt.datetime | None = ModelField(default=None)
    tags: list[str] | None = ModelField(default=None)
    metadata: dict[str, Any] = ModelField(default={})
    # usage: dict[str, Any] = ModelField(default={})
    # config: dict[str, Any] = ModelField(default={})
    status: Literal["running", "completed", "failed"] = ModelField(default="running")
    turn_id: int = ModelField(foreign_key=True, foreign_cls=Turn)
    # request_id: str | None = ModelField(default=None)
    # message_id: str | None = ModelField(default=None)
    
    # Relations
    data: List["DataFlowNode"] = RelationField([], foreign_key="span_id")
    artifacts: List[Artifact] = RelationField(foreign_key="span_id")
    block_trees: List[BlockModel] = RelationField(foreign_key="span_id")
    llm_calls: List[LlmCall] = RelationField(foreign_key="span_id")
    
    _parent_value: "DataFlowNode | None" = None
    
    @property
    def parent_value(self) -> "DataFlowNode | None":
        return self._parent_value

    @property
    def inputs(self) -> dict[str, "DataFlowNode"]:
        return {v.path.split(".")[-1]: v.value for v in self.data if v.io_kind == "input"}
    
    @property
    def outputs(self) -> list["DataFlowNode"]:
        return [v.value for v in self.data if v.io_kind == "output"]
    
    @property
    def children(self) -> list["ExecutionSpan"]:
        return [v.value for v in self.data if v.kind == "span"]
    
    @property
    def ctx(self) -> "Context":
        from ..prompt.context import Context
        ctx = Context.current_or_none()
        if ctx is None:
            raise ValueError("Context is not set")
        return ctx
    
    
    async def log_value(self, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None):
        # from ...prompt.context import Context
        from .artifact_log import ArtifactLog
        return await ArtifactLog.log_value(target, alias, io_kind, name)
    
    
    
    # async def add_child(self, name: str, span_type: SpanType = "component", tags: list[str] = []):
    #     """
    #     Add a child span to the current span by logging it as a span value.
    #     The child will be accessible via the computed 'children' property.
    #     """
    #     # Compute child path
    #     from ...prompt.context import Context
    #     child_index = len(self.children)
    #     child_path = f"{self.path}.{child_index}"        
        

    #     # Create child ExecutionSpan directly
    #     child_span = await ExecutionSpan(
    #         name=name,
    #         span_type=span_type,
    #         tags=tags,
    #         path=child_path,
    #         parent_span_id=self.id
    #     ).save()        

    #     # Log the SpanTree as a value - this adds it to _values
    #     await self.ctx.artifact_log.log_value(self, child_span, io_kind="output")

    #     return child_span


# NOTE: model_rebuild() calls are in __init__.py after all models are imported

