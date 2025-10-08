from codecs import lookup
from typing import Any
from ..model.versioning.models import Turn, Branch, ExecutionSpan, SpanValue, Artifact, ValueIOKind


from collections import defaultdict

# spans = await ExecutionSpan.query().include(Artifact.query().where(turn_id = 3)).print()
# spans = await ExecutionSpan.query()
# spans[0].artifact

class Value:
    
    def __init__(self, span_value: SpanValue, value: Any):
        self.span_value = span_value
        self._value = value
        self._is_parameter = span_value.artifact.kind == "parameter"
        
    @property
    def value(self):
        if self._is_parameter:
            return self._value.value
        return self._value
        
    @property
    def id(self):
        return self.span_value.id
    
    @property
    def io_kind(self):
        return self.span_value.io_kind
    
    @property
    def artifact_id(self):
        return self.span_value.artifact_id


class SpanTree:
        
    # def __init__(self, span: ExecutionSpan, index: int = 0, children: list[ExecutionSpan] | None = None, parent: "SpanTree | None" = None):
    def __init__(
        self, 
        target: str | ExecutionSpan, 
        span_type: str = "component", 
        tags: list[str] = [], 
        index: int = 0, 
        children: list[ExecutionSpan] | None = None, 
        parent: "SpanTree | None" = None,
        values: list[Value] | None = None
    ):    
        """
        Initialize a span tree
        """
        self.parent = parent
        self.index = index
        if isinstance(target, ExecutionSpan):
            self.root = target
        else:
            path = ".".join([str(i) for i in self.path])
            self.root = ExecutionSpan(name=target, span_type=span_type, tags=tags, path=path, parent_span_id=parent.id if parent else None)
        self._lookup = {}
        self.children = children or []
        self._values = values or []

        
    async def save(self):
        """
        Save the span tree to the database
        """
        await self.root.save()
        return self
    
    @classmethod
    async def init_new(cls, name: str, span_type: str = "component", tags: list[str] = [], index: int = 0):
        span = await ExecutionSpan(name=name, span_type=span_type, tags=tags, path="0").save()
        return cls(span)

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
    def name(self):
        return self.root.name
    
    @property
    def values(self):
        return self._values
    
    @property
    def inputs(self):
        return [v for v in self.values if v.io_kind == "input"]
    
    @property
    def outputs(self):
        return [v for v in self.values if v.io_kind == "output"]
    
    def get_last(self):
        last = self
        children = self.children
        while children:
            last = children[-1]
            children = last.children
        return last
    
    def traverse(self):
        yield self
        for child in self.children:
            yield from child.traverse()
            
    def get(self, path: list[int]):
        if len(path) == 0:
            return self
        if len(path) == 1:
            return self.children[path[0]]
        return self.children[path[0]].get(path[1:])
        
    @classmethod
    def load_span_list(cls, span_list: list[ExecutionSpan], value_dict: dict[str, dict[int, Any]]):
        lookup = defaultdict(list)
        root = None
        for span in span_list:
            if span.parent_span_id is None:
                root = span
            else:
                lookup[span.parent_span_id].append(span)
        else:
            if root is None:
                raise ValueError("No root span found")
            
        
        def populate_children(span: SpanTree):
            children = lookup.get(span.id)
            # values = [value_dict.get(v.artifact.model_name).get(v.artifact_id) for v in span.root.values]
            # span._values = [Value(v, values[i]) for i, v in enumerate(span.values)]
            span._values = [
                Value(
                    v, 
                    value_dict.get(v.artifact.model_name).get(v.artifact_id)) 
                for v in span.root.values
            ]
            if children is None:
                return span
            span.children = [SpanTree(c, index=i, parent=span) for i, c in enumerate(children)]
            for child in span.children:
                populate_children(child)
            return span
        return populate_children(SpanTree(root))
    
        
    @classmethod
    async def from_turn(cls, turn_id: int, span_id: int | None = None):
        
        spans_query = (
            ExecutionSpan.query(
                turn_cte = Turn.query().where(lambda t: t.id.isin([turn_id]))
            )
            .include(Artifact)
            .include(SpanValue.query().include(Artifact))
            .order_by("artifact_id")
        )
        if span_id is not None:
            target_span = await ExecutionSpan.query().where(id=span_id).one()
            spans_query = spans_query.where(lambda s: s.artifact_id <= target_span.artifact_id)
        spans = await spans_query
        values = await cls.instantiate_values(spans)
        return cls.load_span_list(spans, values)
    
    @classmethod
    async def instantiate_values(cls, spans):
        from ..model import NamespaceManager
        model_ids = defaultdict(list)
        for s in spans:
            for v in s.values:
                model_ids[v.artifact.model_name].append(v.artifact.id)

        value_dict = {}
        for k in model_ids:
            ns = NamespaceManager.get_namespace(k)
            models = await ns._model_cls.query().where(lambda m: m.artifact_id.isin(model_ids[k]))
            value_dict[k] = {m.artifact_id: m for m in models}
        return value_dict
    
    
    async def log_value(self, value: Any, io_kind: ValueIOKind = "input"):
        """
        Log a value to the current span and add it to the values list
        """
        value = value.root if isinstance(value, SpanTree) else value
        value = await self.root.log_value(value, io_kind=io_kind)
        return value
    
    
    async def add_child(self, name: str, span_type: str = "component", tags: list[str] = []):
        """
        Add a child span to the current span and log the span as an output value
        """
        span_tree = await SpanTree(name, span_type, tags, index=len(self.children), parent=self).save()
        self.children.append(span_tree)  # Add to children list
        await self.log_value(span_tree, io_kind="output")
        await span_tree.save()
        return span_tree
    
    def print_tree(self):
        for s in self.traverse():
            print(s.id, s.path, s.name)
            for v in s.inputs:
                print("  ", v.id, v.io_kind, v.artifact_id)
            for v in s.outputs:
                print("  ", v.id, v.io_kind, v.artifact_id)