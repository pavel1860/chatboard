from codecs import lookup
from typing import Any
from ..model import Turn, Branch, ExecutionSpan, SpanValue, Artifact, ValueIOKind

from collections import defaultdict

# spans = await ExecutionSpan.query().include(Artifact.query().where(turn_id = 3)).print()
# spans = await ExecutionSpan.query()
# spans[0].artifact

class SpanTree:
        
    # def __init__(self, span: ExecutionSpan, index: int = 0, children: list[ExecutionSpan] | None = None, parent: "SpanTree | None" = None):
    def __init__(
        self, 
        target: str | ExecutionSpan, 
        span_type: str = "component", 
        tags: list[str] = [], 
        index: int = 0, 
        children: list[ExecutionSpan] | None = None, 
        parent: "SpanTree | None" = None
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

        
    async def save(self):
        """
        Save the span tree to the database
        """
        await self.root.save()
        return self
    
    @classmethod
    async def init_new(cls, name: str, span_type: str = "component", tags: list[str] = [], index: int = 0):
        span = await ExecutionSpan(name=name, span_type=span_type, tags=tags).save()
        return cls(span)

    @property
    def id(self):
        return self.root.id
    
    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return []
        if self.index is None:
            return self.parent.path
        return self.parent.path + [self.index]
    
    @property
    def name(self):
        return self.root.name
    
    @property
    def values(self):
        return self.root.values
    
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
        
    @classmethod
    def load_span_list(cls, span_list: list[ExecutionSpan]):
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
        def populate_children(span: SpanTree, lookup: dict[int, list[ExecutionSpan]]):
            children = lookup.get(span.id)
            if children is None:
                return span
            span.children = [SpanTree(c, index=i, parent=span) for i, c in enumerate(children)]
            for child in span.children:
                populate_children(child, lookup)
            return span
        return populate_children(SpanTree(root), lookup)
    
        
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
        return cls.load_span_list(spans)
    
    async def log_value(self, value: Any, io_kind: ValueIOKind = "input"):
        """
        Log a value to the current span and add it to the values list
        """
        value = await self.root.log_value(value, io_kind=io_kind)
        self.values.append(value)
        return value
    
    
    async def add_child(self, name: str, span_type: str = "component", tags: list[str] = []):
        """
        Add a child span to the current span and log the span as an output value
        """
        span_tree = SpanTree(name, span_type, tags, index=len(self.children), parent=self)
        await self.log_value(span_tree, io_kind="output")
        await span_tree.save()        
        return span_tree
    
    def print_tree(self):
        for s in self.traverse():
            print(s.id, s.name)
            for v in s.inputs:
                print("  ", v.id, v.io_kind, v.artifact_id)
            for v in s.outputs:
                print("  ", v.id, v.io_kind, v.artifact_id)