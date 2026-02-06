from typing import TYPE_CHECKING, Any, Callable, Generic, List, Type, TypeVar

from chatboard.model.base_namespace import Namespace
from chatboard.model.neo4j.cypher.queries import MatchQuery
from chatboard.model.neo4j.cypher.patterns import NodePattern, PatternChain, RelPattern
from chatboard.model.neo4j.cypher.expressions import CyProperty, CyEq, CyParam
# from chatboard.model2.neo4j.cypher.compiler import CypherCompiler  # You'll need to implement a compiler
from chatboard.model.neo4j.connection import Neo4jConnectionManager  # Your connection manager

if TYPE_CHECKING:
    from chatboard.model.model import Model


MODEL = TypeVar("MODEL", bound="Model")

class QueryProxy:
    """
    Used to allow .where(lambda n: n.id == 42) to build a Cypher expression tree.
    """
    def __init__(self, varname: str = "n"):
        self.varname = varname
    def __getattr__(self, field):
        return CyProperty(self.varname, field)

class Neo4jQuerySet(Generic[MODEL]):
    def __init__(
        self,
        model_class: Type[MODEL],
        query: MatchQuery = None,
        varname: str = "n",
        pattern_chain: list = None,
        return_vars: list = None,
    ):
        self.model_class = model_class
        self.varname = varname
        self.pattern_chain = pattern_chain or [NodePattern(varname, [self.model_class.get_namespace().name])]
        self.query = query or MatchQuery()
        self.return_vars = return_vars or [varname]
        self._params = {}

        # Ensure pattern chain is in the match clause
        if not self.query.patterns:
            self.query.match(PatternChain(*self.pattern_chain))

    def rel(self, rel_type: str, direction: str = "right", varname: str = "m", target_label: str = None, target_props: dict = None):
        """
        Traverses a relationship from current node, returns new QuerySet focused on the target node.
        rel_type: The relationship type/edge label.
        direction: "right" for outgoing, "left" for incoming, "none" for undirected.
        varname: Variable for the target node.
        target_label: Optional, label for target node (e.g. "User" or "Group")
        target_props: Optional, dict of property expressions for the target node.
        """
        # Build a new pattern chain
        new_pattern = list(self.pattern_chain)
        new_pattern.append(RelPattern("", rel_type, direction=direction))
        new_pattern.append(NodePattern(varname, [target_label] if target_label else [], target_props or {}))
        # New QuerySet with updated varname and pattern_chain
        return Neo4jQuerySet(
            model_class=self.model_class,  # could allow .rel to change type
            query=self.query,
            varname=varname,
            pattern_chain=new_pattern,
            return_vars=self.return_vars + [varname]
        )

    def where(self, condition: Callable[[Any], Any] = None, **kwargs) -> "Neo4jQuerySet[MODEL]":
        if condition:
            proxy = QueryProxy(self.varname)
            expr = condition(proxy)
            self.query.where(expr)
        for k, v in kwargs.items():
            self.query.where(CyEq(CyProperty(self.varname, k), CyParam(k)))
            self._params[k] = v
        return self


    def filter(self, condition: Callable[[Any], Any] = None, **kwargs) -> "Neo4jQuerySet[MODEL]":
        return self.where(condition, **kwargs)

    def order_by(self, *fields: str) -> "Neo4jQuerySet[MODEL]":
        for field in fields:
            direction = "ASC"
            if field.startswith("-"):
                direction = "DESC"
                field = field[1:]
            # Use Cypher's ORDER BY clause syntax directly
            self.query.order(field, direction == "ASC")
        return self

    def limit(self, n: int) -> "Neo4jQuerySet[MODEL]":
        self.query.limit(n)
        return self

    def return_(self, *fields: str) -> "Neo4jQuerySet[MODEL]":
        if not fields:
            fields = (self.varname,)
        self.query.return_(*fields)
        return self

    def set_param(self, **kwargs):
        self._params.update(kwargs)
        return self

    def render(self):
        self.query.patterns = [PatternChain(*self.pattern_chain)]
        cypher, params = self.query.compile()
        # Merge in any user-supplied params
        params.update(self._params)
        self._params = params
        return cypher, params

    async def execute(self) -> List[MODEL]:
        cypher, params = self.render()
        records = await Neo4jConnectionManager.execute_read(cypher, params)
        # For each record, hydrate model from node properties:
        results = []
        for row in records:
            node_props = row.get(self.varname)
            if node_props is None and row:
                # If using RETURN n, but driver returns {'n': NodeObject}
                node_props = next(iter(row.values()))
            data = {}
            for fname, finfo in self.model_class.get_namespace()._fields.items():
                val = node_props.get(fname) if node_props else None
                data[fname] = finfo.deserialize(val)
            results.append(self.model_class.from_dict(data))
        return results

    def __await__(self):
        return self.execute().__await__()

    # Syntactic sugar for "first" result (like QuerySetSingleAdapter)
    async def first(self):
        self.limit(1)
        results = await self.execute()
        return results[0] if results else None

    async def last(self):
        # If you want to implement last (order by pk descending, limit 1)
        pk = self.model_class.get_key_field()
        self.order_by(f"-{pk}")
        self.limit(1)
        results = await self.execute()
        return results[0] if results else None