


from chatboard.model.neo4j.cypher.patterns import NodePattern
from chatboard.model.neo4j.cypher.queries import MatchQuery, MergeQuery


class CyBuilderBase:
    def __init__(self, query):
        self._query = query

    def match(self, var, label=None, props=None):
        node = NodePattern(var, [label] if label else [], props or {})
        self._query.match(node)
        return self

    def merge(self, pattern):
        self._query.merge(pattern)
        return self

    def where(self, *exprs):
        self._query.where(*exprs)
        return self

    def set(self, **kwargs):
        self._query.set(**kwargs)
        return self

    def return_(self, *items):
        self._query.return_(*items)
        return self

    def limit(self, n):
        self._query.limit(n)
        return self

    def set_param(self, **kwargs):
        self._query.set_param(**kwargs)
        return self

    def compile(self):
        return self._query.compile()

class CypherQuery:
    @classmethod
    def match(cls, var, label=None, props=None):
        return CyBuilderBase(MatchQuery()).match(var, label, props)

    @classmethod
    def merge(cls, pattern):
        return CyBuilderBase(MergeQuery()).merge(pattern)
