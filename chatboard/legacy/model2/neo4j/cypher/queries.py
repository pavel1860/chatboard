

class MatchQuery:
    def __init__(self):
        self.patterns = []
        self.where_exprs = []
        self.return_items = []
        self.order_by = []
        self.limit_n = None
        self.params = {}
    def match(self, *patterns):
        self.patterns.extend(patterns)
        return self
    def where(self, *exprs):
        self.where_exprs.extend(exprs)
        return self
    def return_(self, *items):
        self.return_items.extend(items)
        return self
    def order(self, item, asc=True):
        self.order_by.append((item, asc))
        return self
    def limit(self, n):
        self.limit_n = n
        return self
    def set_param(self, **kwargs):
        self.params.update(kwargs)
        return self

    def compile(self):
        cypher = "MATCH " + ", ".join(str(p) for p in self.patterns)
        if self.where_exprs:
            cypher += " WHERE " + " AND ".join(str(e) for e in self.where_exprs)
        if self.return_items:
            cypher += " RETURN " + ", ".join(self.return_items)
        if self.order_by:
            cypher += " ORDER BY " + ", ".join(
                f"{i} {'ASC' if asc else 'DESC'}" for i, asc in self.order_by
            )
        if self.limit_n:
            cypher += f" LIMIT {self.limit_n}"
        return cypher, self.params






class MergeQuery:
    def __init__(self):
        self.patterns = []
        self.set_clauses = []
        self.return_items = []
        self.limit_n = None
        self.params = {}

    def merge(self, *patterns):
        self.patterns.extend(patterns)
        return self

    def set(self, **kwargs):
        # kwargs: property path (str) -> CypherExpr/CyParam/CyValue
        # e.g. set(n__name=CyParam("name")) => SET n.name = $name
        for prop, val in kwargs.items():
            # Support dotted or double-underscore path: n__name or n.name
            prop_path = prop.replace("__", ".")
            self.set_clauses.append((prop_path, val))
        return self

    def return_(self, *items):
        self.return_items.extend(items)
        return self

    def limit(self, n):
        self.limit_n = n
        return self

    def set_param(self, **kwargs):
        self.params.update(kwargs)
        return self

    def compile(self):
        cypher = "MERGE " + ", ".join(str(p) for p in self.patterns)
        if self.set_clauses:
            set_str = ", ".join(f"{k} = {v}" for k, v in self.set_clauses)
            cypher += f" SET {set_str}"
        if self.return_items:
            cypher += " RETURN " + ", ".join(self.return_items)
        if self.limit_n:
            cypher += f" LIMIT {self.limit_n}"
        return cypher, self.params







