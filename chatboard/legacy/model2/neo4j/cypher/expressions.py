


class CypherExpr:
    def __and__(self, other):
        return CyAnd(self, other)
    def __or__(self, other):
        return CyOr(self, other)
    def __invert__(self):
        return CyNot(self)

class CyValue(CypherExpr):
    def __init__(self, value): self.value = value
    
    def __str__(self):
        # string quoting if needed
        if isinstance(self.value, str):
            return repr(self.value)
        return str(self.value)

class CyParam(CypherExpr):
    def __init__(self, name): self.name = name
    
    def __str__(self):
        return f"${self.name}"
    
    def __repr__(self):
        return f"CyParam({self.name})"

class CyProperty(CypherExpr):
    def __init__(self, var, prop): self.var = var; self.prop = prop
    def __str__(self): return f"{self.var}.{self.prop}"
    def __eq__(self, other): return CyEq(self, other)
    def __ne__(self, other): return CyNeq(self, other)
    def __gt__(self, other): return CyGt(self, other)
    def __lt__(self, other): return CyLt(self, other)
    def __ge__(self, other): return CyGte(self, other)
    def __le__(self, other): return CyLte(self, other)

class CyEq(CypherExpr):
    def __init__(self, left, right): self.left = left; self.right = right
    def __str__(self): return f"{self.left} = {self.right}"

# class CyAnd(CypherExpr):
#     def __init__(self, *conds): self.conds = conds
#     def __str__(self): return " AND ".join(str(c) for c in self.conds)

class CyAnd:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"({self.left} AND {self.right})"

class CyOr:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"({self.left} OR {self.right})"

class CyNot:
    def __init__(self, expr):
        self.expr = expr
    def __str__(self):
        return f"(NOT {self.expr})"


class CyNeq(CypherExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"{self.left} <> {self.right}"

class CyGt(CypherExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"{self.left} > {self.right}"

class CyLt(CypherExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"{self.left} < {self.right}"

class CyGte(CypherExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"{self.left} >= {self.right}"

class CyLte(CypherExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return f"{self.left} <= {self.right}"