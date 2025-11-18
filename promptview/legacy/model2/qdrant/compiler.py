from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, Range, DatetimeRange, IsNullCondition
from promptview.model.postgres.sql.expressions import Eq, In, Between, IsNull, And, Or, Not, Expression
from typing import Any

class QdrantCompiler:
    """
    Compiles filter expressions from QdrantQuerySet into Qdrant Filter objects.
    """
    def compile_expr(self, expr: Expression) -> Filter:
        # Logical expressions
        if isinstance(expr, And):
            must = [self.compile_expr(cond) for cond in list(expr.conditions)]
            return Filter(must=must)
        elif isinstance(expr, Or):
            should = [self.compile_expr(cond) for cond in list(expr.conditions)]
            return Filter(should=should)
        elif isinstance(expr, Not):
            must_not = [self.compile_expr(expr.condition)]
            return Filter(must_not=must_not)
        # Leaf expressions
        elif isinstance(expr, Eq):
            field = expr.left.name if hasattr(expr.left, 'name') else expr.left
            value = expr.right.value if hasattr(expr.right, 'value') else expr.right
            cond = FieldCondition(key=field, match=MatchValue(value=value))
            return Filter(must=[cond])
        elif isinstance(expr, In):
            field = expr.value.name if hasattr(expr.value, 'name') else expr.value
            options = [v.value if hasattr(v, 'value') else v for v in expr.options]
            cond = FieldCondition(key=field, match=MatchAny(any=options))
            return Filter(must=[cond])
        elif isinstance(expr, Between):
            field = expr.value.name if hasattr(expr.value, 'name') else expr.value
            lower = expr.lower.value if hasattr(expr.lower, 'value') else expr.lower
            upper = expr.upper.value if hasattr(expr.upper, 'value') else expr.upper
            cond = FieldCondition(key=field, range=Range(gte=lower, lte=upper))
            return Filter(must=[cond])
        elif isinstance(expr, IsNull):
            field = expr.value.name if hasattr(expr.value, 'name') else expr.value
            cond = IsNullCondition(is_null=field)
            return Filter(must=[cond])
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}") 