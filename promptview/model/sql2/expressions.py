"""
Expression system for SQL2 relational queries.

This module provides a type-safe way to build SQL expressions using RelField objects
from the new Source-based relation architecture.
"""

from typing import Any


class Expression:
    """Base class for all SQL expressions"""

    def __and__(self, other: "Expression") -> "And":
        """Combine expressions with AND"""
        return And(self, other)

    def __or__(self, other: "Expression") -> "Or":
        """Combine expressions with OR"""
        return Or(self, other)

    def __invert__(self) -> "Not":
        """Negate expression with NOT"""
        return Not(self)


class Value(Expression):
    """Represents a literal value or parameter"""

    def __init__(self, value: Any, inline: bool = False):
        """
        Args:
            value: The Python value
            inline: If True, inline in SQL; if False, use parameter ($1, $2, etc.)
        """
        self.value = value
        self.inline = inline


def param(value: Any) -> Value:
    """Create a parameterized value (safe from SQL injection)"""
    return Value(value, inline=False)


class Null(Expression):
    """Represents SQL NULL"""
    pass


class BinaryExpression(Expression):
    """Base class for binary operations (left operator right)"""

    def __init__(self, left, operator: str, right):
        self.left = left
        self.operator = operator
        # Auto-wrap raw Python values as parameters
        if not isinstance(right, Expression):
            # Check if it's a RelField (avoid circular import)
            if not hasattr(right, 'name') or not hasattr(right, 'source'):
                self.right = param(right)
            else:
                self.right = right
        else:
            self.right = right


class Eq(BinaryExpression):
    """Equality: left = right"""
    def __init__(self, left, right):
        super().__init__(left, '=', right)


class Neq(BinaryExpression):
    """Inequality: left != right"""
    def __init__(self, left, right):
        super().__init__(left, '!=', right)


class Gt(BinaryExpression):
    """Greater than: left > right"""
    def __init__(self, left, right):
        super().__init__(left, '>', right)


class Gte(BinaryExpression):
    """Greater than or equal: left >= right"""
    def __init__(self, left, right):
        super().__init__(left, '>=', right)


class Lt(BinaryExpression):
    """Less than: left < right"""
    def __init__(self, left, right):
        super().__init__(left, '<', right)


class Lte(BinaryExpression):
    """Less than or equal: left <= right"""
    def __init__(self, left, right):
        super().__init__(left, '<=', right)


class And(Expression):
    """Logical AND of multiple conditions"""

    def __init__(self, *conditions: Expression):
        self.conditions = conditions


class Or(Expression):
    """Logical OR of multiple conditions"""

    def __init__(self, *conditions: Expression):
        self.conditions = conditions


class Not(Expression):
    """Logical NOT"""

    def __init__(self, condition: Expression):
        self.condition = condition


class IsNull(Expression):
    """IS NULL check"""

    def __init__(self, value):
        self.value = value


class IsNotNull(Expression):
    """IS NOT NULL check"""

    def __init__(self, value):
        self.value = value


class In(Expression):
    """IN operator: value IN (option1, option2, ...)"""

    def __init__(self, value, options: list):
        self.value = value
        self.options = options


class NotIn(Expression):
    """NOT IN operator"""

    def __init__(self, value, options: list):
        self.value = value
        self.options = options


class Between(Expression):
    """BETWEEN operator: value BETWEEN lower AND upper"""

    def __init__(self, value, lower: Any, upper: Any):
        self.value = value
        self.lower = lower if isinstance(lower, Expression) else param(lower)
        self.upper = upper if isinstance(upper, Expression) else param(upper)


class Like(Expression):
    """LIKE operator for pattern matching"""

    def __init__(self, value, pattern: str):
        self.value = value
        self.pattern = param(pattern)


class ILike(Expression):
    """ILIKE operator (case-insensitive LIKE, Postgres-specific)"""

    def __init__(self, value, pattern: str):
        self.value = value
        self.pattern = param(pattern)


class WhereClause:
    """
    Container for WHERE conditions with fluent API.

    Usage:
        where = WhereClause()
        where &= posts.get("published") == True
        where &= posts.get("views") > 100
    """

    def __init__(self, condition: Expression | None = None):
        self.condition = condition

    def __bool__(self) -> bool:
        """Check if there's a condition"""
        return self.condition is not None

    def __and__(self, other: Expression) -> "WhereClause":
        """Add condition with AND"""
        if self.condition is None:
            self.condition = other
        else:
            self.condition = And(self.condition, other)
        return self

    def __or__(self, other: Expression) -> "WhereClause":
        """Add condition with OR"""
        if self.condition is None:
            self.condition = other
        else:
            self.condition = Or(self.condition, other)
        return self

    def __repr__(self):
        return f"WhereClause({self.condition})"


class JsonBuildObject(Expression):
    """
    jsonb_build_object() function - constructs a JSON object from key-value pairs.

    Takes a dictionary mapping JSON keys to expressions (fields, subqueries, other expressions).

    Example:
        JsonBuildObject({
            "id": posts_rel.get("id"),
            "title": posts_rel.get("title"),
            "comment_count": comment_count_subquery
        })
    """

    def __init__(self, field_map: dict[str, any]):
        """
        Args:
            field_map: Dictionary mapping JSON keys (strings) to expressions/fields
                      e.g., {"id": posts.get("id"), "count": Count()}
        """
        self.field_map = field_map


class AggregateFunction(Expression):
    """Base class for aggregate functions"""
    pass


class JsonAgg(AggregateFunction):
    """
    json_agg() aggregate function - collects values into JSON array.

    Can wrap any expression (fields, JsonBuildObject, subqueries, etc.).

    Example:
        JsonAgg(JsonBuildObject({
            "id": comments.get("id"),
            "text": comments.get("text")
        }))
    """

    def __init__(self, expr):
        """
        Args:
            expr: Expression to aggregate into JSON array
        """
        self.expr = expr


class JsonbAgg(AggregateFunction):
    """
    jsonb_agg() aggregate function - collects values into JSONB array.

    Similar to JsonAgg but returns jsonb type instead of json type.
    Use this when you need type consistency with jsonb_build_object.

    Example:
        JsonbAgg(JsonBuildObject({
            "id": comments.get("id"),
            "text": comments.get("text")
        }))
    """

    def __init__(self, expr):
        """
        Args:
            expr: Expression to aggregate into JSONB array
        """
        self.expr = expr


class Count(AggregateFunction):
    """COUNT() aggregate function"""

    def __init__(self, expr=None, distinct: bool = False):
        """
        Args:
            expr: Expression to count (None for COUNT(*))
            distinct: If True, count only distinct values
        """
        self.expr = expr
        self.distinct = distinct


class Sum(AggregateFunction):
    """SUM() aggregate function"""

    def __init__(self, expr):
        self.expr = expr


class Avg(AggregateFunction):
    """AVG() aggregate function"""

    def __init__(self, expr):
        self.expr = expr


class Min(AggregateFunction):
    """MIN() aggregate function"""

    def __init__(self, expr):
        self.expr = expr


class Max(AggregateFunction):
    """MAX() aggregate function"""

    def __init__(self, expr):
        self.expr = expr


class Coalesce(Expression):
    """
    COALESCE() function - returns the first non-NULL value.

    Example:
        Coalesce(field, Value(0))  # Return field or 0 if NULL
        Coalesce(subquery, Value("[]"))  # Return subquery result or empty array
    """

    def __init__(self, *values):
        """
        Args:
            *values: Variable number of expressions to check for NULL
        """
        if len(values) < 2:
            raise ValueError("COALESCE requires at least 2 arguments")
        self.values = values
