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
