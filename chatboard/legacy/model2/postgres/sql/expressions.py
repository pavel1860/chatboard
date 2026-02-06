


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from chatboard.model.postgres.sql.queries import Column


class Expression:
    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    
class RawValue(Expression):
    def __init__(self, value):
        self.value = value


class Value(Expression):
    def __init__(self, value, inline=True, no_quote=False):
        self.value = value
        self.inline = inline
        self.no_quote = no_quote
        
        
class Null(Value):
    def __init__(self):
        self.value = "NULL"
        self.inline = True
        self.no_quote = True
        
        
        
        
def param(value):
    return Value(value, inline=False)

class Function(Expression):
    def __init__(self, name, *args, alias=None, filter_where=None, distinct=False, order_by=None):
        self.name = name
        self.args = args
        self.alias = alias
        self.filter_where = filter_where
        self.distinct = distinct
        self.order_by = order_by

    def __str__(self):
        inner = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({inner})" + (f" AS {self.alias}" if self.alias else "")



class Coalesce(Expression):
    def __init__(self, *values, alias=None):
        self.values = values
        self.alias = alias

def json_build_object(**kwargs):
    args = []
    for key, value in kwargs.items():
        args.append(Value(key))
        args.append(value)
    return Function("jsonb_build_object", *args)




class BinaryExpression(Expression):
    def __init__(self, left, operator: str, right):
        self.left = left
        self.operator = operator
        self.right = right

class Eq(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '=', right)

class Neq(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '!=', right)

class Gt(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '>', right)

class Gte(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '>=', right)

class Lt(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '<', right)

class Lte(BinaryExpression):
    def __init__(self, left, right):
        super().__init__(left, '<=', right)




class And(Expression):
    def __init__(self, *conditions):
        self.conditions = conditions

class Or(Expression):
    def __init__(self, *conditions):
        self.conditions = conditions

class Not(Expression):
    def __init__(self, condition):
        self.condition = condition


class OrderBy(Expression):
    def __init__(self, column: "Column", direction: str = "ASC"):
        self.column = column
        self.direction = direction.upper()



class IsNull(Expression):
    def __init__(self, value):
        self.value = value

class In(Expression):
    def __init__(self, value, options):
        self.value = value
        self.options = options  # Can be a list or subquery

class Between(Expression):
    def __init__(self, value, lower, upper):
        self.value = value
        self.lower = lower
        self.upper = upper

class Like(Expression):
    def __init__(self, value, pattern):
        self.value = value
        self.pattern = pattern



class WhereClause:
    def __init__(self, condition: Expression | None = None):
        self.condition = condition or None
        
        
    def __call__(self, condition: Expression):
        # if self.condition is None:
        self.condition = condition
        # else:
        #     self.condition = And(self.condition, condition)
        # return self
    def __bool__(self):
        return self.condition is not None
    
    def and_(self, condition: Expression):
        if self.condition is None:
            self.condition = condition
        else:
            self.condition = And(self.condition, condition)
        return self
    def or_(self, condition: Expression):
        if self.condition is None:
            self.condition = condition
        else:
            self.condition = Or(self.condition, condition)
        return self
    
    def invert(self):
        if self.condition is None:
            return Not(self)
        else:
            self.condition = Not(self.condition)
        return self
    
    def isin(self, values: list):
        return In(self.condition, values)
        
    def __and__(self, other: Expression):
        return self.and_(other)
        
    def __or__(self, other: Expression):
        return self.or_(other)
        
    def __invert__(self):
        return self.invert()
        
    def __str__(self):
        return str(self.condition) if self.condition else ""


class RawSQL:
    def __init__(self, sql: str, params: list | None = None):
        self.sql = sql
        self.params = params or []




class VectorDistance(BinaryExpression):
    def __init__(self, left, operator: str, right):
        super().__init__(left, operator, right)

def l2_distance(left, right):
    return VectorDistance(left, "<->", param(right))

def cosine_distance(left, right):
    return VectorDistance(left, "<=>", param(right))

def inner_product(left, right):
    return VectorDistance(left, "<#>", param(right))
