import datetime as dt
from typing import Any, List, Literal, Type, Union

from .sql2.relations import NsRelation
from .sql2.expressions import Eq, Gte, Lte, Gt, Lt, And, Neq, Value



# Define the comparison operators
ComparisonOperator = Literal["==", "!=", ">", "<", ">=", "<="]

# Define the logical operators that can connect conditions
LogicalOperator = Literal["and", "or"]

# Define a single condition as a 3-element list: [field_name, operator, value]
# Using List instead of Tuple for easier compatibility with the examples
Condition = List[Any]  # [str, ComparisonOperator, Any]

# Define a query element which can be either a condition or a logical operator
QueryElement = Union[Condition, LogicalOperator]

# Define a simple query with just one condition
SimpleQuery = List[Condition]

# Define a complex query with multiple conditions connected by logical operators
# This is a list of elements that alternates between conditions and logical operators
ComplexQuery = List[QueryElement]

# The general query type that can be either simple or complex
QueryListType = Union[SimpleQuery, ComplexQuery]



def parse_query_params(model_class, conditions: list[list[Any]], relation: NsRelation | None = None):
    """
    Parse a list of query conditions into a combined SQL expression.

    Args:
        model_class: The model class to query
        conditions: List of [field, operator, value] conditions
        relation: Optional NsRelation to use (created from namespace if not provided)

    Returns:
        Combined SQL expression using SQL2 system
    """
    namespace = model_class.get_namespace()

    if relation is None:
        relation = NsRelation(namespace)

    exprs = []
    for condition in conditions:
        if len(condition) != 3:
            raise ValueError(f"Invalid condition: {condition}")
        field_name, operator, value = condition

        # Get field info for validation and type conversion
        field_info = namespace.get_field(field_name)
        if field_info is None:
            raise ValueError(f"Field {field_name} not found in namespace {namespace.name}")

        # Get RelField from relation (SQL2 field reference)
        field = relation.get(field_name)

        # --- Type conversion ---
        if field_info.is_temporal and isinstance(value, str):
            try:
                value = dt.datetime.fromisoformat(value)
            except Exception:
                raise ValueError(f"Invalid datetime format: {value}")
        elif field_info.is_enum:
            value = value  # Enum conversion here if needed
        elif field_info.data_type is float:
            value = float(value)
        elif field_info.data_type is int:
            value = int(value)
        # else: leave as str

        # Wrap value as parameter (SQL2 uses Value instead of param)
        value = Value(value, inline=False)

        # --- Build expression ---
        if operator == "==":
            exprs.append(Eq(field, value))
        elif operator == ">=":
            exprs.append(Gte(field, value))
        elif operator == "<=":
            exprs.append(Lte(field, value))
        elif operator == ">":
            exprs.append(Gt(field, value))
        elif operator == "<":
            exprs.append(Lt(field, value))
        elif operator == "!=":
            exprs.append(Neq(field, value))
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    # Combine with AND
    if not exprs:
        return None
    elif len(exprs) == 1:
        return exprs[0]
    else:
        return And(*exprs)
