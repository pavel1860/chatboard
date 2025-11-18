import datetime as dt
from typing import Any, List, Type

from promptview.model3.sql.queries import Column, Table
from promptview.model3.sql.expressions import Eq, Gte, Lte, Gt, Lt, And, Neq, param





def parse_query_params(model_class, conditions: list[list[Any]], table: Table | None = None):
    """
    Parse a list of query conditions into a combined SQL expression.
    """
    namespace = model_class.get_namespace()
    
    if table is None:
        table = Table(namespace.table_name)

    exprs = []
    for condition in conditions:
        if len(condition) != 3:
            raise ValueError(f"Invalid condition: {condition}")
        field, operator, value = condition

        # -- This is the critical line --
        field_info = namespace.get_field(field)
        if field_info is None:
            raise ValueError(f"Field {field} not found in namespace {namespace.name}")
        column = Column(field_info.name, table)  # Always a Column object!

        # --- Type conversion ---
        # You can expand this if needed
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
        # else: leave str
        value = param(value)
        # --- Build expression ---
        if operator == "==":
            exprs.append(Eq(column, value))
        elif operator == ">=":
            exprs.append(Gte(column, value))
        elif operator == "<=":
            exprs.append(Lte(column, value))
        elif operator == ">":
            exprs.append(Gt(column, value))
        elif operator == "<":
            exprs.append(Lt(column, value))
        elif operator == "!=":
            exprs.append(Neq(column, value))
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    # Combine with AND
    if not exprs:
        return None
    elif len(exprs) == 1:
        return exprs[0]
    else:
        return And(*exprs)
