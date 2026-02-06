







from enum import Enum
import json
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict
import uuid

from pydantic import BaseModel
from chatboard.model.query_filters import FieldOp, QueryFilter, QueryOp
import datetime as dt

if TYPE_CHECKING:
    from chatboard.model.postgres.namespace import PostgresNamespace
    from chatboard.model.query_filters import QueryProxy


def is_enum(field_type):
    return isinstance(field_type, type) and issubclass(field_type, Enum)

def build_where_clause(query_filter: QueryFilter, alias: str | None = None) -> str:
    """Convert QueryFilter to SQL WHERE clause."""
    if isinstance(query_filter._operator, QueryOp):
        if query_filter._operator == QueryOp.AND:
            # return f"({build_where_clause(query_filter._left, alias)} AND {build_where_clause(query_filter._right, alias)})"
            return f"{build_where_clause(query_filter._left, alias)} AND {build_where_clause(query_filter._right, alias)}"
        elif query_filter._operator == QueryOp.OR:
            # return f"({build_where_clause(query_filter._left, alias)} OR {build_where_clause(query_filter._right, alias)})"
            return f"{build_where_clause(query_filter._left, alias)} OR {build_where_clause(query_filter._right, alias)}"
    elif isinstance(query_filter._operator, FieldOp):
        field = query_filter.field
        field_name = f'"{field.name}"'
        if alias:
            field_name = f"{alias}.{field_name}"
        
        # Get field type information
        if hasattr(field, '_field_info'):
            field_type = field._field_info.data_type
        else:
            field_type = field.type
        
        
        # Handle JSONB fields differently
        if str(field_type).startswith("dict") or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
            json_field = f"payload->'{field.name}'"
            if query_filter._operator == FieldOp.NULL:
                return f"{json_field} IS NULL"
            elif query_filter._operator == FieldOp.EQ:
                return f"{json_field} = '{json.dumps(query_filter.value)}'"
            elif query_filter._operator == FieldOp.NE:
                return f"{json_field} != '{json.dumps(query_filter.value)}'"
            elif query_filter._operator == FieldOp.IN:
                values = [json.dumps(v) for v in query_filter.value]
                return f"{json_field} = ANY(ARRAY[{', '.join(values)}])"
            elif query_filter._operator == FieldOp.NOTIN:
                values = [json.dumps(v) for v in query_filter.value]
                return f"{json_field} != ALL(ARRAY[{', '.join(values)}])"
        
        
        # Handle regular fields with proper type casting
        if query_filter._operator == FieldOp.NULL:
            return f"{field_name} IS NULL"
        elif query_filter._operator == FieldOp.EQ:
            if field_type is bool:
                return f"{field_name} = {str(query_filter.value).lower()}"
            elif field_type is int:
                return f"{field_name} = {query_filter.value}"
            elif field_type is str:
                return f"{field_name} = '{query_filter.value}'"
            elif field_type is uuid.UUID:
                return f"{field_name} = '{query_filter.value}'"
            elif field_type is dt.datetime:
                return f"{field_name} = '{query_filter.value}'"
            elif is_enum(field_type):
                return f"{field_name} = '{query_filter.value.value}'"
            else:
                raise ValueError(f"Unsupported field type: {field_type}")
        elif query_filter._operator == FieldOp.NE:
            if field_type is bool:
                return f"{field_name} != {str(query_filter.value).lower()}"
            elif field_type is int:
                return f"{field_name} != {query_filter.value}"
            elif field_type is str:
                return f"{field_name} != '{query_filter.value}'"
            elif field_type is uuid.UUID:
                return f"{field_name} != '{query_filter.value}'"
            elif field_type is dt.datetime:
                return f"{field_name} != '{query_filter.value}'"
            elif is_enum(field_type):
                return f"{field_name} != '{query_filter.value.value}'"
            else:
                raise ValueError(f"Unsupported field type: {field_type}")
        elif query_filter._operator == FieldOp.IN:
            if field_type is int:
                values = [str(v) for v in query_filter.value]
                return f"{field_name} = ANY(ARRAY[{', '.join(values)}])"
            else:
                values = [f"'{v}'" for v in query_filter.value]
                return f"{field_name} = ANY(ARRAY[{', '.join(values)}])"
        elif query_filter._operator == FieldOp.NOTIN:
            if field_type is int:
                values = [str(v) for v in query_filter.value]
                return f"{field_name} != ALL(ARRAY[{', '.join(values)}])"
            else:
                values = [f"'{v}'" for v in query_filter.value]
                return f"{field_name} != ALL(ARRAY[{', '.join(values)}])"
        elif query_filter._operator == FieldOp.RANGE:
            conditions = []
            if query_filter.value.gt is not None:
                if field_type is int or field_type is float:
                    conditions.append(f"{field_name} > {query_filter.value.gt}")
                else:
                    conditions.append(f"{field_name} > '{query_filter.value.gt}'")
            if query_filter.value.ge is not None:
                if field_type is int or field_type is float:
                    conditions.append(f"{field_name} >= {query_filter.value.ge}")#f"{field_name} >= to_timestamp('{query_filter.value.ge.strfmt('YYYY-MM-DD HH24:MI:SS')}')"
                # elif field_type == dt.datetime:                        
                    # conditions.append(f"{field_name} >= to_timestamp('{query_filter.value.ge.strftime('%Y-%m-%d %H:%M:%S')}')")
                else:
                    conditions.append(f"{field_name} >= '{query_filter.value.ge}'")
            if query_filter.value.lt is not None:
                if field_type is int or field_type is float:
                    conditions.append(f"{field_name} < {query_filter.value.lt}")
                else:
                    conditions.append(f"{field_name} < '{query_filter.value.lt}'")
            if query_filter.value.le is not None:
                if field_type is int or field_type is float:
                    conditions.append(f"{field_name} <= {query_filter.value.le}")
                else:
                    conditions.append(f"{field_name} <= '{query_filter.value.le}'")
            conditions_sql = ' AND '.join(conditions)
            return conditions_sql
            # if len(conditions) > 1:
            #     return f"({conditions_str})"
            # else:
            #     return conditions[0]
        else:
            raise ValueError(f"Unsupported query filter operator: {query_filter._operator}")
    return ""





class JoinType(TypedDict):
    """Join information"""
    primary_table: str
    primary_key: str
    foreign_table: str
    foreign_key: str
    
class SelectType(TypedDict):
    """Select information"""
    namespace: str
    fields: list[str]





def build_query(
        namespace: "PostgresNamespace", 
        select: list[SelectType] | None = None,
        filters: Dict[str, Any] | None = None, 
        limit: int | None = None,
        order_by: str | None = None,
        offset: int | None = None,
        joins: list[JoinType] | None = None,
        filter_proxy: "QueryProxy | None" = None,
        alias: str | None = None,        
        start_placeholder: int = 0,
    ) -> tuple[str, list[Any]]:
        """
        Query records with versioning support.
        
        Args:
            namespace: The namespace to query
            partition_id: The partition ID to query from (optional)
            branch_id: The branch ID to query from (optional)
            filters: Filters to apply to the query
            limit: Maximum number of records to return
            order_by: Field and direction to order by (e.g., "created_at desc")
            offset: Number of records to skip
            joins: List of joins to apply to the query
            
            
        Returns:
            A list of records matching the query
        """
        if not alias:
            alias = ""
            
        # Regular query without versioning
        select_clause = "*"
        
        if select:
            select_parts = []
            for select_type in select:                                
                select_parts.append(', '.join([f"{select_type['namespace']}.{f}" for f in select_type['fields']]))
            select_clause = ",".join(select_parts)
        

        sql = f'SELECT {select_clause}\n'
        sql += f'FROM "{namespace.table_name}" {alias}\n'
        
        if joins:
            for join in joins:
                # sql += f" JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}"
                sql += f"JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}\n"
        
        # Add filters
        values = []
        if filters or filter_proxy:
            filters_sql = "WHERE "
            if filters:
                where_parts = []
                for key, value in filters.items():
                    wq = f'"{key}" = ${start_placeholder + len(values) + 1}'
                    if alias:
                        wq = f"{alias}.{wq}"
                    where_parts.append(wq)
                    values.append(value)
                    
                filters_sql += ' AND '.join(where_parts)
            if filter_proxy: 
                filters_sql += " AND " if filters else ""           
                filters_sql += build_where_clause(filter_proxy, alias)
            sql += filters_sql
        
        # Add order by
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        # Add limit
        if limit:
            sql += f" LIMIT {limit}"
        
        # Add offset
        if offset:
            sql += f" OFFSET {offset}"
            
            
        return sql, values





