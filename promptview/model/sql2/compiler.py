from .relational_queries import QuerySet, SelectQuerySet, Relation
from .relations import Source, RelField
from .expressions import (
    Expression, BinaryExpression, And, Or, Not, IsNull, IsNotNull,
    In, NotIn, Between, Like, ILike, Value, JsonBuildObject, JsonAgg,
    Count, Sum, Avg, Min, Max, AggregateFunction
)
import textwrap







class Compiler:
    def __init__(self):
        self.params = []
        self.param_counter = 1
        self.indent_level = 0  # Track current indentation level for nested expressions

    def compile(self, query: QuerySet):
    
        cte_sql = self.compile_ctes(query)
        sql = ""        
        if isinstance(query, SelectQuerySet):
            sql = self.compile_select_query(query)
        else:
            raise ValueError(f"Unknown query type: {type(query)}")
            
        sql = cte_sql + sql
        return sql, self.params

    def compile_ctes(self, query: QuerySet):
        if not query.ctes:
            return ""
        ctes_sql = []
        for cte in query.ctes:
            cte_sql = self.compile(cte)
            cte_sql = "\n" + textwrap.indent(cte_sql, "    ")
            cte_name = cte.final_name + "_cte"
            ctes_sql.append(f"{cte_name} AS ({cte_sql})")
        return "WITH " + ", ".join(ctes_sql) + "\n"
            
            
            
    def compile_expr(self, expr: Expression) -> str:
        """Compile an expression to SQL"""
        if isinstance(expr, RelField):
            # Field reference
            return f"{expr.source.final_name}.{expr.name}"

        elif isinstance(expr, Value):
            # Literal value or parameter
            if expr.inline:
                # Inline value (use carefully!)
                return str(expr.value)
            else:
                # Parameterized value (safe from SQL injection)
                self.params.append(expr.value)
                param_num = self.param_counter
                self.param_counter += 1
                return f"${param_num}"

        elif isinstance(expr, BinaryExpression):
            # Binary operations: left operator right
            left_sql = self.compile_expr(expr.left)
            right_sql = self.compile_expr(expr.right)
            return f"{left_sql} {expr.operator} {right_sql}"

        elif isinstance(expr, And):
            # Logical AND
            conditions = [self.compile_expr(cond) for cond in expr.conditions]
            return "(" + " AND ".join(conditions) + ")"

        elif isinstance(expr, Or):
            # Logical OR
            conditions = [self.compile_expr(cond) for cond in expr.conditions]
            return "(" + " OR ".join(conditions) + ")"

        elif isinstance(expr, Not):
            # Logical NOT
            condition_sql = self.compile_expr(expr.condition)
            return f"NOT ({condition_sql})"

        elif isinstance(expr, IsNull):
            # IS NULL check
            value_sql = self.compile_expr(expr.value)
            return f"{value_sql} IS NULL"

        elif isinstance(expr, IsNotNull):
            # IS NOT NULL check
            value_sql = self.compile_expr(expr.value)
            return f"{value_sql} IS NOT NULL"

        elif isinstance(expr, In):
            # IN operator
            value_sql = self.compile_expr(expr.value)
            options_sql = ", ".join([self.compile_expr(opt) if isinstance(opt, Expression) else self.compile_expr(Value(opt, inline=False)) for opt in expr.options])
            return f"{value_sql} IN ({options_sql})"

        elif isinstance(expr, NotIn):
            # NOT IN operator
            value_sql = self.compile_expr(expr.value)
            options_sql = ", ".join([self.compile_expr(opt) if isinstance(opt, Expression) else self.compile_expr(Value(opt, inline=False)) for opt in expr.options])
            return f"{value_sql} NOT IN ({options_sql})"

        elif isinstance(expr, Between):
            # BETWEEN operator
            value_sql = self.compile_expr(expr.value)
            lower_sql = self.compile_expr(expr.lower)
            upper_sql = self.compile_expr(expr.upper)
            return f"{value_sql} BETWEEN {lower_sql} AND {upper_sql}"

        elif isinstance(expr, Like):
            # LIKE operator
            value_sql = self.compile_expr(expr.value)
            pattern_sql = self.compile_expr(expr.pattern)
            return f"{value_sql} LIKE {pattern_sql}"

        elif isinstance(expr, ILike):
            # ILIKE operator (Postgres-specific)
            value_sql = self.compile_expr(expr.value)
            pattern_sql = self.compile_expr(expr.pattern)
            return f"{value_sql} ILIKE {pattern_sql}"

        elif isinstance(expr, JsonBuildObject):
            # jsonb_build_object('key1', value1, 'key2', value2, ...)
            # Format with newlines for readability
            pairs = []
            has_nested_json = False

            # Increase indent level for nested content
            self.indent_level += 1
            base_indent = "    " * self.indent_level

            for key, value_expr in expr.field_map.items():
                # Key is always a string literal
                key_sql = f"'{key}'"
                # Value can be any expression
                value_sql = self.compile_expr(value_expr)

                # Check if this value is a nested JSON object
                if isinstance(value_expr, (JsonBuildObject, JsonAgg)):
                    has_nested_json = True

                pairs.append(f"{key_sql}, {value_sql}")

            # Decrease indent level after processing
            self.indent_level -= 1
            outer_indent = "    " * self.indent_level

            # Use multi-line format if:
            # - More than 2 keys, OR
            # - Contains nested JSON objects, OR
            # - We're already nested (indent_level > 1)
            if len(pairs) > 2 or has_nested_json or self.indent_level > 0:
                # Multi-line format with proper indentation
                pairs_formatted = f",\n{base_indent}".join(pairs)
                return f"jsonb_build_object(\n{base_indent}{pairs_formatted}\n{outer_indent})"
            else:
                # Single line for simple cases (only at top level with <= 2 keys)
                pairs_flat = ", ".join(pairs)
                return f"jsonb_build_object({pairs_flat})"

        elif isinstance(expr, JsonAgg):
            # json_agg(expression)
            inner_sql = self.compile_expr(expr.expr)
            return f"json_agg({inner_sql})"

        elif isinstance(expr, Count):
            # COUNT(*) or COUNT(DISTINCT expr)
            if expr.expr is None:
                return "COUNT(*)"
            else:
                distinct = "DISTINCT " if expr.distinct else ""
                expr_sql = self.compile_expr(expr.expr)
                return f"COUNT({distinct}{expr_sql})"

        elif isinstance(expr, Sum):
            # SUM(expression)
            expr_sql = self.compile_expr(expr.expr)
            return f"SUM({expr_sql})"

        elif isinstance(expr, Avg):
            # AVG(expression)
            expr_sql = self.compile_expr(expr.expr)
            return f"AVG({expr_sql})"

        elif isinstance(expr, Min):
            # MIN(expression)
            expr_sql = self.compile_expr(expr.expr)
            return f"MIN({expr_sql})"

        elif isinstance(expr, Max):
            # MAX(expression)
            expr_sql = self.compile_expr(expr.expr)
            return f"MAX({expr_sql})"

        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def compile_select_query(self, query: SelectQuerySet):
        """Compile a SELECT query with FROM and JOIN clauses"""

        sql = "SELECT"

        # Add DISTINCT or DISTINCT ON
        if query.distinct_on_fields:
            # DISTINCT ON (Postgres-specific)
            distinct_fields = ", ".join(query.distinct_on_fields)
            sql += f" DISTINCT ON ({distinct_fields})"
        elif query.distinct_enabled:
            # Simple DISTINCT
            sql += " DISTINCT"

        sql += "\n"

        # Compile projection fields
        for field in query.iter_projection_fields():
            if field.is_query:
                # Subquery field (from include)
                sub_sql = self.compile(field.source)
                sub_sql = textwrap.indent(sub_sql, "  ")
                sub_sql = f"(\n{sub_sql}\n) AS {field.name},\n"
                sub_sql = textwrap.indent(sub_sql, "    ")
                sql += sub_sql
            elif isinstance(field.source, Expression):
                # Expression field (from select_expr)
                expr_sql = self.compile_expr(field.source)
                sql += f"    {expr_sql} AS {field.name},\n"
            else:
                # Regular field
                sql += f"    {field.source.final_name}.{field.name},\n"

        # Remove trailing comma
        sql = sql.rstrip(",\n") + "\n"

        # Compile FROM and JOIN clauses
        if not query.sources:
            raise ValueError("Query has no sources")

        # First source is the FROM clause (should have no join_on)
        first_source = query.sources[0]

        # Check if first source is a subquery
        if isinstance(first_source.base, QuerySet):
            # Compile subquery
            sub_sql = self.compile(first_source.base)
            sub_sql = textwrap.indent(sub_sql, "    ")
            sql += f"FROM (\n{sub_sql}\n) AS {first_source.final_name}\n"
        else:
            sql += f"FROM {first_source.final_name}\n"

        # Rest are JOINs
        for source in query.sources[1:]:
            if not isinstance(source, Source):
                raise ValueError(f"Expected Source, got {type(source)}")

            if source.join_on is None:
                raise ValueError(f"Source {source.name} is missing join information")

            # Check if source is a subquery
            if isinstance(source.base, QuerySet):
                # Compile subquery for JOIN
                sub_sql = self.compile(source.base)
                sub_sql = textwrap.indent(sub_sql, "    ")
                sql += f"{source.join_type} JOIN (\n{sub_sql}\n) AS {source.final_name} ON {source.get_on_clause()}\n"
            else:
                # Regular table JOIN
                sql += f"{source.join_type} JOIN {source.final_name} ON {source.get_on_clause()}\n"

        # Add WHERE clause if present
        if query.where_clause:
            where_sql = self.compile_expr(query.where_clause.condition)
            sql += f"WHERE {where_sql}\n"

        # Add GROUP BY clause if present
        if query.group_by_fields:
            group_by_sql = ", ".join(query.group_by_fields)
            sql += f"GROUP BY {group_by_sql}\n"

        # Add ORDER BY clause if present
        if query.order_by_fields:
            order_parts = [f"{field} {direction}" for field, direction in query.order_by_fields]
            order_by_sql = ", ".join(order_parts)
            sql += f"ORDER BY {order_by_sql}\n"

        # Add LIMIT clause if present
        if query.limit_value is not None:
            sql += f"LIMIT {query.limit_value}\n"

        # Add OFFSET clause if present
        if query.offset_value is not None:
            sql += f"OFFSET {query.offset_value}\n"

        return sql
