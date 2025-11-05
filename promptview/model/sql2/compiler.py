from .relational_queries import QuerySet, SelectQuerySet, Relation
from .relations import Source, RelField, RawRelation
from .expressions import (
    Expression, BinaryExpression, And, Or, Not, IsNull, IsNotNull,
    In, NotIn, Between, Like, ILike, Value, Raw, JsonBuildObject, JsonAgg, JsonbAgg,
    Count, Sum, Avg, Min, Max, AggregateFunction, Coalesce,
    LtreeNlevel, LtreeSubpath, LtreeLca
)
import textwrap







class Compiler:
    def __init__(self):
        self.params = []
        self.param_counter = 1
        self.indent_level = 0  # Track current indentation level for nested expressions
        self.cte_map = {}  # Map QuerySet id -> CTE name for detecting CTE references

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
            # Use alias if provided, otherwise generate name from final_name
            if cte.alias:
                cte_name = cte.alias
            else:
                cte_name = cte.final_name + "_cte"

            # Register this CTE so we can reference it later
            self.cte_map[id(cte)] = cte_name

            # Compile the CTE based on its type
            if isinstance(cte, RawRelation):
                # Raw SQL CTE - use the SQL directly
                cte_sql = cte.sql.strip()
            else:
                # QuerySet CTE - compile it
                cte_sql, _ = self.compile(cte)  # Unpack tuple (sql, params)

            cte_sql = "\n" + textwrap.indent(cte_sql, "    ")
            ctes_sql.append(f"{cte_name} AS ({cte_sql})")
        return "WITH " + ", ".join(ctes_sql) + "\n"
            
            
            
    def compile_expr(self, expr: Expression) -> str:
        """Compile an expression to SQL"""
        if isinstance(expr, RelField):
            # Field reference
            return f"{expr.source.final_name}.{expr.name}"

        elif isinstance(expr, Raw):
            # Raw SQL escape hatch
            # Compile any parameters that were passed
            for param in expr.params:
                self.compile_expr(param)
            # Return the raw SQL string as-is
            return expr.sql

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
            # Check if the inner expression is complex (JsonBuildObject)
            if isinstance(expr.expr, JsonBuildObject):
                # Increase indent for nested content
                self.indent_level += 1
                inner_sql = self.compile_expr(expr.expr)
                self.indent_level -= 1

                # Format with newlines if JsonBuildObject was multi-line
                if '\n' in inner_sql:
                    indent = "    " * (self.indent_level + 1)
                    inner_sql_indented = textwrap.indent(inner_sql, indent)
                    return f"json_agg(\n{inner_sql_indented}\n{' ' * (self.indent_level * 4)})"
                else:
                    return f"json_agg({inner_sql})"
            else:
                inner_sql = self.compile_expr(expr.expr)
                return f"json_agg({inner_sql})"

        elif isinstance(expr, JsonbAgg):
            # jsonb_agg(expression) - same as json_agg but returns jsonb type
            # Check if the inner expression is complex (JsonBuildObject)
            if isinstance(expr.expr, JsonBuildObject):
                # Increase indent for nested content
                self.indent_level += 1
                inner_sql = self.compile_expr(expr.expr)
                self.indent_level -= 1

                # Format with newlines if JsonBuildObject was multi-line
                if '\n' in inner_sql:
                    indent = "    " * (self.indent_level + 1)
                    inner_sql_indented = textwrap.indent(inner_sql, indent)
                    return f"jsonb_agg(\n{inner_sql_indented}\n{' ' * (self.indent_level * 4)})"
                else:
                    return f"jsonb_agg({inner_sql})"
            else:
                inner_sql = self.compile_expr(expr.expr)
                return f"jsonb_agg({inner_sql})"

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

        elif isinstance(expr, Coalesce):
            # COALESCE(value1, value2, ...)
            values_sql = []
            for value in expr.values:
                # Check if it's a QuerySet (subquery)
                if isinstance(value, QuerySet):
                    # Compile as subquery
                    sub_sql, _ = self.compile(value)
                    # Indent the subquery for readability
                    sub_sql = textwrap.indent(sub_sql.rstrip(), "    ")
                    values_sql.append(f"(\n{sub_sql}\n    )")
                else:
                    # Regular expression
                    values_sql.append(self.compile_expr(value))
            return f"COALESCE({', '.join(values_sql)})"

        # LTREE Functions
        elif isinstance(expr, LtreeNlevel):
            # nlevel(ltree) - returns number of labels in path
            field_sql = self.compile_expr(expr.field)
            return f"nlevel({field_sql})"

        elif isinstance(expr, LtreeSubpath):
            # subpath(ltree, offset [, len])
            field_sql = self.compile_expr(expr.field)
            if expr.length is not None:
                return f"subpath({field_sql}, {expr.offset}, {expr.length})"
            else:
                return f"subpath({field_sql}, {expr.offset})"

        elif isinstance(expr, LtreeLca):
            # lca(ltree, ltree, ...) - lowest common ancestor
            fields_sql = [self.compile_expr(field) for field in expr.fields]
            return f"lca({', '.join(fields_sql)})"

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

        # Check if this is a scalar subquery
        if query.scalar_expr is not None:
            # Scalar subquery: just compile the expression without alias
            expr_sql = self.compile_expr(query.scalar_expr)
            sql += f"    {expr_sql}\n"
        else:
            # Regular query: compile projection fields with aliases
            for field in query.iter_projection_fields():
                if field.is_query:
                    # Subquery field (from include or select_subquery)
                    # Compile the subquery - it will automatically use CTE references
                    # if its FROM source is a registered CTE
                    sub_sql, _ = self.compile(field.source)
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

        # Check if first source is a QuerySet (subquery)
        if isinstance(first_source.base, QuerySet):
            # Check if this is a CTE reference
            if id(first_source.base) in self.cte_map:
                # Reference the CTE by name
                cte_name = self.cte_map[id(first_source.base)]
                sql += f"FROM {cte_name} AS {first_source.final_name}\n"
            else:
                # Inline subquery
                sub_sql, _ = self.compile(first_source.base)
                sub_sql = textwrap.indent(sub_sql, "    ")
                sql += f"FROM (\n{sub_sql}\n) AS {first_source.final_name}\n"
        # Check if first source is a RawRelation
        elif isinstance(first_source.base, RawRelation):
            # Check if this RawRelation is a registered CTE
            if id(first_source.base) in self.cte_map:
                # Reference the CTE by name
                cte_name = self.cte_map[id(first_source.base)]
                sql += f"FROM {cte_name} AS {first_source.final_name}\n"
            else:
                # Inline raw SQL in parentheses
                raw_sql = textwrap.indent(first_source.base.sql.strip(), "    ")
                sql += f"FROM (\n{raw_sql}\n) AS {first_source.final_name}\n"
        else:
            # Regular table
            sql += f"FROM {first_source.final_name}\n"

        # Rest are JOINs
        for source in query.sources[1:]:
            if not isinstance(source, Source):
                raise ValueError(f"Expected Source, got {type(source)}")

            if source.join_on is None:
                raise ValueError(f"Source {source.name} is missing join information")

            # Check if source is a QuerySet (subquery)
            if isinstance(source.base, QuerySet):
                # Check if this is a CTE reference
                if id(source.base) in self.cte_map:
                    # Reference the CTE by name
                    cte_name = self.cte_map[id(source.base)]
                    sql += f"{source.join_type} JOIN {cte_name} AS {source.final_name} ON {source.get_on_clause()}\n"
                else:
                    # Inline subquery for JOIN
                    sub_sql, _ = self.compile(source.base)
                    sub_sql = textwrap.indent(sub_sql, "    ")
                    sql += f"{source.join_type} JOIN (\n{sub_sql}\n) AS {source.final_name} ON {source.get_on_clause()}\n"
            # Check if source is a RawRelation
            elif isinstance(source.base, RawRelation):
                # Check if this RawRelation is a registered CTE
                if id(source.base) in self.cte_map:
                    # Reference the CTE by name
                    cte_name = self.cte_map[id(source.base)]
                    sql += f"{source.join_type} JOIN {cte_name} AS {source.final_name} ON {source.get_on_clause()}\n"
                else:
                    # Inline raw SQL in parentheses
                    raw_sql = textwrap.indent(source.base.sql.strip(), "    ")
                    sql += f"{source.join_type} JOIN (\n{raw_sql}\n) AS {source.final_name} ON {source.get_on_clause()}\n"
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
