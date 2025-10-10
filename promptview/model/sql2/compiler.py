from .relational_queries import QuerySet, SelectQuerySet, Relation
from .relations import Source, RelField
from .expressions import (
    Expression, BinaryExpression, And, Or, Not, IsNull, IsNotNull,
    In, NotIn, Between, Like, ILike, Value
)
import textwrap







class Compiler:
    def __init__(self):
        self.params = []
        self.param_counter = 1

    def compile(self, query: QuerySet):
    
        cte_sql = self.compile_ctes(query)
        sql = ""        
        if isinstance(query, SelectQuerySet):
            sql = self.compile_select_query(query)
        else:
            raise ValueError(f"Unknown query type: {type(query)}")
            
        sql = cte_sql + sql
        return sql

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

        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def compile_select_query(self, query: SelectQuerySet):
        """Compile a SELECT query with FROM and JOIN clauses"""

        sql = "SELECT\n"

        # Compile projection fields
        for field in query.iter_projection_fields():
            if field.is_query:
                # Subquery field
                sub_sql = self.compile(field.source)
                sub_sql = textwrap.indent(sub_sql, "  ")
                sub_sql = f"(\n{sub_sql}\n) AS {field.name},\n"
                sub_sql = textwrap.indent(sub_sql, "    ")
                sql += sub_sql
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
        sql += f"FROM {first_source.final_name}\n"

        # Rest are JOINs
        for source in query.sources[1:]:
            if not isinstance(source, Source):
                raise ValueError(f"Expected Source, got {type(source)}")

            if source.join_on is None:
                raise ValueError(f"Source {source.name} is missing join information")

            # Add JOIN clause
            sql += f"{source.join_type} JOIN {source.final_name} ON {source.get_on_clause()}\n"

        # Add WHERE clause if present
        if query.where_clause:
            where_sql = self.compile_expr(query.where_clause.condition)
            sql += f"WHERE {where_sql}\n"

        return sql
