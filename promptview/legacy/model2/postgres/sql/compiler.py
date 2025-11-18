





import textwrap
from promptview.model.postgres.sql.expressions import BinaryExpression, Coalesce, Expression, RawSQL, RawValue, Value, And, Or, Not, IsNull, In, Between, Like, Function, OrderBy, VectorDistance
from promptview.model.postgres.sql.helpers import NestedQuery
from promptview.model.postgres.sql.queries import Column, DeleteQuery, InsertQuery, SelectQuery, Subquery, Table, UnionQuery, UpdateQuery, Column





    
def tab(text, indent_level=1):
    return textwrap.indent(text, " " * indent_level)





class Compiler:
    def __init__(self):
        self.params = []
        self.param_counter = 1

    def compile(self, query):
        # self.params = []
        # self.param_counter = 1

        if isinstance(query, SelectQuery):
            sql = ""

            # Add CTEs first
            if query.ctes:
                sql += self._compile_ctes(query.ctes, query.recursive) + "\n"

            sql += self._compile_select(query)
        elif isinstance(query, UnionQuery):
            left_sql, _ = self.compile(query.left)
            right_sql, _ = self.compile(query.right)
            sql = f"({left_sql} \n\nUNION ALL\n\n{right_sql})"
        elif isinstance(query, InsertQuery):
            sql = self._compile_insert(query)
        elif isinstance(query, UpdateQuery):
            sql = self._compile_update(query)
        elif isinstance(query, DeleteQuery):
            sql = self._compile_delete(query)
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        return sql, self.params

    def add_param(self, value):
        self.params.append(value)
        placeholder = f"${self.param_counter}"
        self.param_counter += 1
        return placeholder
    

    def compile_expr(self, expr):
        # if isinstance(expr, Column):
        #     table_prefix = f"{expr.table}." if expr.table else ""
        #     base = f"{table_prefix}{expr.name}"
        #     return f"{base} AS {expr.alias}" if expr.alias else base
        if isinstance(expr, Column):
            table_prefix = ""
            if expr.table:
                if isinstance(expr.table, Subquery):
                    table_prefix = f"{expr.table.alias}."                    
                elif isinstance(expr.table, Expression):
                    table_prefix = self.compile_expr(expr.table)
                else:
                    table_prefix = f"{expr.table}."

            base = f"{table_prefix}{expr.name}"
            return f"{base} AS {expr.alias}" if expr.alias else base

        elif isinstance(expr, Value):
            if expr.value == "*":
                return "*"
            elif expr.inline:
                if expr.no_quote:
                    return expr.value
                return repr(expr.value)  # inline as string
            else:
                return self.add_param(expr.value)
        elif isinstance(expr, RawValue):
            return expr.value
        elif isinstance(expr, BinaryExpression):
            left = self.compile_expr(expr.left)
            right = self.compile_expr(expr.right)
            return f"({left} {expr.operator} {right})"

        elif isinstance(expr, And):
            return f"({' AND '.join(self.compile_expr(c) for c in expr.conditions)})"

        elif isinstance(expr, Or):
            return f"({' OR '.join(self.compile_expr(c) for c in expr.conditions)})"

        elif isinstance(expr, Not):
            return f"(NOT {self.compile_expr(expr.condition)})"

        elif isinstance(expr, IsNull):
            return f"({self.compile_expr(expr.value)} IS NULL)"

        elif isinstance(expr, In):
            val = self.compile_expr(expr.value)
            options = ', '.join(self.add_param(opt) for opt in expr.options)
            return f"({val} IN ({options}))"

        elif isinstance(expr, Between):
            val = self.compile_expr(expr.value)
            lower = self.compile_expr(expr.lower)
            upper = self.compile_expr(expr.upper)
            return f"({val} BETWEEN {lower} AND {upper})"

        elif isinstance(expr, Like):
            val = self.compile_expr(expr.value)
            pattern = self.compile_expr(expr.pattern)
            return f"({val} LIKE {pattern})"
        
        elif isinstance(expr, Coalesce):
            args = ", ".join(self.compile_expr(v) for v in expr.values)
            # compiled = f"COALESCE({args})"
            # if expr.alias:
            #     compiled += f" AS {expr.alias}"
            compiled = f"COALESCE(\n{tab(args)}\n)"
            if expr.alias:
                compiled += f" AS {expr.alias}"
            return compiled
        
        elif isinstance(expr, SelectQuery):
            # Compile subquery and inline it (only the SQL part, not the parameters)
            subquery_sql, _ = self.compile(expr)
            return f"(\n{tab(subquery_sql)}\n)"

        elif isinstance(expr, Function):
            # args = ", ".join(self.compile_expr(arg) for arg in expr.args)
            # args = ", \n ".join(self.compile_expr(arg) for arg in expr.args)
            comp_args = [self.compile_expr(arg) for arg in expr.args]
            if len(comp_args) % 2 == 0:
                args = ", \n".join([f"{comp_args[i]}, {comp_args[i+1]}" for i in range(0, len(comp_args) - 1, 2)])
            else:
                args = ", ".join(comp_args)
            if expr.distinct:
                args = f"DISTINCT {args}"
            compiled = f"{expr.name}(\n{tab(args)}\n)"
            if expr.filter_where:
                compiled += f" FILTER (WHERE {self.compile_expr(expr.filter_where)})"
            if expr.order_by:
                compiled += f" ORDER BY {', '.join(self.compile_expr(c) for c in expr.order_by)}"
            if expr.alias:
                compiled += f" AS {expr.alias}"
            return compiled
        elif isinstance(expr, OrderBy):
            return f"{self.compile_expr(expr.column)} {expr.direction}"
        elif isinstance(expr, VectorDistance):
            left = self.compile_expr(expr.left)
            right = self.compile_expr(expr.right)
            return f"({left} {expr.operator} {right})"
        elif isinstance(expr, NestedQuery):
            query = expr.build_query()
            return self.compile_expr(query)
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def compile_joins(self, joins):
        return " ".join(
            f"{join.join_type} JOIN {self.compile_table(join.table)} ON {self.compile_expr(join.condition)}"
            for join in joins
        )

    def compile_table(self, table):
        if isinstance(table, Subquery):
            sub_sql, _ = self.compile(table.query)  # compile inner query
            return f"({sub_sql}) AS {table.alias}"
        if isinstance(table, SelectQuery) and table._is_subquery:
            return f"({self._compile_select(table)}) AS {table.alias}"
        if hasattr(table, "name") and hasattr(table, "alias"):
            return f"{table.name}" + (f" AS {table.alias}" if table.alias else "")

        return str(table)


    def _compile_select(self, q: SelectQuery):
        if not q.columns:
            raise ValueError("Select query has no columns")
        sql = "SELECT \n"
        
        if q.distinct_on:
            distinct_on_exprs = ", ".join(self.compile_expr(col) for col in q.distinct_on)
            sql += f"DISTINCT ON ({distinct_on_exprs}) "
        elif q.distinct:
            sql += "DISTINCT "
            
        sql += tab(", \n".join(self.compile_expr(col) for col in q.columns or ['*']))

        sql += f"\nFROM {self.compile_table(q.from_table)}"

        if q.joins:
            sql += f"\n{self.compile_joins(q.joins)}"

        if q.where.condition:
            sql += f"\nWHERE {self.compile_expr(q.where.condition)}"

        if q.group_by:
            sql += "\nGROUP BY " + ", ".join(self.compile_expr(c) for c in q.group_by)

        if q.having:
            sql += f"\nHAVING {self.compile_expr(q.having)}"

        if q.order_by:
            sql += "\nORDER BY " + ", ".join(self.compile_expr(c) for c in q.order_by)

        if q.limit is not None:
            sql += f"\nLIMIT {q.limit}"

        if q.offset is not None:
            sql += f"\nOFFSET {q.offset}"

        return sql



    def _compile_insert(self, q: InsertQuery):
        table = self.compile_table(q.table)

        # Compile column names
        columns = ", ".join(col.name for col in q.columns)

        # Compile values
        value_rows = []
        for row in q.values:
            placeholders = [self.compile_expr(v) for v in row]
            value_rows.append(f"({', '.join(placeholders)})")
        values_sql = ",\n".join(value_rows)

        sql = f"INSERT INTO {table} ({columns})\nVALUES {values_sql}"

        # RETURNING clause
        if q.returning:
            returning = ", ".join(self.compile_expr(c) for c in q.returning)
            sql += f"\nRETURNING {returning}"

        return sql

    
    def _compile_update(self, q: UpdateQuery):
        table = self.compile_table(q.table)

        # SET clause
        set_fragments = []
        for col, val in q.set_clauses:
            col_sql = self.compile_expr(col)
            val_sql = self.compile_expr(val)
            set_fragments.append(f"{col_sql} = {val_sql}")
        set_clause = ", ".join(set_fragments)

        sql = f"UPDATE {table}\nSET {set_clause}"

        # WHERE clause
        if q.where.condition:
            sql += f"\nWHERE {self.compile_expr(q.where.condition)}"

        # RETURNING clause
        if q.returning:
            returning_sql = ", ".join(self.compile_expr(c) for c in q.returning)
            sql += f"\nRETURNING {returning_sql}"

        return sql


    def _compile_delete(self, q: DeleteQuery):
        table = self.compile_table(q.table)
        sql = f"DELETE FROM {table}"

        # WHERE clause
        if q.where:
            sql += f"\nWHERE {self.compile_expr(q.where.condition)}"

        # RETURNING clause
        if q.returning:
            returning_sql = ", ".join(self.compile_expr(c) for c in q.returning)
            sql += f"\nRETURNING {returning_sql}"

        return sql

    
    
    def _compile_ctes(self, ctes, recursive=False):
        parts = []
        for alias, cte_query in ctes:
            if isinstance(cte_query, RawSQL):
                parts.append(f"{alias} AS ({cte_query.sql})")
                self.params.extend(cte_query.params)
            else:
                cte_sql, params = self.compile(cte_query)
                parts.append(f"{alias} AS ({cte_sql})")
                # self.params.extend(params)
        prefix = "WITH "
        if recursive:
            prefix = "WITH RECURSIVE "
        return prefix + ",\n     ".join(parts)
