from .relational_queries import QuerySet, SelectQuerySet, Relation
from .relations import Source
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

        return sql
