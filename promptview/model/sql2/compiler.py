from .relational_queries import QuerySet, SelectQuerySet, JoinedRelation, Relation
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
            
        sql = "SELECT\n"
        for field in query.iter_projection_fields():
            if field.is_query:
                sub_sql = self.compile(field.source)            
                sub_sql = textwrap.indent(sub_sql, "  ")
                sub_sql = f"(\n{sub_sql}\n) AS {field.name},\n"
                sub_sql = textwrap.indent(sub_sql, "    ")
                sql += sub_sql
            else:
                sql += f"    {field.source.final_name}.{field.name},\n"
        # sql += f"FROM {query.target.name}"
        sql += f"FROM "
        
        for source in query.sources:
            if not isinstance(source, JoinedRelation):
                sql += f"{source.final_name}\n"
            elif isinstance(source, Relation):
                sql += f"{source.join_type} JOIN {source.final_name} ON {source.get_on_clause()}\n"
            else:
                continue
        return sql
