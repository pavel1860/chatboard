from chatboard.model.postgres.sql.queries import SelectQuery, NestedSubquery,  Table, Column, param
from chatboard.model.postgres.sql.compiler import Compiler
from chatboard.model.postgres.sql.joins import InnerJoin, LeftJoin, RightJoin, FullJoin
from chatboard.model.postgres.sql.expressions import Eq, Function, Coalesce, Null, Value, Expression
from chatboard.model.postgres.sql.joins import Join
from copy import deepcopy




class Preprocessor:
    
    
    def process_nested_subquery(self, subquery: NestedSubquery, depth: int, parent_query: SelectQuery) -> tuple[SelectQuery | Expression, list[Join]]:
        joins = []
        query = deepcopy(subquery.query)
        wrap_in_array = subquery.type != "one_to_one"
        if subquery.junction_col:
            query.from_table = subquery.junction_col[1].table
            query.joins.append(subquery.get_join())
            query.where &= Eq(subquery.primary_col, subquery.junction_col[0])
        else:
            query.where.and_(subquery.get_where_clause())
        query = self.process(query, depth=depth+1, parent_query=parent_query, wrap_in_array=wrap_in_array)
        
        
        return query, joins
    
    def _query_to_json(
        self, 
        query: SelectQuery, 
        json_obj: Expression, 
        group_by: list[Column] = [],
        joins: list[Join] = [],
        wrap_in_array: bool = False
    ):
        compiled = SelectQuery()
        if query.order_by:
            json_obj.order_by = query.order_by
        if wrap_in_array:
            json_obj = Function("json_agg", json_obj)
        compiled.select(json_obj)
        compiled.from_table = query.from_table
        compiled.where = query.where
        
        # if parent_query:
            # compiled.where.and_(Eq(Column(foreign_key, parent_table), Column(primary_key, query.from_table)))
        compiled.joins = query.joins + joins
        # compiled.order_by = query.order_by
        compiled.group_by = query.group_by + group_by
        compiled.limit = query.limit
        compiled.offset = query.offset
        compiled.distinct = query.distinct
        compiled.distinct_on = query.distinct_on
        compiled._is_subquery = True
        return compiled
    
    def process(
        self, 
        query: SelectQuery, depth: int = 0, 
        parent_query: SelectQuery | None = None,
        wrap_in_array=False,
    ) -> SelectQuery | Expression:
        query = deepcopy(query)
        joins = []

        json_pairs = []
        for col in query.columns:
            if isinstance(col, Column) and isinstance(col.table, NestedSubquery):
                # Nested subquery: compile it and embed as JSON field
                nested_alias = col.alias or col.name
                nested_subquery, sub_joins = self.process_nested_subquery(col.table, depth=depth+1, parent_query=query)
                joins.extend(sub_joins)
                json_pairs.append(Value(nested_alias))
                json_pairs.append(nested_subquery)
            else:
                json_pairs.append(Value(col.alias or col.name))
                json_pairs.append(col)
                
        json_obj = Function("jsonb_build_object", *json_pairs)
                    
        json_obj = self._query_to_json(query, json_obj, joins=joins, wrap_in_array=wrap_in_array)
        
        default_value = Value("[]", inline=True) if wrap_in_array else Null()
        return Coalesce(json_obj, default_value)
        

    def process_query(self, query: SelectQuery) -> SelectQuery | Expression:
        query = deepcopy(query)
        joins = []
        need_to_group_by = False
        for col in query.columns:
            if isinstance(col, Column) and isinstance(col.table, NestedSubquery):
                # Nested subquery: compile it and embed as JSON field
                nested_alias = col.alias or col.name
                nested_subquery, sub_joins = self.process_nested_subquery(col.table, depth=1, parent_query=query)
                joins.extend(sub_joins)
                col.alias = nested_alias
                col.table = nested_subquery
                col.name = ''
                need_to_group_by = True
                
        query.joins += joins
        if need_to_group_by:
            query.group_by = [Column("id", query.from_table)]
        return query