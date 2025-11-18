




import re
from typing import Literal

from promptview.model.postgres.sql.joins import InnerJoin, Join
from promptview.model.postgres.sql.expressions import Function, Coalesce, Eq, Expression, Neq, Gt, Gte, Lt, Lte, OrderBy, Value, WhereClause, json_build_object, param, In

class Table:
    def __init__(self, name, alias=None):
        self.name = name
        self.alias = alias
        
        
    def __str__(self):
        return self.alias or self.name

    def __repr__(self):
        return f"<Table {self.name} AS {self.alias}>" if self.alias else f"<Table {self.name}>"


def is_camel_case(name: str) -> bool:
    return bool(re.fullmatch(r'[a-z]+(?:[A-Z][a-z]*)+', name))


class Column:
    def __init__(self, name, table=None, alias=None):
        # some table coumns have camel case names
        #the query returned parenthesized column name if it is camel case
        # self.name = name
        self.name = name if not is_camel_case(name) else f'"{name}"'
        # self.name = name if all(c.islower() for c in name) else f'"{name}"'
        self.table = table
        self.alias = alias
        
    def __str__(self):
        prefix = f"{str(self.table)}." if self.table else ""
        base = f"{prefix}{self.name}"
        return f"{base} AS {self.alias}" if self.alias else base

    def __repr__(self):
        return str(self)
    
    
    # Inside Column class (or better: a ColumnExpression wrapper if you want to separate it from raw SQL)

    def __eq__(self, other):
        return Eq(self, param(other) if not isinstance(other, Expression) else other)

    def __ne__(self, other):
        return Neq(self, param(other) if not isinstance(other, Expression) else other)

    def __gt__(self, other):
        return Gt(self, param(other) if not isinstance(other, Expression) else other)

    def __ge__(self, other):
        return Gte(self, param(other) if not isinstance(other, Expression) else other)

    def __lt__(self, other):
        return Lt(self, param(other) if not isinstance(other, Expression) else other)

    def __le__(self, other):
        return Lte(self, param(other) if not isinstance(other, Expression) else other)
    
    def isin(self, values: list):
        return In(self, values)


JoinType = Literal["LEFT", "RIGHT", "INNER"]



        
        



class SelectQuery:
    def __init__(self):
        self.columns = []
        self.from_table = None
        self.joins = []
        self.where = WhereClause()
        self.group_by = []
        self.having = None
        self.order_by = []
        self.limit = None
        self.offset = None
        self.distinct = False
        self.distinct_on = [] 
        self.alias = None
        self.ctes = []  
        self.recursive = False  # <-
        self._is_subquery = False
    
    
    def copy_query(self, exclude: set[str] = set()):
        new_query = SelectQuery()
        if "columns" not in exclude:
            new_query.columns = self.columns
        if "from_table" not in exclude:
            new_query.from_table = self.from_table
        if "joins" not in exclude:
            new_query.joins = self.joins
        if "where" not in exclude:
            new_query.where = self.where
        if "group_by" not in exclude:
            new_query.group_by = self.group_by
        if "having" not in exclude:
            new_query.having = self.having
        if "order_by" not in exclude:
            new_query.order_by = self.order_by
        if "limit" not in exclude:
            new_query.limit = self.limit
        if "offset" not in exclude:
            new_query.offset = self.offset
        if "distinct" not in exclude:
            new_query.distinct = self.distinct
        if "distinct_on" not in exclude:
            new_query.distinct_on = self.distinct_on
        if "alias" not in exclude:
            new_query.alias = self.alias
        if "ctes" not in exclude:
            new_query.ctes = self.ctes
        if "recursive" not in exclude:
            new_query.recursive = self.recursive
        return new_query
    
    def join(self, table, condition, join_type='LEFT', alias=None):
        self.joins.append(Join(table, condition, join_type, alias))
        return self

    def left_join(self, table, condition, alias=None):
        return self.join(table, condition, 'LEFT', alias)

    def right_join(self, table, condition, alias=None):
        return self.join(table, condition, 'RIGHT', alias)

    def inner_join(self, table, condition, alias=None):
        return self.join(table, condition, 'INNER', alias)
    
    
    def select(self, *columns):
        self.columns = list(columns)
        return self

    def from_(self, table):
        self.from_table = table
        return self


    def group_by_(self, *cols):
        self.group_by = list(cols)
        return self

    def order_by_(self, *cols):
        self.order_by = list(cols)
        return self

    def limit_(self, count):
        self.limit = count
        return self

    def offset_(self, count):
        self.offset = count
        return self

    def with_cte(self, name, query, recursive: bool = False):
        self.ctes.append((name, query))
        if self.recursive == False:
            self.recursive = recursive
        return self

    def distinct_(self, enabled=True):
        self.distinct = enabled
        return self
        
    def distinct_on_(self, *cols):
        self.distinct_on = list(cols)
        return self

    def as_list_subquery(self, parent_table, primary_key, foreign_key, use_joins=False):
        """
        Returns a the query as a subquery.
        parent_table - the parent table
        primary_key - the primary key of the parent table
        foreign_key - the foreign key that joins this query to the parent table
        """
        try:
            columns = {}
            join_filter = set()
            for c in self.columns:
                if isinstance(c, Column):
                    columns[c.name] = c
                elif isinstance(c, Expression):
                    columns[c.alias] = c
                    join_filter.add(c.alias)
                    c.alias = None
            obj = json_build_object(**columns)
            obj.order_by = self.order_by
            obj.limit = self.limit
            obj.offset = self.offset
            subq = self.copy_query(exclude={"group_by", "order_by", "limit", "offset"})
            subq.columns = [Function(
                    "json_agg", 
                    obj, 
                    # order_by=query.order_by
                    )]
            subq.from_table = self.from_table
            if use_joins:
                subq.joins = [j for j in self.joins if j.table.name not in join_filter]
            
            subq.where &= Eq(Column(foreign_key, self.from_table), Column(primary_key, parent_table))
            
            coalesced = Coalesce(subq, Value("[]", inline=True))
            return coalesced
        except Exception as e:
            print(e)
            raise
        
    def as_subquery(self, parent_table, primary_key, foreign_key, use_joins=False):
        """
        Returns a the query as a subquery.
        parent_table - the parent table
        primary_key - the primary key of the parent table
        foreign_key - the foreign key that joins this query to the parent table
        """
        try:
            columns = {}
            join_filter = set()
            for c in self.columns:
                if isinstance(c, Column):
                    columns[c.name] = c
                elif isinstance(c, Expression):
                    columns[c.alias] = c
                    join_filter.add(c.alias)
                    c.alias = None
            obj = json_build_object(**columns)
            subq = self.copy_query(exclude={"group_by", "order_by", "limit", "offset"})            
            subq.from_table = self.from_table
            subq.columns = [obj]
            if use_joins:
                subq.joins = [j for j in self.joins if j.table.name not in join_filter]
            
            subq.where &= Eq(Column(foreign_key, self.from_table), Column(primary_key, parent_table))
            
            coalesced = Coalesce(subq, Value("{}", inline=True))
            return coalesced
        except Exception as e:
            print(e)
            raise
        




class NestedSubquery:
    def __init__(
        self, 
        query: SelectQuery, 
        alias: str, 
        primary_col: Column,
        foreign_col: Column,
        junction_col: tuple[Column, Column] | None = None,
        type: Literal["one_to_one", "one_to_many", "many_to_many"] | None = None,
        
    ):
        self.query = query
        self.alias = alias
        self.type = type
        self.primary_col = primary_col
        self.foreign_col = foreign_col
        self.junction_col = junction_col
        
    def get_where_clause(self):
        return Eq(self.primary_col, self.foreign_col)
    
    
    
    def get_join(self):
        if self.junction_col:
            return Join(self.foreign_col.table, Eq(self.foreign_col, self.junction_col[1]))
        else:
            return Join(self.primary_col.table, self.get_where_clause())


class Subquery:
    def __init__(self, query: SelectQuery, alias: str):
        self.query = query
        self.alias = alias      
        
        
class InsertQuery:
    def __init__(self, table):
        self.table = table
        self.columns = []
        self.values = []  # List of rows (each row is a list of values)
        self.returning = []
        self.on_conflict = None  # Optional: for UPSERT



class UpdateQuery:
    def __init__(self, table):
        self.table = table
        self.set_clauses = []
        self.where = WhereClause()
        self.returning = []
        
    def set(self, values: list[tuple[Column, Value]]):
        self.set_clauses = values
        return self


class DeleteQuery:
    def __init__(self, table):
        self.table = table
        self.where = WhereClause()
        self.returning = []





class UnionQuery:
    def __init__(self, left: SelectQuery, right: SelectQuery):
        self.left = left
        self.right = right
        
    
        