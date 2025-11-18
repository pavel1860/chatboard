

from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, List, Literal, Self, Type
from typing_extensions import TypeVar

from promptview.model.base_namespace import NSManyToManyRelationInfo, Namespace, NSRelationInfo

from promptview.model.postgres.fields_query import PgFieldInfo
# from promptview.model2.query_filters import QueryProxy
from promptview.model.postgres.sql.helpers import NestedQuery
from promptview.model.postgres.sql.json_processor import Preprocessor
from promptview.utils.db_connections import PGConnectionManager


from promptview.model.postgres.sql.queries import JoinType, NestedSubquery, SelectQuery, Table, Column, Subquery
from promptview.model.postgres.sql.expressions import Eq, Expression, IsNull, Not, RawSQL, Value, Function, Coalesce, Gt, json_build_object, param, OrderBy
from promptview.model.postgres.sql.compiler import Compiler
from functools import reduce
from operator import and_



# Your ORM interfaces
if TYPE_CHECKING:
    from promptview.model.model import Model

class QueryWrapper:
    def __init__(self, query: SelectQuery, type: Literal["select", "join_nested", "join_first"]):
        self.query = query
        self.type = type

    

MODEL = TypeVar("MODEL", bound="Model")

T_co = TypeVar("T_co", covariant=True)

class QuerySetSingleAdapter(Generic[T_co]):
    def __init__(self, queryset: "SelectQuerySet[T_co]"):
        self.queryset = queryset

    def __await__(self) -> Generator[Any, None, T_co]:
        async def await_query():
            results = await self.queryset.execute()
            if results:
                return results[0]
            return None
            # raise ValueError("No results found")
            # return None
            # raise DoesNotExist(self.queryset.model)
        return await_query().__await__()  



class QueryProxy:
    def __init__(self, model_class, table: Table):
        self.model_class = model_class
        self.table = table
        # self.table.alias = model_class.get_namespace().table_alias  # or pass in if needed

    def __getattr__(self, field_name):
        return Column(field_name, self.table)


def wrap_query_in_json_agg(query: SelectQuery, alias: str, pk_col: Column) -> Coalesce:
    obj = json_build_object(**{col.name: col for col in query.columns})
    agg = Function(
        "json_agg",
        obj,
        distinct=True,
        filter_where=Not(IsNull(pk_col))
    )
    return Coalesce(agg, Value("[]", inline=True), alias=alias)

# def embed_query_as_subquery(outer_query: SelectQuery, inner_query: SelectQuery, alias: str):
#     subquery = Subquery(inner_query, alias=alias)
#     outer_query.from_table = subquery
#     return outer_query


def join_as_subquery(query, rel, parent_table):
    try:
        columns = {}
        join_filter = set()
        for c in query.columns:
            if isinstance(c, Column):
                columns[c.name] = c
            elif isinstance(c, Expression):
                columns[c.alias] = c
                join_filter.add(c.alias)
                c.alias = None
        obj = json_build_object(**columns)
        obj.order_by = query.order_by
        subq = SelectQuery()
        subq.columns = [Function(
                "json_agg", 
                obj, 
                # order_by=query.order_by
                )]
        subq.from_table = query.from_table
        subq.joins = [j for j in query.joins if j.table.name not in join_filter]
        
        subq.where &= Eq(Column(rel.foreign_key, query.from_table), Column(rel.primary_key, parent_table))
        
        coalesced = Coalesce(subq, Value("[]", inline=True))
        return coalesced
    except Exception as e:
        print(e)
        raise



def include_junction_table(
    query_set: "SelectQuerySet", 
    rel: NSManyToManyRelationInfo, 
    join_type: JoinType = "LEFT", 
    alias: str | None = None
):
    j_ns = rel.junction_namespace
    target_table = query_set.from_table
    junction_table = Table(j_ns.table_name, alias=alias)
    query_set.query.from_table = junction_table
    query_set.query.join(
        target_table, 
        Eq(
            Column(rel.foreign_key, target_table), 
            Column(rel.junction_keys[1], junction_table)
        ),
        join_type
    )
    nested_query = join_as_subquery(query_set.query, rel, query_set.from_table)    
    nested_query.values[0].where(Eq(Column(rel.junction_keys[0], junction_table), Column(rel.primary_key, query_set.from_table)))
    return nested_query
 
class SelectQuerySet(Generic[MODEL]):
    
    
    def __init__(self, model_class: Type[MODEL], query: SelectQuery | None = None, alias: str | None = None, parse: Callable[[MODEL], Any] | None = None):
        self.model_class = model_class
        self.alias_lookup = {}        
        self.table_lookup = {}
        table = Table(model_class.get_namespace().table_name)
        table.alias = self._set_alias(table.name, alias)
        self.table_lookup[str(table)] = table
        self.query = SelectQuery().from_(table) if query is None else query
        self._params = []
        self._parse = parse

    def _set_alias(self, name: str, alias: str | None = None) -> str:
        if alias is not None:
            if alias in self.alias_lookup:
                raise ValueError(f"Alias {alias} already exists")
            self.alias_lookup[alias] = name
            return alias
        base = name[0].lower()
        alias = base
        for i in range(10):
            if alias not in self.alias_lookup:
                break
            alias = f"{base}{i}"
        self.alias_lookup[alias] = name
        return alias
    
    
    def _gen_alias(self, name: str) -> str:
        base = name[0].lower()
        alias = base
        for i in range(10):
            if alias not in self.table_lookup:
                break
            alias = f"{base}{i}"
        return alias
    
    
    def __await__(self):
        return self.execute().__await__()

    @property
    def namespace(self) -> Namespace[MODEL, PgFieldInfo]:
        return self.model_class.get_namespace()
    
    @property
    def from_table(self):
        return self.query.from_table
    
    @property
    def ctes(self):
        return self.query.ctes
    

    
    def select(self, *fields: str) -> "SelectQuerySet[MODEL]":
                
        if len(fields) == 1 and fields[0] == "*":
            columns = [Column(f.name, self.from_table) for f in self.model_class.iter_fields()]
            # o2o_columns = [Column(f.name, self.from_table) for f in self.namespace.iter_relations(is_one_to_one=True)]
            self.query.select(*columns)
        else:
            self.query.select(*[Column(f, self.from_table) for f in fields])
        return self
    
    
    def distinct_on(self, *fields: str) -> "SelectQuerySet[MODEL]":
        self.query.distinct_on_(*[Column(f, self.from_table) for f in fields])
        return self


    def where(self, condition: Callable[[MODEL], Any] | None = None, **kwargs) -> "SelectQuerySet[MODEL]":
        expressions = []

        if condition is not None:
            proxy = QueryProxy(self.model_class, self.from_table)
            # self.query.where(condition(proxy))
            self.query.where &= condition(proxy)
        if kwargs:
            for field, value in kwargs.items():
                col = Column(field, self.from_table)
                expressions.append(Eq(col, param(value)))
                
        if expressions:
            expr = reduce(and_, expressions) if len(expressions) > 1 else expressions[0]
            # self.query.where(expr)
            self.query.where &= expr

        return self
    
    
    def filter(self, condition: Callable[[MODEL], Any] | None = None, **kwargs) -> "SelectQuerySet[MODEL]":
        return self.where(condition, **kwargs)
    
    def _get_query_set(self, target: "SelectQuerySet | Type[Model]") -> "SelectQuerySet":
        if isinstance(target, SelectQuerySet):
            query_set = target
        else:
            # query_set = SelectQuerySet(target).select("*")
            # query_set = target.query()
            query_set = target.get_namespace().query()
        # self._merge_alias_lookup(query_set)
        self._merge_table_lookup(query_set)
        self._gen_query_set_alias(query_set)        
        return query_set
    
    def _merge_alias_lookup(self, query_set: "SelectQuerySet"):
        for alias, name in query_set.alias_lookup.items():
            if alias in self.alias_lookup:
                raise ValueError(f"Alias {alias} already exists")
            self.alias_lookup[alias] = name
    
    
    def _merge_table_lookup(self, query_set: "SelectQuerySet"):
        for table in query_set.table_lookup.values():
            if str(table) in self.table_lookup:
                table.alias = self._gen_alias(table.name)
                # raise ValueError(f"Table {table} already exists")
            self.table_lookup[str(table)] = table
        
    def _gen_query_set_alias(self, query: "SelectQuerySet"):
        if not query.query.from_table:
            raise ValueError("Query set has no from table")
        query.query.from_table.alias = self._set_alias(query.query.from_table.name)
        return query
    
    def with_cte(self, name: str, query: "SelectQuery | SelectQuerySet | RawSQL", recursive: bool = False):
        if isinstance(query, SelectQuerySet):
            if query.query.ctes:
                # move nested cte to the top level
                self.query.ctes = query.query.ctes + self.query.ctes
                if query.query.recursive:
                    query.query.recursive = False
                    self.query.recursive = True
                query.query.ctes = []
        if hasattr(query, "query"):  # If it's a SelectQuerySet
            query = query.query
            
        self.query.with_cte(name, query, recursive=recursive)    
        return self
    
    
    def join_cte(self, cte_name: str, on_left: str, on_right: str, alias=None, join_type: JoinType = "LEFT"):
        """
        Join a Common Table Expression (CTE) to the current query.

        Args:
            cte_name (str): Name of the CTE to join with
            on_left (str): Column name from the current table to use in the join condition
            on_right (str): Column name from the CTE to use in the join condition 
            alias (str, optional): Alias to give the CTE in the join. Defaults to None.

        Returns:
            SelectQuerySet[MODEL]: Returns self for method chaining

        Example:
            query.with_cte("my_cte", subquery) \
                .join_cte("my_cte", "id", "parent_id")
        """
        cte_table = Table(cte_name, alias=alias)
        self.query.join(
            cte_table,
            Eq(Column(on_left, self.from_table), Column(on_right, cte_table)),
            join_type
        )
        return self
    
    
    def _copy_ctes(self, query_set: "SelectQuerySet"):
        cte_lookup = {}
        for cte in query_set.query.ctes:
            cte_lookup[cte[0]] = cte
        for cte in self.query.ctes:
            cte_lookup[cte[0]] = cte
            # if cte.name in cte_lookup:
                # cte.query = cte_lookup[cte.name].query
        return [cte for cte in cte_lookup.values()]
    
    def _merge_ctes(self, query_set: "SelectQuerySet"):
        self.query.ctes = self._copy_ctes(query_set)
        query_set.query.ctes = []
        if query_set.query.recursive:
            self.query.recursive = True
            query_set.query.recursive = False
        return query_set
    
    def join(self, target: "SelectQuerySet | Type[Model]", join_type: JoinType = "LEFT") -> "SelectQuerySet[MODEL]":
        """
        Join a table to the current query.
        target can be a SelectQuerySet or a Model class.
        if target is a Model class the currentmodel needs to have a relation with the target model.
        join_type can be "LEFT", "RIGHT", "INNER", "FULL OUTER"
        """
        query_set = self._get_query_set(target)
        rel = self.namespace.get_relation_by_type(query_set.model_class)
        if rel is None:
            raise ValueError("No relation found")
        if query_set.query.ctes:
            query_set = self._merge_ctes(query_set)

        if isinstance(rel, NSManyToManyRelationInfo):            
            j_ns = rel.junction_namespace
            j_alias=self._set_alias(j_ns.table_name)
            target_table = query_set.from_table
            junction_table = Table(j_ns.table_name, alias=j_alias)
            query_set.query.from_table = junction_table
            query_set.query.join(
                target_table, 
                Eq(
                    Column(rel.foreign_key, target_table), 
                    Column(rel.junction_keys[1], junction_table)
                ),
                join_type
            )
            nested_query = join_as_subquery(query_set.query, rel, self.from_table)    
            nested_query.values[0].where(Eq(Column(rel.junction_keys[0], junction_table), Column(rel.primary_key, self.from_table)))
        else:
            self.query.join(
                query_set.from_table, 
                Eq(
                    Column(rel.primary_key, self.from_table), 
                    Column(rel.foreign_key, query_set.from_table)
                ),
                join_type
            )        
            nested_query = join_as_subquery(query_set.query, rel, self.from_table)    
        nested_query.alias = rel.name
        self.query.columns.append(nested_query)
        # self.query.columns.append(Column("", nested_query, alias=rel.name))
        if not self.query.group_by:
            pk = self.model_class.get_namespace().primary_key.name
            p_id = Column(pk, self.from_table)
            self.query.group_by = [p_id] 
        return self
    
    
    # def recursive(self, field: str):
    def include(self, target: "SelectQuerySet | Type[Model]", alias: str | None = None) -> "SelectQuerySet[MODEL]":
        """
        Include a query set as a column in the current query set.
        The query set will be embedded as a subquery and the result will be nested in the current query set.
        The query set will be embedded as a subquery and the result will be nested in the current query set.
        """
        query_set = self._get_query_set(target)
        rel = self.namespace.get_relation_by_type(query_set.model_class)
        if rel is None:
            raise ValueError("No relation found")
        if query_set.query.ctes:
            query_set = self._merge_ctes(query_set)
        if isinstance(rel, NSManyToManyRelationInfo): 
            j_ns = rel.junction_namespace
            j_alias=self._set_alias(j_ns.table_name)
            junction_table = Table(j_ns.table_name, alias=j_alias)           
            nested_query = Column(
                rel.name,
                NestedSubquery(
                    query_set.query,
                    rel.name,
                    Column(rel.primary_key, self.from_table),
                    Column(rel.foreign_key, query_set.from_table),
                    (Column(rel.junction_keys[0], junction_table), Column(rel.junction_keys[1], junction_table))
                )
            )
        else:
            nested_query = Column(
                rel.name,
                NestedSubquery(
                    query_set.query,
                    rel.name,
                    Column(rel.primary_key, self.from_table),
                    Column(rel.foreign_key, query_set.from_table)
                )
            )
            # if rel.is_one_to_one:
            #     nested_query = query_set.query.as_subquery(self.from_table, rel.primary_key, rel.foreign_key)    
            # else:
            #     nested_query = query_set.query.as_list_subquery(self.from_table, rel.primary_key, rel.foreign_key)    
        nested_query.alias = rel.name
        self.query.columns.append(nested_query)
        return self
    
    def reversed(self, target: "SelectQuerySet | Type[Model]", primary_key: str, foreign_key: str) -> "SelectQuerySet[MODEL]":
        """
        Reverse the relation of the current query set.
        """
        query_set = self._get_query_set(target)        
        self.query.join(
            query_set.from_table,
            Eq(Column(primary_key, self.from_table), Column(foreign_key, query_set.from_table)),
            "LEFT"
        )
        self.query.columns.append(Column("test", self.from_table))
        return self
      
    def from_subquery(self, query_set: "SelectQuerySet"):
        if query_set.ctes:
            query_set = self._merge_ctes(query_set)
        self.query.select(Column("*", query_set.from_table))
        self.query.from_(query_set.as_subquery())
        # self.query.from_table = query_set.query.from_table
        return self
    
    def as_subquery(self, alias: str | None = None):
        return Subquery(self.query, alias=alias or str(self.from_table))
        
    def _add_columns(self, *args):
        if self.query_depth == 1:
            self.query.columns.extend(list(args))
        else:
            args = self.args + args            


    def order_by(self, *fields: str) -> "SelectQuerySet[MODEL]":
        orderings = []
        for field in fields:
            direction = "ASC"
            if field.startswith("-"):
                direction = "DESC"
                field = field[1:]
            orderings.append(OrderBy(Column(field, self.from_table), direction))

        self.query.order_by_(*orderings)
        return self


    def limit(self, n: int) -> "SelectQuerySet[MODEL]":
        self.query.limit_(n)
        return self

    def offset(self, n: int) -> "SelectQuerySet[MODEL]":
        self.query.offset_(n)
        return self
    
    
    def first(self) -> "QuerySetSingleAdapter[MODEL]":
        self.order_by(self.model_class.get_key_field())
        self.limit(1)
        return QuerySetSingleAdapter(self)
    
    def last(self) -> "QuerySetSingleAdapter[MODEL]":
        self.order_by("-" + self.model_class.get_key_field())
        self.limit(1)
        return QuerySetSingleAdapter(self)
    
    def tail(self, n: int) -> "SelectQuerySet[MODEL]":
        if not self.namespace.default_temporal_field:
            raise ValueError("No default temporal field found")
        self.order_by("-" + self.namespace.default_temporal_field.name)
        self.limit(n)
        return self
    
    def head(self, n: int) -> "SelectQuerySet[MODEL]":
        if not self.namespace.default_temporal_field:
            raise ValueError("No default temporal field found")
        self.order_by(self.namespace.default_temporal_field.name)
        self.limit(n)
        return self

    def render(self) -> str:
        compiler = Compiler()
        processor = Preprocessor()
        compiled = processor.process_query(self.query)
        sql, params = compiler.compile(compiled)
        self._params = params
        return sql
    
    
    def parse(self, row: dict[str, Any]) -> MODEL:
        m = self.model_class.from_dict(self.namespace.pack_record(row))
        if self._parse is not None:
            return self._parse(m)
        return m
    
    
    async def execute(self) -> List[MODEL]:
        # sql = self.render()        
        compiler = Compiler()
        processor = Preprocessor()
        compiled = processor.process_query(self.query)
        sql, params = compiler.compile(compiled)
        results = await self.execute_sql(sql, *params)
        return [self.parse(row) for row in results]
        # return [self.model_class(**self.namespace.pack_record(dict(row))) for row in results]



    async def execute_sql(self, sql: str, *params: Any):
        return await PGConnectionManager.fetch(sql, *params)
    
    
