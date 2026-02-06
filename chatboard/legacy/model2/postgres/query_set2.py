









from typing import Any, Callable, Generic, List, Self, Type
from typing_extensions import TypeVar

from chatboard.model.base_namespace import Namespace
from chatboard.model.model import Model
from chatboard.model.postgres.fields_query import PgFieldInfo
from chatboard.model.postgres.query_builders2 import Coalesce, Condition, Filter, JsonAgg, JsonBuild, JsonNestedQuery, JsonSubNestedQuery, Query, SelectQuery, TableField, TableName, Where
from chatboard.model.query_filters import QueryProxy
from chatboard.utils.db_connections import PGConnectionManager


MODEL = TypeVar("MODEL", bound="Model")


    
    
class SelectQuerySet(Generic[MODEL]):
    
    model_class: Type[MODEL]
    query: SelectQuery
    query_stack: list[Query]
    model_stack: list[Model]
    alias_lookup: dict[str, TableName]
    
    
    
    def __init__(self, model_class: Type[MODEL]):
        # self.model_class = model_class
        self.model_class = model_class
        self.alias_lookup = {}
        table_name = TableName(model_class.get_namespace().table_name)
        self._set_alias(table_name)
        self.query = SelectQuery(
            select="*",
            from_table=table_name,
        )
        self.query_stack = [self.query]
        self.model_stack = [model_class]
        
        
        
    @property
    def namespace(self) -> Namespace[MODEL, PgFieldInfo]:
        return self.model_class.get_namespace()
     
    @property
    def curr_query(self):
        return self.query_stack[-1]
        
    @property
    def curr_model(self):
        return self.model_stack[-1]
    
    async def execute_sql(self, sql: str, *values: Any) -> List[Any]:
        return await PGConnectionManager.fetch(sql, *values)
    
    def pack_record(self, data: Any) -> Any:
        """Pack the record for the model"""
        return self.model_class.get_namespace().pack_record(data)

    
    async def execute(self) -> List[MODEL]:
        sql = self.render()
        results = await self.execute_sql(sql)
        return [self.model_class(**self.pack_record(dict(result))) for result in results] 

        
    def select(self, *args: str, **kwargs: str) -> "SelectQuerySet[MODEL]":
        if len(args) and args[0] == "*":
            fields = [f.name for f in self.curr_model.iter_fields()]
        else:    
            fields = list(args) 
            if kwargs:
                fields += [(v, k) for k, v in kwargs.items()]
        self.curr_query._select = fields
        return self
    
    def alias(self, alias: str) -> "SelectQuerySet[MODEL]":
        if alias in self.alias_lookup:
            raise ValueError(f"Alias {alias} already exists")
        self.alias_lookup[alias] = self.curr_query.table_name
        self.curr_query.alias(alias)
        return self
    
    def _set_alias(self, table_name: TableName) -> TableName:
        alias = table_name.table_name[0].lower()
        for i in range(10):
            if alias not in self.alias_lookup:
                break
            alias = f"{table_name.table_name[0].lower()}{i}"
        table_name.set_alias(alias)
        self.alias_lookup[alias] = table_name
        return table_name
    
    def join(self, model: Type[Model]) -> "SelectQuerySet[MODEL]":        
        # print("Joining", model, "with", self.curr_model)
        relation = self.curr_model.get_namespace().get_relation_by_type(model)
        if relation is None:
            raise ValueError(f"Relation not found for model {model}")
        fields = [f.name for f in model.iter_fields()]
        foreign_table = self._set_alias(TableName(model.get_namespace().table_name))
        if len(self.query_stack) == 1:
            query = JsonNestedQuery(
                select=fields,
                from_table=foreign_table,
                primary_key=relation.primary_key,
            )
        else:
            query = JsonSubNestedQuery(
                select=fields,
                from_table=foreign_table,
                where=Where([
                    Condition(
                        TableField(foreign_table, relation.foreign_key), "=" ,TableField(self.curr_query.table_name, relation.primary_key)
                    )],
                    table_name=foreign_table,
                ),
                # primary_key=relation.primary_key,
            )
        
        self.curr_query.add_select((query, relation.name))
        self.query.join(self.curr_query.table_name, relation.primary_key, foreign_table, relation.foreign_key)
        if len(self.query_stack) == 1:
            self.curr_query.group_by(relation.primary_key)
        self.query_stack.append(query)
        self.model_stack.append(model)
        return self
    
    

    def where(self, filter_fn: Callable[[MODEL], bool] | None = None, **kwargs) -> "SelectQuerySet[MODEL]":
        """Filter the query"""           
        if filter_fn is not None:
            proxy = QueryProxy[MODEL, PgFieldInfo](self.curr_model, self.curr_model.namespace)
            self.curr_query.where(filter_fn(proxy))
        return self
    
    def render(self, new_line: bool = True) -> str:
        return self.query.render(new_line)
    
    
