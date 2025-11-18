
from ast import Expression
from functools import reduce
from operator import and_

from typing import Any, Callable, Self
from ..base.base_namespace import BaseNamespace
from ..postgres2.pg_query_set import MODEL, QueryProxy
from ..relation_info import RelationInfo
from ..sql.compiler import Compiler
from ..sql.expressions import Coalesce, Null, Value, Function, Eq, WhereClause, param
from ..sql.joins import Join
from ..sql.queries import Column, SelectQuery, Table
from ..postgres2.pg_field_info import PgFieldInfo









class TableRegistry:
    
    def __init__(self):
        self.alias_lookup = {}
        
    def register(self, name: str, alias: str):
        self.alias_lookup[name] = alias
        
    def gen_alias(self, name: str) -> str:
        base = name[0].lower()
        alias = base
        i = 1
        while alias in self.alias_lookup:
            alias = f"{base}{i}"
            i += 1
        return alias
    
    def get_ns_table(self, namespace: BaseNamespace) -> Table:
        if namespace.name in self.alias_lookup:
            return Table(namespace.name, alias=self.alias_lookup[namespace.name])
        else:
            alias = self.gen_alias(namespace.name)
            self.alias_lookup[namespace.name] = alias
            return Table(namespace.name, alias=alias)
        
        



# Selection(user_ns, ["name"])



     
    

class SelectionNamespace:
    
    def __init__(self, namespace: BaseNamespace, table_registry: TableRegistry):
        self.namespace = namespace
        self.from_table = table_registry.get_ns_table(namespace)
        self._table_registry = table_registry
        self.filter = WhereClause()
        self.order_by = []
        self.limit = None
        self.offset = None
        self.relation = None
        self.group_by = []
        self.joins = []

            
        # return self
    def as_relation(self, parent_proj: "ProjectionNamespace", relation: RelationInfo):
        self.relation = relation
        if relation.is_many_to_many:
            junction_ns = relation.relation_model.get_namespace()
            jt = self._table_registry.get_ns_table(junction_ns)
            self.joins.append(Join(jt, Eq(Column(relation.foreign_key, self.from_table), Column(relation.junction_keys[1], jt))))            
            self.filter &= Eq(Column(relation.primary_key, parent_proj.from_table), Column(relation.junction_keys[0], jt))
        else:
            self.filter &= Eq(Column(relation.foreign_key, self.from_table), Column(relation.primary_key, parent_proj.from_table))
    
        
    def where(
        self,
        condition: Callable[[MODEL], bool] | None = None,
        # condition: Callable[[QueryProxy], Expression] | Expression | None = None,
        **kwargs: Any
    ) -> Self:
        """
        Add a WHERE clause to the query.
        condition: callable taking a QueryProxy or direct Expression
        kwargs: field=value pairs, ANDed together
        """
        expressions = []

        # Callable condition: lambda m: m.id > 5
        if condition is not None:
            if callable(condition):
                proxy = QueryProxy(self.namespace._model_cls, self.from_table)
                expr = condition(proxy)
            else:
                expr = condition  # Already an Expression
            expressions.append(expr)

        # kwargs: field=value
        for field, value in kwargs.items():
            col = Column(field, self.from_table)
            expressions.append(Eq(col, param(value)))

        # Merge with AND if multiple
        if expressions:
            expr = reduce(and_, expressions)
            self.filter &= expr

        return self
    
    def group(self, *fields: str) -> Self:
        self.group_by = [Column(field, self.from_table) for field in fields]
        return self
    
    def group_by_key(self):
        return self.group(self.namespace.primary_key)
    
    def build_query(self):
        query = SelectQuery().from_(self.from_table)
        query.where = self.filter
        query.group_by = self.group_by
        query.order_by = self.order_by
        query.limit = self.limit
        query.offset = self.offset
        query.joins = self.joins
        return query


class ProjValue:
    def __init__(self, value: PgFieldInfo | RelationInfo, alias: str | None = None):
        self._value = value
        self.alias = alias
        self.is_relation = isinstance(value, RelationInfo)
        
    def resolve(self, table: Table):
        return Column(self._value.name, table)
        


    

class ProjectionNamespace:
    
    def __init__(self, selection: SelectionNamespace, table_registry: TableRegistry):
        self._selection = selection
        self._table_registry = table_registry
        self.columns: list[Column] = []
        
    
    @property
    def fields(self):
        return self._selection.namespace._fields 
    
    @property
    def relations(self):
        return self._selection.namespace._relations
    
    @property
    def from_table(self):
        return self._selection.from_table
    
    @property
    def namespace(self):
        return self._selection.namespace
    
    @property
    def filter(self):
        return self._selection.filter
            
    def project_all(self):
        for field_name, field in self.fields.items():
            self.project(field_name)
        return self
    
    def project(self, target: "str | ProjectionNamespace", alias: str | None = None):
        if isinstance(target, str):
            field = self.fields.get(target, None)
            if field is not None:
                col = Column(field.name, self.from_table)
                self.columns.append(col)
                return self
            relation = self.relations.get(target, None)
            if relation is not None:
                rel_sel = SelectionNamespace(relation.foreign_namespace, self._table_registry)
                rel_sel.as_relation(self, relation)
                proj_ns = ProjectionNamespace(rel_sel, self._table_registry).project_all()
                col = Column(name='', table=proj_ns, alias=target)
                self.columns.append(col)
                return self
        elif isinstance(target, ProjectionNamespace):
            relation = self.namespace.get_relation_for_namespace(target.namespace)
            if not relation:
                raise ValueError(f"Relation {target.namespace.name} not found in {self.namespace.name}")
            target._selection.as_relation(self, relation)
            col = Column(name='', table=target, alias=relation.name)
            self.columns.append(col)
            return self
            
        raise ValueError(f"Field {field} not found in {self._selection.namespace.name}")
    
    
        
    
    def to_json(self):        
        query = self._selection.build_query()
        json_pairs = []
        for col in self.columns:
            if isinstance(col.table, Table):
                json_pairs.append(Value(col.alias or col.name))
                json_pairs.append(col)
            elif isinstance(col.table, ProjectionNamespace):
                col.table = col.table.to_json()
                json_pairs.append(Value(col.alias or col.name))
                col.alias = ''
                json_pairs.append(col)
            else:
                raise ValueError("ProjectionNamespace contains non-Column objects")
        json_obj = Function("jsonb_build_object", *json_pairs)
        if self._selection.relation and not self._selection.relation.is_one_to_one:
            json_obj = Function("json_agg", json_obj)
            default_value = Value("[]", inline=True)
        else:
            default_value = Null()
        query.select(json_obj)
        return Coalesce(query, default_value)
            
    
    
    
    def to_query(self):
        # query = SelectQuery().from_(self.from_table)
        query = self._selection.build_query()
                   
        for col in self.columns:
            if isinstance(col.table, Table):
                query.columns.append(col)
            elif isinstance(col.table, ProjectionNamespace):
                col.table = col.table.to_json()
                query.columns.append(col)
                query.group_by = [Column(self._selection.namespace.primary_key, self.from_table)]
        
        return query



class JoinedNamespace(BaseNamespace):
    
    def __init__(self, left: BaseNamespace, right: BaseNamespace):
        self.left = left
        self.right = right
        self.relation = self.left.get_relation_for_namespace(right)
        
        
    def process_relation(self, left_table: Table, right_table: Table):
        if self.relation.is_many_to_many:
            raise ValueError("Many-to-many relations are not supported yet")
        else:
            return Column(self.relation.name, left_table)

    def __getattr__(self, item):
        return getattr(self.left, item)