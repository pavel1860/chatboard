
import textwrap
from typing import Any, Iterator, List, Optional, Self, Set, TYPE_CHECKING



if TYPE_CHECKING:
    from promptview.model.postgres.namespace import PostgresNamespace
    from promptview.model.postgres.fields_query import PgFieldInfo


# def build_with_clause(queries: list[str]) -> str:
#     return "WITH " + ", ".join(queries) + " AS "




class SqlQuery:
    
    def __init__(
        self, 
        target: "PostgresNamespace | SqlQuery | SqlContainer",
        alias: str | None = None,
        inputs: Optional[Set[str]] = None,
        select: Optional[Set[str]] = None,
        labels: Optional[List[dict[str, str]]] = None,
        overrides: Optional[dict[str, Any]] = None,
    ):
        self.target = target
        self.alias = alias
        # self._all_inputs = not inputs
        # self._select_all = not select
        self._inputs = inputs
        self._select = select
        self._labels = labels or {}
        self._start_idx = 1
        self._overrides = overrides or {}
        
        
    def iter_fields(self, keys: bool = True, select: Set[str] | None = None) -> "Iterator[PgFieldInfo]":
        return self.target.iter_fields(keys, select)
        # for field in self.namespace.iter_fields():
        #     if not keys and field.is_key:
        #         continue
        #     if select and field.name not in select:
        #         continue
        #     yield field
            
    def inputs_len(self) -> int:
        return len(self.inputs) - len(self._overrides)
    
    def outputs_len(self) -> int:
        return len(self.outputs)
                
    def render_select_fields(self, select: Set[str] | None = None, alias: str | None = None, labels: dict[str, str] = {}, new_line: bool = True) -> str:
        select = select or self._select
        alias = alias or self.alias
        labels = labels or self._labels     
        if not select and not labels:
            return "*"
        delimiter = ", \n" if new_line else ", "
        fields_sql = []
        for field in self.iter_fields(select=select):
            field_sql = field.name
            if field.name in labels:
                field_sql = f"{field.name} AS {labels[field.name]}"            
            fields_sql.append(field_sql)
        return delimiter.join(fields_sql)
    
    def render_set_fields(self, inputs: Set[str] | None = None, alias: str | None = None, labels: dict[str, str] = {}, new_line: bool = True) -> str:
        alias = alias or self.alias
        overrides = self._overrides
        inputs = inputs or self._inputs
        delimiter = ", \n" if new_line else ", "
        fields_sql = []
        for field in self.iter_fields(keys=False, select=inputs):
            field_sql = field.name
            if field.name in overrides:
                field_sql = f"{field.name} = {overrides[field.name]}"            
            fields_sql.append(field_sql)
        return delimiter.join(fields_sql)
    
    
    def render_insert_fields(self, inputs: Set[str] | None = None, alias: str | None = None) -> str:        
        alias = alias or self.alias
        inputs = inputs or self._inputs
        fields_sql = [] 
        for field in self.iter_fields(keys=False, select=inputs):
            field_sql = field.name
            fields_sql.append(field_sql)
        return ", ".join(fields_sql)
    
    def render_placeholders(self, inputs: Set[str] | None = None, start_idx: int = 1) -> str:
        idx = start_idx
        placeholders = []
        inputs = inputs or self._inputs
        all_inputs = inputs is None
        for field in self.iter_fields(keys=False, select=inputs):
            if all_inputs or field.name in inputs:
                if not self._overrides.get(field.name):
                    placeholders.append(f'${idx}')
                    idx += 1
                else:
                    placeholders.append(self._overrides[field.name])
        return ", ".join(placeholders)
    
    def override(self, key: str, value: Any):
        self._overrides[key] = value
        
    def set_inputs(self, inputs: Set[str]):
        self._inputs = inputs
    

    
    @property
    def table_name(self) -> str:
        return self.target.table_name
    
    @property
    def primary_key(self) -> str:
        return self.target.primary_key.name
    
    @property
    def inputs(self) -> Set[str]:
        return self._inputs or set()
    
    @property
    def outputs(self) -> Set[str]:
        return self._select or set()
    
    def render(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def indent(self, text: str) -> str:
        return textwrap.indent(text, "    ")
    
    


class SqlContainer:
    
    def __init__(self, queries: list[SqlQuery]):
        self.queries = queries
        prev_len = 1
        for q in self.queries:
            q._start_idx = prev_len
            prev_len += q.inputs_len()
            
    @property
    def inputs(self) -> list[Any]:
        return self.queries[-1].inputs
    
    @property
    def inputs_len(self) -> int:
        return self.queries[-1].inputs_len()
    
    @property
    def outputs_len(self) -> int:
        return self.queries[-1].outputs_len()
    
    @property
    def outputs(self) -> list[Any]:
        return self.queries[-1].outputs
    
    
    @property
    def table_name(self) -> str:
        return self.queries[-1].alias or self.queries[-1].table_name
    
    @property
    def primary_key(self) -> str:
        return self.queries[-1].primary_key
    
    
    def render(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
        
        
        

class InsertQuery(SqlQuery):
            
    def render(self) -> str:
        sql = (
            f"INSERT INTO {self.table_name} ({self.render_insert_fields()})\n"
            f"VALUES ({self.render_placeholders(start_idx=self._start_idx)})\n"
            f"RETURNING {self.render_select_fields(new_line=False)}\n"
        )
        return sql
    
    

class UpdateQuery(SqlQuery):
    
    def render(self) -> str:
        sql = (
            f"UPDATE {self.table_name}\n"
            f"SET\n{self.indent(self.render_set_fields())}\n"
            f"WHERE {self.primary_key} = $1\n"
            f"RETURNING {self.render_select_fields(new_line=False)}\n"
        )
        return sql
    

    
    
class WithSubQuery(SqlContainer):
    
    
    
    def render(self) -> str:
        sql = "WITH "
        for idx, query in enumerate(self.queries):
            alias = query.alias or f"subquery_{idx}"
            sql += alias + " AS (\n" 
            sql += textwrap.indent(query.render(), "    ")
            sql += ")"
            sql += ", \n" if idx < len(self.queries) - 1 else "\n"
        return sql
    
    
    
class SelectQuery(SqlQuery):
    
    def __init__(
        self, 
        target: "PostgresNamespace | SqlQuery | SqlContainer", 
        alias: str | None = None, 
        inputs: Optional[Set[str]] = None, 
        select: Optional[Set[str]] = None, 
        labels: Optional[List[dict[str, str]]] = None, 
        overrides: Optional[dict[str, Any]] = None
    ):
        from promptview.model.postgres.namespace import PostgresNamespace
        super().__init__(target, alias, inputs, select, labels, overrides)
        self._from_table = None
        self._from_query = None
        
        
    def from_table(self, table_name: str) -> Self:
        self._from_table = table_name
        return self
    
    def from_query(self, query: SqlQuery) -> Self:
        self._from_query = query
        return self
    
    def render(self) -> str:
        sql = ""
        if isinstance(self.target, SqlContainer) or isinstance(self.target, SqlQuery):
            sql += self.target.render() + "\n"
        sql += f"SELECT \n"
        sql += textwrap.indent(self.render_select_fields(), '    ')
        sql += f"\nFROM {self.table_name}"
        if self.alias:
            sql += f"\nAS {self.alias}"
        sql += "\n"
        return sql
    
    
    
    
    