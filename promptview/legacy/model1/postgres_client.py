from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timezone
import datetime as dt
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, TypedDict, Type, get_args, get_origin
from uuid import uuid4
import asyncpg
import os
import itertools
import json
import numpy as np
from pydantic import BaseModel

from promptview.artifact_log.artifact_log3 import ArtifactLog
from promptview.utils.model_utils import get_list_type, is_list_type


if TYPE_CHECKING:
    from promptview.model.model import Model

from .fields import VectorSpaceMetrics


if TYPE_CHECKING:
    from .resource_manager import VectorSpace, NamespaceManager, NamespaceParams
from .query import QueryFilter, QueryProxy, FieldComparable, FieldOp, QueryOp, QueryProxyAny, QuerySet


def unpack_list_model(pydantic_model):
    return get_args(pydantic_model)[0]

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


class OrderBy(TypedDict):
    key: str
    direction: Literal["asc", "desc"]
    start_from: int | float | datetime







def shorten_vector_strings(text: str):
    import re
    pattern = re.compile(r'\[\s*(?P<first>[-+]?\d*\.\d{3})[^\]]*?(?P<last>\d{5})\s*\]')

    def shorten_vector_aux(match):
        # Retrieve the captured groups.
        first = match.group('first')
        last = match.group('last')
        # Build the new string in the desired format.
        return f'[{first} ... {last}]'

    result = pattern.sub(shorten_vector_aux, text)
    return result



def stringify_vector(vector: list[float] | np.ndarray):
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return '[' + ', '.join(map(str, vector)) + ']'







FieldTypes = Literal["field", "vector", "foreign_key", "relation", "key"]

@dataclass
class SqlFieldBase:
    _:KW_ONLY    
    name: str
    index: Literal["btree", "gist", "hash"] | None = field(default=None)
    index_extra: dict[str, Any] | None = field(default=None)
    default: str | None = field(default=None)
    is_optional: bool = field(default=False)
    type: FieldTypes = field(default="field")
    
    def render_placeholder(self, idx: int) -> str:
        raise NotImplementedError("Placeholder rendering is not supported for this field type")
    
    @abstractmethod
    def render_insert_value(self, value: Any) -> str:
        raise NotImplementedError(f"Insert statements are not supported for field {self.name} of type {self.type}")
    
    @abstractmethod
    def render_create(self) -> str:
        raise NotImplementedError(f"Create statements are not supported for field {self.name} of type {self.type}")
    
    def render_augment(self) -> str:
        raise NotImplementedError(f"Augment statements are not supported for field {self.name} of type {self.type}")
    
    def render_create_index(self, table: str) -> str:
        return f"""
CREATE INDEX IF NOT EXISTS {self.name}_index ON {table} USING {self.index} ("{self.name}");
"""

    def unpack_field(self, value: Any) -> Any:
        return value
        


@dataclass
class SqlKeyField(SqlFieldBase):
    _:KW_ONLY  
    key_type: str
    is_primary_key: bool = True    
    type: FieldTypes = field(default="key")
    
    def render_create(self) -> str:
        if self.is_primary_key:
            return f'"{self.name}" {self.key_type} PRIMARY KEY'
        else:
            return f'"{self.name}" {self.key_type}'

@dataclass
class SqlFieldType(SqlFieldBase):
    _:KW_ONLY  
    sql_type: str 
    type: FieldTypes = field(default="field")    
    
    def render_create(self) -> str:
        sql = f'"{self.name}" {self.sql_type}'
        if self.default:
            sql += f" DEFAULT {self.default}"
        elif not self.is_optional:
            sql += " NOT NULL"
        return sql
    
    
    def render_placeholder(self, idx: int) -> str:
        if self.sql_type == "JSONB":
            return f"${idx}::JSONB"
        return f"${idx}"
    
    def render_insert_value(self, value: Any | None = None) -> Any:
        # if value is None:
        #     if self.default:
        #         return self.default
        #     elif self.is_optional:
        #         return "NULL"
        #     else:
        #         raise ValueError(f"Value is None and no default is set for field {self.name}")
            
        if self.sql_type == "JSONB":
            return json.dumps(value)
        elif isinstance(value, Enum):
            return value.value
        else:
            return value

    def unpack_field(self, value: Any) -> Any:
        if self.sql_type == "JSONB":
            return json.loads(value)
        else:
            return value

@dataclass
class SqlVectorField(SqlFieldBase):
    _:KW_ONLY  
    size: int
    type: FieldTypes = field(default="vector")
    
    def render_placeholder(self, idx: int) -> str:
        return f"${idx}::vector"
    
    
    def render_create(self) -> str:
        return f'"{self.name}" vector({self.size})'
    
    def render_insert_value(self, value: Any) -> str:
        return stringify_vector(value)
    
    def render_create_index(self, table: str) -> str:
        if self.index is None:
            return ""
        sql = f"""CREATE INDEX IF NOT EXISTS {self.name}_index ON {table} USING ivfflat ({self.name} vector_cosine_ops)"""
        if self.index_extra:
            sql += f" WITH (lists = {self.index_extra.get('lists', 100)})"
        return sql
    
    def unpack_field(self, value: Any) -> Any:
        return json.loads(value)

@dataclass
class SqlForeignKeyField(SqlFieldBase):
    _:KW_ONLY  
    table: str
    foreign_table: str    
    foreign_key: str = "id"
    type: FieldTypes = field(default="foreign_key")
    
    def render_placeholder(self, idx: int) -> str:
        return f"${idx}"
    
    def render_insert_value(self, value: Any) -> str:
        return value
    
    def render_create(self) -> str:
        return f"{self.name} INT,\nFOREIGN KEY ({self.name}) REFERENCES {self.foreign_table} ({self.foreign_key})"
        
@dataclass
class RelationField(SqlFieldBase):
    _:KW_ONLY  
    table: str
    foreign_table: str    
    foreign_key: str = "id"   
    type: FieldTypes = field(default="relation")
    
    def render_placeholder(self, idx: int) -> str:
        return f"${idx}"
    
    def render_create(self) -> str:
        raise NotImplementedError("Relation field CREATE is not supported in CREATE TABLE statements")
    
    def render_insert_value(self, value: Any) -> str:
        return value
    
    def render_augment(self) -> str:
        return f"""
DO $$
BEGIN
    -- Check if the "test_case_id" column exists; if not, add it.
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = '{self.table}'
        AND column_name = '{self.name}'
    ) THEN
        EXECUTE 'ALTER TABLE {self.table} ADD COLUMN "{self.name}" INT';
    END IF;

    -- Check if the foreign key constraint exists; if not, add it.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_{self.name}'
        AND conrelid = '{self.table}'::regclass
    ) THEN
        EXECUTE '
        ALTER TABLE {self.table} 
        ADD CONSTRAINT fk_{self.name} 
        FOREIGN KEY ("{self.name}") 
        REFERENCES {self.foreign_table}("{self.foreign_key}")
        ';
    END IF;
END $$;
"""
    





class FieldMapper:
    table_name: str
    model_cls: "Type[Model]"
    field_lookup: dict[str, SqlFieldBase]
    
    def __init__(self, table_name: str, model_cls: "Type[Model]"):
        self.table_name = table_name
        self.model_cls = model_cls
        self.field_lookup= {}
        
        
    def add_field(self, sql_field_type: SqlFieldBase):
        self.field_lookup[sql_field_type.name] = sql_field_type
        
    def set_index(self, field_name: str, index_type: str):
        self.field_lookup[field_name].index_type = index_type
        
            
    def get_field(self, field_name: str) -> SqlFieldBase:
        return self.field_lookup[field_name]
    
    
    def get_field_sql(self, field_name: str) -> str:
        pass
    
    def iter_fields(self, values: dict[str, Any] | None = None, exclude_types: list[FieldTypes] = [], exclude_fields: list[str] = [], has_index: bool | None = None):
        exclude_lookup = {field_type: True for field_type in exclude_types}  
        exclude_fields_lookup = {field_name: True for field_name in exclude_fields}
        for field in self.field_lookup.values():
            if field.name in exclude_fields_lookup:
                continue
            if field.type in exclude_lookup:
                continue
            if has_index is not None:
                if has_index == False and field.index is not None:
                    continue
                if has_index == True and field.index is None:
                    continue
            if values and field.name not in values:
                if field.is_optional or field.default:
                    continue
                else:
                    raise ValueError(f"Value is None and no default is set for field {field.name}")
            yield field
            
           
    def render_create(self, exclude_types: list[FieldTypes] = []) -> str:
        fields = "".join([f"{field.render_create()},\n" for field in self.iter_fields(exclude_types=exclude_types)])
        if fields:
            fields = fields.rstrip(",\n")
        return f"""
CREATE TABLE IF NOT EXISTS {self.table_name} (   
    {fields}
);
"""

    def render_augment(self, exclude_types: list[FieldTypes] = []) -> str:
        return "".join([f"{field.render_augment()}\n" for field in self.iter_fields(exclude_types=exclude_types)])
     

    def render_create_indices(self) -> str:
        return "".join([f"{field.render_create_index(self.table_name)}" for field in self.iter_fields() if field.index is not None])
    
    def render_insert_fields(self, values: dict[str, Any], exclude_types: list[FieldTypes] = [], exclude_fields: list[str] = []) -> str:
        # return ', '.join([f'"{k}"' for k in values.keys()])
        return 'INSERT INTO ' + self.table_name + ' (' + ', '.join([f'"{field.name}"' for field in self.iter_fields(values=values, exclude_types=exclude_types, exclude_fields=exclude_fields)]) + ')'
    
    # def render_placeholders(self, values: dict[str, Any], start_idx: int = 1) -> str:
    #     return '(' + ', '.join([self.field_lookup[k].render_placeholder(i + start_idx) for i, k in enumerate(values.keys())]) + ')'
    
        
        
    
    def render_placeholders(self, items: list[dict[str, Any]], exclude_types: list[FieldTypes] = [], exclude_fields: list[str] = []) -> str:
        ph_list = []
        step = len(items[0].keys())
        for idx in range(len(items)):
            _field_iter = self.iter_fields(items[idx], exclude_types, exclude_fields)
            ph_list.append('(' + ', '.join([field.render_placeholder(1 + i + idx * step) for i, field in enumerate(_field_iter)]) + ')')
        return "VALUES\n" + ',\n'.join(ph_list)
    
    def render_insert_values(self, values: dict[str, Any], exclude_types: list[FieldTypes] = [], exclude_fields: list[str] = []) -> list[Any]:
        _field_iter = self.iter_fields(values, exclude_types, exclude_fields)
        return [field.render_insert_value(values.get(field.name, None)) for field in _field_iter]
        
    def unpack_record(self, record: Any):
        return {k: self.field_lookup[k].unpack_field(v) for k, v in record.items()}
            
    
            
            
            
            
            
            
            





class PostgresClient:
    def __init__(self, url=None, user=None, password=None, database=None, host=None, port=None):
        self.url = url or os.environ.get("POSTGRES_URL")
        if not self.url:
            self.user = user or os.environ.get("POSTGRES_USER", "postgres")
            self.password = password or os.environ.get("POSTGRES_PASSWORD", "postgres")
            self.database = database or os.environ.get("POSTGRES_DB", "postgres")
            self.host = host or os.environ.get("POSTGRES_HOST", "localhost")
            self.port = port or os.environ.get("POSTGRES_PORT", 5432)
        if not self.url and not (self.user and self.password and self.database and self.host and self.port):
            raise ValueError("Either url or user, password, database, host, and port must be provided")
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Initialize the connection pool if not already initialized."""
        if self.pool is None:
            if self.url:
                self.pool = await asyncpg.create_pool(self.url)
            else:
                self.pool = await asyncpg.create_pool(
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    host=self.host,
                    port=self.port
                )

    async def close(self):
        """Close the connection pool if it exists."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _ensure_connected(self):
        """Ensure we have a connection pool."""
        if self.pool is None:
            await self.connect()
        assert self.pool is not None
    
    async def init_extensions(self):
        await self._ensure_connected()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            # Enable pgvector extension if not enabled
            return await conn.execute('''
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            ''')
            
            
    def build_field_mapper(
        self, 
        namespace_manager: "NamespaceManager",
        namespace: "NamespaceParams",
        # collection_name: str, 
        # model_cls: "Type[Model]", 
        # vector_spaces: list["VectorSpace"], 
        # indices: list[dict[str, str]] | None = None,
        # versioned: bool = False,
        # is_head: bool = False,
        
    ):
        """Create a table for vector storage with pgvector extension."""

        table_name = namespace.table_name
        
        if namespace.versioned and namespace.is_head:
            raise ValueError("Versioned and head models cannot be created at the same time")

        
        # Get model class from the collection name to inspect fields
        field_mapper = FieldMapper(table_name, namespace.model_cls)
        # Create table with proper columns for each field
        field_mapper.add_field(SqlKeyField(name="id", key_type="SERIAL", is_primary_key=True))
        field_mapper.add_field(SqlFieldType(name="created_at", sql_type="TIMESTAMP", index="btree", default="NOW()"))
        field_mapper.add_field(SqlFieldType(name="updated_at", sql_type="TIMESTAMP", index="btree", default="NOW()"))                
                
        if namespace.versioned:
            field_mapper.add_field(SqlForeignKeyField(name="turn_id", table=table_name, foreign_table="turns", foreign_key="id", index="btree"))
            field_mapper.add_field(SqlForeignKeyField(name="branch_id", table=table_name, foreign_table="branches", foreign_key="id", index="btree"))
        if namespace.is_head:
            field_mapper.add_field(SqlForeignKeyField(name="head_id", table=table_name, foreign_table="heads", foreign_key="id", index="btree"))
            # create_table_sql += """
            # head_id INTEGER,
            # FOREIGN KEY (head_id) REFERENCES heads(id),
            # """

        
        
        index_lookup = {index['field']: index for index in namespace.indices} if namespace.indices else {}
        
        # Add columns for each model field
        for field_name, field in namespace.model_fields():            
            if field_name in ["id", "turn_id", "branch_id", "head_id", "_subspace", "score", "created_at", "updated_at"]:
                continue
            # create_table_sql += "\n"
            field_type = field.annotation
            
            
            if field.json_schema_extra.get('is_relation'):
                continue
            is_optional = False
            if type(None) in get_args(field_type):
                is_optional = True
            
            index = "btree" if index_lookup.get(field_name) else None
            
            if is_list_type(field_type):
                list_type = get_list_type(field_type)
                if isinstance(list_type, int):
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="INTEGER[]", index=index, is_optional=is_optional))
                elif isinstance(list_type, float):
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="FLOAT[]", index=index, is_optional=is_optional))
                elif isinstance(list_type, str):
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="TEXT[]", index=index, is_optional=is_optional))
                # elif isinstance(list_type, Model):
                #     partition = field.json_schema_extra.get("partition")
                #     create_table_sql += f'"{field_name}" UUID FOREIGN KEY REFERENCES {model_to_table_name(field_type)} ("{partition}")'
                else:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="JSONB", is_optional=is_optional))
            else:
                if field.json_schema_extra.get("db_type"):
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type=field.json_schema_extra.get("db_type"), index=index, is_optional=is_optional))
                elif field_type == bool:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="BOOLEAN", index=index, is_optional=is_optional))
                elif field_type == int:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="INTEGER", index=index, is_optional=is_optional))
                elif field_type == float:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="FLOAT", index=index, is_optional=is_optional))
                elif field_type == str:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="TEXT", index=index, is_optional=is_optional))
                elif field_type == datetime or field_type == dt.datetime:
                    # TODO: sql_type = "TIMESTAMP WITH TIME ZONE"
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="TIMESTAMP", index=index, is_optional=is_optional))
                elif isinstance(field_type, dict) or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="JSONB", index=index, is_optional=is_optional))
                else:
                    field_mapper.add_field(SqlFieldType(name=field_name, sql_type="TEXT", index=index, is_optional=is_optional))  # Default to TEXT for unknown types
            

        # Add vector columns for each vector space
        for vs in namespace.vector_spaces.values():
            if vs.vectorizer.type == "dense":
                index = "gist" if index_lookup.get(vs.name) else None
                field_mapper.add_field(SqlVectorField(name=vs.name, size=vs.vectorizer.size, index=index, index_extra={"lists": 100}, is_optional=False))
                
                
        for relation in namespace_manager.get_relations(table_name):
            fk_name = f'fk_{relation.key}'
            index = "btree" if index_lookup.get(relation.key) else None              
            ref_ns = namespace_manager.get_namespace(relation.get_target_namespace())
            ref_ns.add_foreign_key(relation.referenced_column, relation.referenced_table, "id")            
            field_mapper.add_field(RelationField(name=relation.key, table=table_name, foreign_table=relation.referenced_table, foreign_key="id", index=index, is_optional=False))
            # field_mapper.add_field(SqlForeignKeyField(name=relation.key, table=table_name, foreign_table=relation.referenced_table, foreign_key="id", index=index, is_optional=False))
    
        # for relation in namespace_manager.get_relations(table_name):
        #     source_table_name = relation["source_namespace"]
        #     column_name = relation["partition"]
        #     fk_name = f'fk_{column_name}'
        #     index = "btree" if index_lookup.get(column_name) else None              
        #     ref_ns = namespace_manager.get_namespace(source_table_name)
        #     ref_ns.add_foreign_key(column_name, table_name, "id")            
        #     # field_mapper.add_field(RelationField(name=column_name, table=table_name, foreign_table=source_table_name, foreign_key="id", index=index, is_optional=False))
        #     field_mapper.add_field(SqlForeignKeyField(name=column_name, table=table_name, foreign_table=source_table_name, foreign_key="id", index=index, is_optional=False))
        

        return field_mapper

            
    def build_table_sql2(
        self, 
        collection_name: str, 
        model_cls: "Type[Model]", 
        vector_spaces: list["VectorSpace"], 
        indices: list[dict[str, str]] | None = None,
        versioned: bool = False,
        is_head: bool = False,
        relations: dict[str, dict[str, str]] | None = None,
    ):
        """Create a table for vector storage with pgvector extension."""

        table_name = camel_to_snake(collection_name)
        
        if versioned and is_head:
            raise ValueError("Versioned and head models cannot be created at the same time")

        
        # Get model class from the collection name to inspect fields
        
        # Create table with proper columns for each field
#         create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
# id UUID PRIMARY KEY DEFAULT uuid_generate_v4()"""
        create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),            
            """
        if versioned:
            create_table_sql += """
            turn_id INT NOT NULL,
            FOREIGN KEY (turn_id) REFERENCES turns(id),
            branch_id INT NOT NULL,
            FOREIGN KEY (branch_id) REFERENCES branches(id),
            """
        if is_head:
            # create_table_sql += """
            # head_id INT NOT NULL,
            # FOREIGN KEY (head_id) REFERENCES heads(id),
            # """
            create_table_sql += """
            head_id INTEGER,
            FOREIGN KEY (head_id) REFERENCES heads(id),
            """

        field_lookup = {}
        
        # Add columns for each model field
        for field_name, field in model_cls.model_fields.items():            
            if field_name in ["id", "turn_id", "branch_id", "head_id", "_subspace", "score", "created_at", "updated_at"]:
                continue
            create_table_sql += "\n"
            field_type = field.annotation
            
            if is_list_type(field_type):
                list_type = get_list_type(field_type)
                if isinstance(list_type, int):
                    sql_type = "INTEGER[]"
                elif isinstance(list_type, float):
                    sql_type = "FLOAT[]"
                elif isinstance(list_type, str):
                    sql_type = "TEXT[]"
                # elif isinstance(list_type, Model):
                #     partition = field.json_schema_extra.get("partition")
                #     create_table_sql += f'"{field_name}" UUID FOREIGN KEY REFERENCES {model_to_table_name(field_type)} ("{partition}")'
                else:
                    sql_type = "JSONB"
            else:
                if field.json_schema_extra.get("db_type"):
                    sql_type = field.json_schema_extra.get("db_type")                
                elif field_type == bool:
                    sql_type = "BOOLEAN"
                elif field_type == int:
                    sql_type = "INTEGER"
                elif field_type == float:
                    sql_type = "FLOAT"
                elif field_type == str:
                    sql_type = "TEXT"
                elif field_type == datetime or field_type == dt.datetime:
                    # TODO: sql_type = "TIMESTAMP WITH TIME ZONE"
                    sql_type = "TIMESTAMP"
                # elif str(field_type).startswith("list["):
                #     # Handle arrays
                #     inner_type = str(field_type).split("[")[1].rstrip("]")
                #     if inner_type == "int":
                #         sql_type = "INTEGER[]"
                #     else:
                #         sql_type = "TEXT[]"
                # elif str(field_type).startswith("dict") or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
                #     sql_type = "JSONB"                    
                    
                elif isinstance(field_type, dict) or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
                    sql_type = "JSONB"
                else:
                    sql_type = "TEXT"  # Default to TEXT for unknown types
            
                create_table_sql += f'"{field_name}" {sql_type},'

        # Add vector columns for each vector space
        for vs in vector_spaces:
            if vs.vectorizer.type == "dense":
                create_table_sql += f'"{vs.name}" vector({vs.vectorizer.size}),\n'
        
        relations_sql = []
            
        for relation in relations.get(table_name, []):
            # create_table_sql += f'\nFOREIGN KEY ("{relation["partition"]}") REFERENCES {relation["source_namespace"]} ("id")'
            # create_table_sql += f'\n"{relation["partition"]}" INT REFERENCES {relation["source_namespace"]}("id")'
            # relations_sql.append(f"""
            # ALTER TABLE {table_name}
            # ADD COLUMN "{relation["partition"]}" INT;
                                 
            # ALTER TABLE {table_name}
            # ADD CONSTRAINT fk_{relation["partition"]}
            # FOREIGN KEY ("{relation["partition"]}")
            # REFERENCES {relation["source_namespace"]} ("id");
            # """)
            source_table_name = relation["source_namespace"]
            column_name = relation["partition"]
            fk_name = f'fk_{column_name}'
            relations_sql.append(f"""
            DO $$
            BEGIN
                -- Check if the "test_case_id" column exists; if not, add it.
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = '{table_name}'
                    AND column_name = '{column_name}'
                ) THEN
                    EXECUTE 'ALTER TABLE {table_name} ADD COLUMN "{column_name}" INT';
                END IF;

                -- Check if the foreign key constraint exists; if not, add it.
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = '{fk_name}'
                    AND conrelid = '{table_name}'::regclass
                ) THEN
                    EXECUTE '
                    ALTER TABLE {table_name} 
                    ADD CONSTRAINT {fk_name} 
                    FOREIGN KEY ("{column_name}") 
                    REFERENCES {source_table_name}("id")
                    ';
                END IF;
            END $$;

            """)
                
        create_table_sql = create_table_sql.rstrip(",")
        create_table_sql += "\n);"
        
        
        indices_sql = []

        # Create indices
        if indices:
            for index in indices:
                field = index['field']
                # Create GiST index for vector columns
                if field in [vs.name for vs in vector_spaces]:
                    indices_sql.append(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_{field}_idx 
                    ON {table_name} 
                    USING ivfflat ("{field}" vector_cosine_ops)
                    WITH (lists = 100);
                    """)
                else:
                    # Create B-tree index for regular columns
                    indices_sql.append(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_{field}_idx 
                    ON {table_name} 
                    USING btree ("{field}");
                    """)
        # return create_table_sql + "\n" + "\n".join(indices_sql)
        return create_table_sql, "\n".join(indices_sql), "\n".join(relations_sql)

    async def execute_sql(self, sql: str):
        
        await self._ensure_connected()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            try:
                return await conn.execute(sql)
            except Exception as e:
                print(e)
                print(shorten_vector_strings(sql))
                raise e

    async def upsert(
        self,
        namespace: str,
        vectors: List[dict[str, List[float] | Any]],
        metadata: List[Dict],
        model_cls: Type[BaseModel],
        batch_size=100,
        is_versioned: bool = False,
        is_head: bool = False,
        is_detached_head: bool = False,
        field_mapper: FieldMapper | None = None,
    ):
        """Upsert vectors and metadata into the collection with fixed turn_id and type fields."""
        await self._ensure_connected()
        assert self.pool is not None
        if is_versioned and is_head:
            raise ValueError("Versioned and head models cannot be created at the same time")
        artifact_values = {}
        if field_mapper is None:
            raise ValueError("field_mapper is required")
        # if is_versioned:
        #     artifact_log = ArtifactLog.get_current()
        #     assert artifact_log.is_initialized
        #     artifact_values = {"turn_id": artifact_log.head["turn_id"], "branch_id": artifact_log.head["branch_id"]}
        # if is_head:
        #     artifact_log = ArtifactLog()
        #     head = await artifact_log.create_head(init_repo=not is_detached_head)
        #     artifact_values = {"head_id": head["id"]}
        artifact_log = None
        if is_versioned:
            artifact_log = ArtifactLog.get_current()
            assert artifact_log.is_initialized
        if is_head:
            artifact_log = ArtifactLog()
            
        def zip_metadata(vectors, metadata):
            if not vectors:
                for meta in metadata:
                    yield meta
            else:
                for vec, meta in zip(vectors, metadata):
                    yield {**vec, **meta}
        

        results = []
        async with self.pool.acquire() as conn:
            for chunk in chunks(zip_metadata(vectors, metadata), batch_size=batch_size):            
                values = []             
                for item in chunk:
                    item.pop("_subspace")
                    if artifact_log:
                        if is_versioned:
                            item.update({"turn_id": artifact_log.head["turn_id"], "branch_id": artifact_log.head["branch_id"]})
                        if is_head:
                            head = await artifact_log.create_head(init_repo=not is_detached_head)
                            item.update({"head_id": head["id"]})                        
                    # item_values = field_mapper.render_insert_values(item, exclude_types=["relation", "key"], exclude_fields=["_subspace"])
                    item_values = field_mapper.render_insert_values(item, exclude_types=["key"], exclude_fields=["_subspace"])
                    values.extend(item_values)
                # fields_sql = field_mapper.render_insert_fields(chunk[0], exclude_types=["relation", "key"], exclude_fields=["_subspace"])
                # place_holders_sql = field_mapper.render_placeholders(chunk, exclude_types=["relation", "key"], exclude_fields=["_subspace"])
                fields_sql = field_mapper.render_insert_fields(chunk[0], exclude_types=["key"], exclude_fields=["_subspace"])
                place_holders_sql = field_mapper.render_placeholders(chunk, exclude_types=["key"], exclude_fields=["_subspace"])
                sql = f"{fields_sql}\n{place_holders_sql}\nRETURNING *;"
                try:
                    results.extend(await conn.fetch(sql, *values))
                except Exception as e:
                    print(e)                    
                    print(shorten_vector_strings(sql))
                    print(values)
                    raise e                
                
        return results
    
    
    async def update(
        self,
        namespace: str,
        id: str | int,
        vectors: List[dict[str, List[float] | Any]],
        metadata: List[Dict],
        is_versioned: bool = False,
        field_mapper: FieldMapper | None = None,
    ):
        if is_versioned:
            raise ValueError("Versioned models cannot be updated")
        await self._ensure_connected()
        assert self.pool is not None
        
        if field_mapper is None:
            raise ValueError("Field mapper is required")
            
        table_name = namespace
        
        # Combine vectors and metadata
        update_values = {}
        if vectors and len(vectors) > 0:
            update_values.update(vectors[0])
        if metadata and len(metadata) > 0:
            update_values.update(metadata[0])
            
        if "_subspace" in update_values:
            update_values.pop("_subspace")
            
        if not update_values:
            return []
            
        # Build SET clause
        set_parts = []
        values = []
        next_idx = 1
        
        for field_name, value in update_values.items():
            set_parts.append(f'"{field_name}" = ${next_idx}')
            values.append(field_mapper.field_lookup[field_name].render_insert_value(value))
            next_idx += 1
            
        set_clause = ", ".join(set_parts)
        
        # Add updated_at timestamp
        set_clause += ", updated_at = NOW()"
        
        query = f"""
        UPDATE {table_name} 
        SET {set_clause}
        WHERE id = ${next_idx}
        RETURNING *
        """
        
        values.append(id)
        
        async with self.pool.acquire() as conn:
            try:
                results = await conn.fetch(query, *values)
                return results
            except Exception as e:
                print(e)
                print(shorten_vector_strings(query))
                print(values)
                raise e

    async def upsert2(
        self,
        namespace: str,
        vectors: List[dict[str, List[float] | Any]],
        metadata: List[Dict],
        model_cls: Type[BaseModel],
        ids=None,
        batch_size=100,
        is_versioned: bool = False,
        is_head: bool = False,
        is_detached_head: bool = False
    ):
        """Upsert vectors and metadata into the collection with fixed turn_id and type fields."""
        await self._ensure_connected()
        assert self.pool is not None
        if is_versioned and is_head:
            raise ValueError("Versioned and head models cannot be created at the same time")
        artifact_values = {}
        if is_versioned:
            artifact_log = ArtifactLog.get_current()
            assert artifact_log.is_initialized
            artifact_values = {"turn_id": artifact_log.head["turn_id"], "branch_id": artifact_log.head["branch_id"]}
        if is_head:
            artifact_log = ArtifactLog()
            head = await artifact_log.create_head(init_repo=not is_detached_head)
            artifact_values = {"head_id": head["id"]}
        


        # Define the fixed values you want to add to every record.
        # You can use hard-coded values (e.g. {"turn_id": 42, "type": "my_type"})
        # or values derived from artifact_log (e.g. artifact_log.head["turn_id"])
        

        # Build the base columns from the first record and add the fixed columns.
        if vectors:
            base_columns = list(vectors[0].keys())
        else:
            base_columns = list(metadata[0].keys())

        # Remove any keys you want to skip (e.g. "_subspace") and also do not duplicate fixed keys.
        base_columns = [col for col in base_columns if col != "_subspace" and col not in artifact_values]
        # Now add the fixed columns.
        columns = base_columns + list(artifact_values.keys())

        def zip_metadata(vectors, metadata):
            if not vectors:
                for meta in metadata:
                    yield meta
            else:
                for vec, meta in zip(vectors, metadata):
                    yield {**vec, **meta}

        results = []
        async with self.pool.acquire() as conn:
            for chunk in chunks(zip_metadata(vectors, metadata), batch_size=batch_size):
                # Update each row with our fixed values.
                for item in chunk:
                    item.update(artifact_values)

                placeholders = []
                values = []
                for i, item in enumerate(chunk):
                    start_idx = i * len(columns)
                    col_placeholders = []
                    for j, col in enumerate(columns, 1):
                        placeholder_idx = start_idx + j
                        if isinstance(item[col], np.ndarray):
                            item[col] = stringify_vector(item[col])
                            col_placeholders.append(f"${placeholder_idx}::vector")
                        elif isinstance(item[col], list):
                            item[col] = stringify_vector(item[col])
                            col_placeholders.append(f"${placeholder_idx}::vector")
                        elif isinstance(item[col], dict):
                            item[col] = json.dumps(item[col])
                            col_placeholders.append(f"${placeholder_idx}::jsonb")
                        elif isinstance(item[col], BaseModel):
                            item[col] = item[col].model_dump_json()
                            col_placeholders.append(f"${placeholder_idx}::jsonb")
                        elif isinstance(item[col], Enum):
                            item[col] = item[col].value
                            col_placeholders.append(f"${placeholder_idx}")
                        else:
                            col_placeholders.append(f"${placeholder_idx}")
                        values.append(item[col])
                    placeholders.append(f"({', '.join(col_placeholders)})")

                query = f"""
                INSERT INTO {namespace} ({', '.join(f'"{col}"' for col in columns)})
                VALUES {', '.join(placeholders)}
                RETURNING *;
                """
                try:
                    results.extend(await conn.fetch(query, *values))
                except Exception as e:
                    print(e)
                    print(shorten_vector_strings(query))
                    print(values)
                    raise e
        
        return results

    async def search(self, collection_name: str, query, limit=3, filters=None, with_vectors=False, threshold=None):
        """Search for similar vectors using cosine similarity."""
        await self._ensure_connected()
        assert self.pool is not None

        table_name = camel_to_snake(collection_name)

        async with self.pool.acquire() as conn:
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            # Assuming query contains vector name and values
            vector_name, vector_values = next(iter(query.items()))
            
            select_clause = "id, payload"
            if with_vectors:
                select_clause += f', "{vector_name}"'

            query = f"""
            SELECT {select_clause},
                   1 - ("{vector_name}" <=> $1::vector) as score
            FROM {table_name}
            {where_clause}
            ORDER BY score DESC
            LIMIT {limit}
            """

            if threshold:
                query = query.replace("ORDER BY", f"HAVING score >= {threshold}\nORDER BY")

            return await conn.fetch(query, vector_values)

    def _build_where_clause(self, query_filter: QueryFilter) -> str:
        """Convert QueryFilter to SQL WHERE clause."""
        if isinstance(query_filter._operator, QueryOp):
            if query_filter._operator == QueryOp.AND:
                return f"({self._build_where_clause(query_filter._left)} AND {self._build_where_clause(query_filter._right)})"
            elif query_filter._operator == QueryOp.OR:
                return f"({self._build_where_clause(query_filter._left)} OR {self._build_where_clause(query_filter._right)})"
        elif isinstance(query_filter._operator, FieldOp):
            field = query_filter.field
            field_name = f'"{field.name}"'
            
            # Get field type information
            if hasattr(field, '_field_info'):
                field_type = field._field_info.annotation
            else:
                field_type = field.type
                
            
            # Handle JSONB fields differently
            if str(field_type).startswith("dict") or (isinstance(field_type, type) and issubclass(field_type, BaseModel)):
                json_field = f"payload->'{field.name}'"
                if query_filter._operator == FieldOp.NULL:
                    return f"{json_field} IS NULL"
                elif query_filter._operator == FieldOp.EQ:
                    return f"{json_field} = '{json.dumps(query_filter.value)}'"
                elif query_filter._operator == FieldOp.NE:
                    return f"{json_field} != '{json.dumps(query_filter.value)}'"
                elif query_filter._operator == FieldOp.IN:
                    values = [json.dumps(v) for v in query_filter.value]
                    return f"{json_field} = ANY(ARRAY[{', '.join(values)}])"
                elif query_filter._operator == FieldOp.NOTIN:
                    values = [json.dumps(v) for v in query_filter.value]
                    return f"{json_field} != ALL(ARRAY[{', '.join(values)}])"
            
            # Handle regular fields with proper type casting
            if query_filter._operator == FieldOp.NULL:
                return f"{field_name} IS NULL"
            elif query_filter._operator == FieldOp.EQ:
                if field_type == bool:
                    return f"{field_name} = {str(query_filter.value).lower()}"
                elif field_type == int:
                    return f"{field_name} = {query_filter.value}"
                elif field_type == str:
                    return f"{field_name} = '{query_filter.value}'"
                elif field_type == dt.datetime:
                    return f"{field_name} = '{query_filter.value}'"
            elif query_filter._operator == FieldOp.NE:
                if field_type == bool:
                    return f"{field_name} != {str(query_filter.value).lower()}"
                elif field_type == int:
                    return f"{field_name} != {query_filter.value}"
                elif field_type == str:
                    return f"{field_name} != '{query_filter.value}'"
                elif field_type == dt.datetime:
                    return f"{field_name} != '{query_filter.value}'"
            elif query_filter._operator == FieldOp.IN:
                if field_type == int:
                    values = [str(v) for v in query_filter.value]
                    return f"{field_name} = ANY(ARRAY[{', '.join(values)}])"
                else:
                    values = [f"'{v}'" for v in query_filter.value]
                    return f"{field_name} = ANY(ARRAY[{', '.join(values)}])"
            elif query_filter._operator == FieldOp.NOTIN:
                if field_type == int:
                    values = [str(v) for v in query_filter.value]
                    return f"{field_name} != ALL(ARRAY[{', '.join(values)}])"
                else:
                    values = [f"'{v}'" for v in query_filter.value]
                    return f"{field_name} != ALL(ARRAY[{', '.join(values)}])"
            elif query_filter._operator == FieldOp.RANGE:
                conditions = []
                if query_filter.value.gt is not None:
                    if field_type == int:
                        conditions.append(f"{field_name} > {query_filter.value.gt}")
                    else:
                        conditions.append(f"{field_name} > '{query_filter.value.gt}'")
                if query_filter.value.ge is not None:
                    if field_type == int:
                        conditions.append(f"{field_name} >= {query_filter.value.ge}")#f"{field_name} >= to_timestamp('{query_filter.value.ge.strfmt('YYYY-MM-DD HH24:MI:SS')}')"
                    # elif field_type == dt.datetime:                        
                        # conditions.append(f"{field_name} >= to_timestamp('{query_filter.value.ge.strftime('%Y-%m-%d %H:%M:%S')}')")
                    else:
                        conditions.append(f"{field_name} >= '{query_filter.value.ge}'")
                if query_filter.value.lt is not None:
                    if field_type == int:
                        conditions.append(f"{field_name} < {query_filter.value.lt}")
                    else:
                        conditions.append(f"{field_name} < '{query_filter.value.lt}'")
                if query_filter.value.le is not None:
                    if field_type == int:
                        conditions.append(f"{field_name} <= {query_filter.value.le}")
                    else:
                        conditions.append(f"{field_name} <= '{query_filter.value.le}'")
                return f"({' AND '.join(conditions)})"
        return ""

    async def delete(self, collection_name: str, ids: List[str] | List[int] | None = None, filters: Any | None = None):
        """Delete records from the collection."""
        await self._ensure_connected()
        assert self.pool is not None

        table_name = collection_name

        async with self.pool.acquire() as conn:
            if ids:
                if isinstance(ids[0], str):
                    return await conn.execute(
                        f'DELETE FROM "{table_name}" WHERE id = ANY($1::text[]);',
                        ids
                    )
                else:
                    return await conn.execute(
                        f'DELETE FROM "{table_name}" WHERE id = ANY($1::int[]);',
                        ids
                    )
            elif filters:
                where_clause = self._build_where_clause(filters)
                return await conn.execute(
                    f'DELETE FROM "{table_name}" WHERE {where_clause};'
                )
            else:
                raise ValueError("Either ids or filters must be provided.")

    async def scroll(
        self, 
        collection_name: str, 
        filters=None, 
        limit=10, 
        offset=0, 
        with_vectors=False, 
        order_by=None,
        field_mapper: FieldMapper | None = None
        ):
        """Implement scrolling pagination."""
        await self._ensure_connected()
        assert self.pool is not None

        table_name = camel_to_snake(collection_name)

        async with self.pool.acquire() as conn:
            where_clause = ""
            if filters:
                where_clause = f"WHERE {self._build_where_clause(filters)}"

            order_clause = 'ORDER BY "id"'
            if order_by:
                if isinstance(order_by, str):
                    order_clause = f'ORDER BY "{order_by}"'
                elif isinstance(order_by, dict):
                    order_clause = f'ORDER BY "{order_by["key"]}" {order_by["direction"]}'

            select_clause = "*"
            if with_vectors:
                # Get all vector columns
                vector_columns = await conn.fetch("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = $1 
                    AND data_type = 'vector'
                """, table_name)
                if vector_columns:
                    select_clause = '"id", ' + ', '.join(f'"{r["column_name"]}"' for r in vector_columns)

            query = f"""
            SELECT {select_clause}
            FROM {table_name}
            {where_clause}
            {order_clause}
            LIMIT {limit} OFFSET {offset}
            """

            records = await conn.fetch(query)
            next_offset = offset + len(records) if len(records) == limit else None
            records = [field_mapper.unpack_record(r) for r in records]
            return records, next_offset

    async def delete_collection(self, collection_name: str):
        """Drop the collection table."""
        await self._ensure_connected()
        assert self.pool is not None

        table_name = camel_to_snake(collection_name)

        async with self.pool.acquire() as conn:
            await conn.execute(f'DROP TABLE IF EXISTS {table_name}')

    async def get_collections(self):
        """Get all collection (table) names."""
        await self._ensure_connected()
        assert self.pool is not None

        async with self.pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            return [table["table_name"] for table in tables]

    async def get_collection(self, collection_name: str, raise_error=True):
        """Get collection (table) information."""
        await self._ensure_connected()
        assert self.pool is not None

        table_name = camel_to_snake(collection_name)

        async with self.pool.acquire() as conn:
            try:
                info = await conn.fetchrow("""
                    SELECT * 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = $1
                """, table_name)
                if not info and raise_error:
                    raise ValueError(f"Collection {collection_name} does not exist")
                return info
            except Exception as e:
                if raise_error:
                    raise e
                return None 

    async def execute_query(
        self, 
        collection_name: str, 
        query_set: "QuerySet", 
        is_versioned: bool = False, 
        is_head: bool = False,
        field_mapper: FieldMapper | None = None
        ):
        """Execute a query set and return the results."""
        
        table_name = collection_name
        
        if field_mapper is None:
            raise ValueError("Field mapper is required")
        if query_set.query_type == "vector":
            # Handle vector similarity search
            if not query_set._vector_query or not query_set._vector_query.vector_lookup:
                raise ValueError("Vector query not provided or vectors not embedded")
            
            vector_name, vector_values = next(iter(query_set._vector_query.vector_lookup.items()))
            vector_values = stringify_vector(vector_values)
            where_clause = ""
            if query_set._filters:
                where_clause = f"WHERE {self._build_where_clause(query_set._filters)}"

            query = f"""
            SELECT *,
                   1 - ("{vector_name}" <=> $1::vector) as score
            FROM {table_name}
            {where_clause}
            ORDER BY score DESC
            LIMIT {query_set._limit}
            """

            if query_set._vector_query.threshold:
                query = query.replace("ORDER BY", f"HAVING score >= {query_set._vector_query.threshold}\nORDER BY")

            async with self.pool.acquire() as conn:
                return await conn.fetch(query, vector_values)
                
        elif query_set.query_type == "scroll":
            # Handle regular queries with filtering and ordering
            where_clause = ""
            if query_set._filters:
                where_clause = f"WHERE {self._build_where_clause(query_set._filters)}"

            order_clause = 'ORDER BY "id"'
            if query_set._order_by:
                if isinstance(query_set._order_by, str):
                    order_clause = f'ORDER BY "{query_set._order_by}"'
                elif isinstance(query_set._order_by, dict):
                    order_clause = f'ORDER BY "{query_set._order_by["key"]}" {query_set._order_by["direction"]}'
            if query_set._partitions:
                partition_clauses = []
                for partition_name, partition_value in query_set._partitions.items():
                    if partition_name == "_subspace":
                        continue
                    partition_clauses.append(f'"{partition_name}" = {partition_value}')
                if partition_clauses:
                    if where_clause:
                        where_clause += f" AND ({', '.join(partition_clauses)})"
                    else:
                        where_clause = f"WHERE ({', '.join(partition_clauses)})"

            query = f"""
            SELECT *
            FROM {table_name}
            {where_clause}
            {order_clause}
            LIMIT {query_set._limit}
            """
            
            if query_set._offset is not None:
                query += f" OFFSET {query_set._offset}"

            if is_versioned:
                artifact_log = ArtifactLog.get_current()
                records = await artifact_log.artifact_cte_raw_query(
                    artifact_table=table_name,
                    artifact_query=query
                )
            else:
                async with self.pool.acquire() as conn:
                    records = await conn.fetch(query)
            # return res
            # async with self.pool.acquire() as conn:
            #     return await conn.fetch(query)
                
        elif query_set.query_type == "id":
            # Handle queries by ID
            if not query_set._ids:
                raise ValueError("No IDs provided for ID query")
                
            async with self.pool.acquire() as conn:
                records = await conn.fetch(
                    f'SELECT * FROM {table_name} WHERE id = ANY($1::text[])',
                    query_set._ids
                )
        else:
            raise ValueError(f"Unsupported query type: {query_set.query_type}") 
            
        return [field_mapper.unpack_record(r) for r in records]
        
        
        
    async def retrieve(self, namespace: str, ids: List[str] | List[int], field_mapper: FieldMapper):
        await self._ensure_connected()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            records = await conn.fetch(f'SELECT * FROM "{namespace}" WHERE id = ANY($1)', ids)
            return [field_mapper.unpack_record(r) for r in records]
        
        
        
    