from enum import Enum
import inspect
from typing import TYPE_CHECKING, Any, Literal
from chatboard.utils.db_connections import PGConnectionManager, SyncPGConnectionManager
from chatboard.utils.model_utils import get_list_type, is_list_type
import datetime as dt
from pydantic import BaseModel


if TYPE_CHECKING:
    from chatboard.model.postgres2.pg_namespace import PgNamespace, PgFieldInfo



class PgManager:
    """SQL builder for PostgreSQL"""

    # PostgreSQL type constants
    
    
    @classmethod
    def execute(cls, sql: str):
        """Execute a SQL statement"""
        try:
            res = SyncPGConnectionManager.execute(sql)
            return res
        except Exception as e:
            print(sql)
            raise e

    @classmethod
    def fetch(cls, sql: str):
        """Fetch results from a SQL query"""
        try:
            res = SyncPGConnectionManager.fetch(sql)
            return res
        except Exception as e:
            print(sql)
            raise e
        



    
    @classmethod
    def update_table_fields(cls, table_name: str, fields: "list[PgFieldInfo]"):
        """Update the fields of a table"""
        sql = ''
        for field in fields:
            sql += f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS "{field.name}" {field.sql_type}'
            if field.is_optional:
                sql += " NULL"
            else:
                sql += " NOT NULL"
            sql += ",\n"
        sql = sql[:-2]
        sql += ";   "
        cls.execute(sql)
    
    @classmethod
    def create_index(
        cls,
        index_name: str,
        table_name: str,
        columns: list[str],
        unique: bool = False
    ) -> None:
        """Create an index for a table and columns, optionally unique."""
        unique_sql = "UNIQUE " if unique else ""
        columns_sql = ", ".join(f'"{col}"' for col in columns)
        sql = f'CREATE {unique_sql}INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ({columns_sql});'
        cls.execute(sql)
        
        
    @classmethod
    def list_enum_types(cls) -> list[str]:
        """Get all enum types"""
        sql = """
        SELECT typname, typcategory, typnamespace::regnamespace 
        FROM pg_type 
        WHERE typcategory IN ('E')
        ORDER BY typname;
        """
        res = cls.fetch(sql)
        return [row["typname"] for row in res]
    
    @classmethod
    def create_enum(cls, enum_name: str, enum_values: list[str]):
        enum_clouse = ", ".join([f"'{v}'" for v in enum_values])
        query = f"""
            DO $$ 
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = '{enum_name}') THEN
                    CREATE TYPE {enum_name} AS ENUM ({enum_clouse});
                END IF;
            END $$;
            """
        cls.execute(query)
    


                
                
    @classmethod
    def drop_enum_types(cls, exclude: list[str] | None = None) -> None:
        """Drop an enum for a namespace"""
        if exclude is None:
            exclude = []
        
        def should_drop(enum_name: str) -> bool:
            for e in exclude:
                if enum_name.startswith(e):
                    return False
            return True
        
        for enum_name in cls.list_enum_types():
            if should_drop(enum_name):
                cls.execute(f"DROP TYPE IF EXISTS {enum_name};")

    @classmethod
    def drop_table(cls, namespace: "PgNamespace | str") -> str:
        if isinstance(namespace, str):
            sql = f"DROP TABLE IF EXISTS {namespace}"
        else:
            sql = f"DROP TABLE IF EXISTS {namespace.name}"
        return cls.execute(sql)
    
    
    @classmethod
    def drop_many_tables(cls, table_names: list[str]) -> None:
        sql = f"DROP TABLE IF EXISTS {', '.join(table_names)}"
        cls.execute(sql)
        
    @classmethod
    def drop_all_tables(cls, exclude: list[str] | None = None) -> None:
        if exclude is None:
            exclude = []
        tables = cls.get_tables()
        tables = [table for table in tables if table not in exclude]
        if not tables:
            return
        cls.drop_many_tables(tables)
        cls.drop_enum_types(exclude)
        # for table in tables:
        #     if table not in exclude:
        #         await cls.drop_table(table)
        
    @classmethod
    def get_tables(cls, schema: str | None = "public") -> list[str]:
        sql = "SELECT table_name FROM information_schema.tables"
        if schema:
            sql += f" WHERE table_schema='{schema}'"
        res = cls.fetch(sql)
        return [row["table_name"] for row in res]
    
    
    @classmethod
    def get_table_fields(cls, table_name: str, schema: str = 'public'):
        query = """
        SELECT *
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = $2
        ORDER BY ordinal_position
        """
        rows = PGConnectionManager.fetch(query, schema, table_name)
        return rows
        # return [(row['column_name'], row['data_type']) for row in rows]

    @classmethod
    def create_foreign_key(
        cls,
        table_name: str,
        column_name: str,
        column_type: str,
        referenced_table: str,
        referenced_column: str = "id",
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
    ) -> None:
        """
        Create a foreign key constraint.
        
        Args:
            table_name: The name of the table
            column_name: The name of the column
            referenced_table: The name of the referenced table
            referenced_column: The name of the referenced column
            on_delete: The action to take when the referenced row is deleted
            on_update: The action to take when the referenced row is updated
        """
        # Check if the column exists
        sql = f"""
        DO $$
        BEGIN
            -- Check if the column exists; if not, add it.
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = '{table_name}'
                AND column_name = '{column_name}'
            ) THEN
                EXECUTE 'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type}';
            END IF;
            
            -- Check if the foreign key constraint exists; if not, add it.
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'fk_{table_name}_{column_name}'
                AND conrelid = '{table_name}'::regclass
            ) THEN
                EXECUTE '
                ALTER TABLE "{table_name}"
                ADD CONSTRAINT fk_{table_name}_{column_name}
                FOREIGN KEY ("{column_name}")
                REFERENCES "{referenced_table}"("{referenced_column}")
                ON DELETE {on_delete}
                ON UPDATE {on_update}
                ';
            END IF;
        END $$;
        """
        cls.execute(sql)




    @classmethod
    def create_extension(cls, extension_name: str) -> None:
        """Create an extension"""
        cls.execute(f"CREATE EXTENSION IF NOT EXISTS {extension_name};")

