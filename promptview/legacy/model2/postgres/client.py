
from typing import Any
from promptview.utils.db_connections import PGConnectionManager



def print_error_sql(sql: str, values: Any | None = None, error: Exception | None = None):
    print("---------- SQL ------------:\n", sql)
    if values:
        print("---------- VALUES ----------:\n", values)
    if error:
        print("---------- ERROR ----------:\n", error)


class PostgresClient:
    
    
    async def fetch(self, sql: str, *values: Any):
        """Fetch results from a SQL query"""
        try:
            res = await PGConnectionManager.fetch(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
    
    async def execute(self, sql: str, *values: Any):
        """Execute a SQL statement"""
        try:
            res = await PGConnectionManager.execute(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
        
        
        
        
class VersionedPostgresClient(PostgresClient):
    
    
    def __init__(self, branch_id: int):
        self.branch_id = branch_id
    
    
    async def fetch(self, sql: str, *values: Any):
        """Fetch results from a SQL query"""
        try:
            res = await PGConnectionManager.fetch(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
    
    async def execute(self, sql: str, *values: Any):
        """Execute a SQL statement"""
        try:
            res = await PGConnectionManager.execute(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e