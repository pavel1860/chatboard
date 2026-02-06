from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypedDict, Union
import json
import datetime as dt
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel

from chatboard.model.postgres.query_parser import build_where_clause
from chatboard.model.query_filters import QueryProxy
from chatboard.utils.db_connections import PGConnectionManager
from chatboard.model.versioning import ArtifactLog, Turn, Branch, TurnStatus

if TYPE_CHECKING:
    from chatboard.model.postgres.namespace import PgFieldInfo, NSRelationInfo, PostgresNamespace
    


def print_error_sql(sql: str, values: Any | None = None, error: Exception | None = None):
    print("---------- SQL ------------:\n", sql)
    if values:
        print("---------- VALUES ----------:\n", values)
    if error:
        print("---------- ERROR ----------:\n", error)
        
class JoinType(TypedDict):
    """Join information"""
    primary_table: str
    primary_key: str
    foreign_table: str
    foreign_key: str
    
class SelectType(TypedDict):
    """Select information"""
    namespace: str
    fields: list[str]

class PostgresOperations:
    """Operations for PostgreSQL database"""

    
    # @classmethod
    # async def create_branch(cls, name: Optional[str] = None, forked_from_turn_id: Optional[int] = None) -> Branch:
    #     """Create a new branch"""
    #     # Initialize versioning if needed
    #     # await cls.initialize_versioning()
        
    #     turn = None
    #     if forked_from_turn_id is not None:
    #         turn = await cls.get_turn(forked_from_turn_id)
    #         turn_index = turn.index
    #         turn_branch_id = turn.branch_id
    #     else:
    #         turn_index = 1
    #         turn_branch_id = None
        
    #     if name is None:
    #         if forked_from_turn_id is None:
    #             name = "main"
    #         else:
    #             name = f"branch_from_{forked_from_turn_id}"
        
    #     # Create the branch
    #     query = "INSERT INTO branches (name, forked_from_turn_index, forked_from_branch_id) VALUES ($1, $2, $3) RETURNING *;"
    #     branch_row = await PGConnectionManager.fetch_one(query, name, turn_index, turn_branch_id)
    #     if branch_row is None:
    #         raise ValueError("Failed to create branch")
        
        
    #     return Branch(**dict(branch_row))
        # Create a turn for the branch
        # new_turn = await cls.create_turn(branch_row['id'], 1 if turn is None else turn.index + 1, TurnStatus.STAGED)
        
        # return Branch(**dict(branch_row), last_turn=new_turn)
        
    @classmethod
    async def fetch(cls, sql: str, *values: Any):
        """Fetch results from a SQL query"""
        try:
            res = await PGConnectionManager.fetch(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
    @classmethod
    async def execute(cls, sql: str, *values: Any):
        """Execute a SQL statement"""
        try:
            res = await PGConnectionManager.execute(sql, *values)
            return res
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
    # @classmethod
    # async def create_turn(cls, partition_id: int, branch_id: int = 1, status: TurnStatus=TurnStatus.STAGED) -> Turn:
    #     """
    #     Create a new turn
        
    #     Args:
    #         partition_id: The partition ID
    #         branch_id: The branch ID
    #         status: The status of the turn
    #     """
    #     query = """
    #     WITH updated_branch AS (
    #         UPDATE branches
    #         SET current_index = current_index + 1
    #         WHERE id = $1
    #         RETURNING id, current_index
    #     ),
    #     new_turn AS (
    #         INSERT INTO turns (partition_id, branch_id, index, status)
    #         SELECT $2, id, current_index, $3
    #         FROM updated_branch
    #         RETURNING *
    #     )
    #     SELECT * FROM new_turn;
    #     """
        
    #     # Use a transaction to ensure atomicity
    #     async with PGConnectionManager.transaction() as tx:
    #         turn_row = await tx.fetch_one(query, branch_id, partition_id, status.value)
    #         if turn_row is None:
    #             raise ValueError(f"Failed to create turn for branch {branch_id}")
        
    #     return Turn(**dict(turn_row))
    
    # async def get_branch_turns(self, branch_id: int, limit: int = 10, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[Turn]:
    #     query = f"""
    #         SELECT 
    #             t.id,
    #             t.branch_id,
    #             t.index,
    #             t.status,
    #             t.created_at,
    #             t.ended_at,
    #             t.message,
    #             t.local_state,
    #             COALESCE(
    #                 json_agg(
    #                     json_build_object(
    #                         'id', b.id,
    #                         'name', b.name,
    #                         'forked_from_turn_index', b.forked_from_turn_index,
    #                         'forked_from_branch_id', b.forked_from_branch_id,
    #                         'created_at', b.created_at,
    #                         'updated_at', b.updated_at
    #                     ) ORDER BY b.created_at  -- ordering the aggregated forked branches by creation time
    #                 ) FILTER (WHERE b.id IS NOT NULL),
    #                 '[]'
    #             ) AS forked_branches
    #         FROM turns t
    #         LEFT JOIN branches b 
    #             ON b.forked_from_branch_id = t.branch_id 
    #         AND b.forked_from_turn_index = t.index
    #         WHERE t.branch_id = {branch_id}  -- Parameter for the specific branch_id
    #         GROUP BY t.id
    #         ORDER BY t.index ASC     -- Ordering the turns by their index
    #         LIMIT {limit} OFFSET {offset};      -- Limit and offset parameters

    #     """
    #     rows = await PGConnectionManager.fetch(query)
    #     return [Turn(**dict(row)) for row in rows]
    
    # @classmethod
    # async def create_turn2(cls, branch_id: int, status: TurnStatus) -> Turn:
    #     """Create a new turn"""
    #     query = """
    #     WITH updated_branch AS (
    #         UPDATE branches
    #         SET current_index = current_index + 1
    #         WHERE id = $1
    #         RETURNING id, current_index
    #     ),
    #     new_turn AS (
    #         INSERT INTO turns (branch_id, index, status)
    #         SELECT id, current_index, $2
    #         FROM updated_branch
    #         RETURNING *
    #     )
    #     SELECT * FROM new_turn;
    #     """
        
    #     # Use a transaction to ensure atomicity
    #     async with PGConnectionManager.transaction() as tx:
    #         turn_row = await tx.fetch_one(query, branch_id, status.value)
    #         if turn_row is None:
    #             raise ValueError(f"Failed to create turn for branch {branch_id}")
        
    #     return Turn(**dict(turn_row))
    
    # @classmethod
    # async def get_turn(cls, turn_id: int) -> Turn:
    #     """Get a turn by ID"""
    #     query = "SELECT * FROM turns WHERE id = $1;"
    #     turn_row = await PGConnectionManager.fetch_one(query, turn_id)
    #     if turn_row is None:
    #         raise ValueError(f"Turn {turn_id} not found")
        
    #     return Turn(**dict(turn_row))
    
    
    # @classmethod
    # async def get_branch_or_none(cls, branch_id: int) -> Branch | None:
    #     """Get a branch by ID"""
    #     # Get the branch
    #     query = "SELECT * FROM branches WHERE id = $1;"
    #     branch_row = await PGConnectionManager.fetch_one(query, branch_id)
    #     if branch_row is None:
    #         return None
    #     return Branch(**dict(branch_row))
    
    
    # @classmethod
    # async def get_branch(cls, branch_id: int) -> Branch:
    #     """Get a branch by ID"""
    #     # Get the branch
    #     branch = await cls.get_branch_or_none(branch_id)
    #     if branch is None:
    #         raise ValueError(f"Branch {branch_id} not found")        
    #     return branch  
    
    # @classmethod
    # async def list_branches(cls, limit: int = 100, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> list[Branch]:
    #     """List all branches"""
    #     query = f"SELECT * FROM branches ORDER BY {order_by} {order_direction} LIMIT $1 OFFSET $2;"
    #     branches = await PGConnectionManager.fetch(query, limit, offset)
    #     return [Branch(**dict(branch)) for branch in branches]
    
    # @classmethod
    # async def commit_turn(cls, turn_id: int, message: Optional[str] = None) -> Turn:
    #     """Commit the current turn and create a new one"""        
    #     # Update the turn
    #     now = datetime.now(timezone.utc).replace(tzinfo=None)
    #     query = """
    #     UPDATE turns
    #     SET status = $1, ended_at = $2, message = $3
    #     WHERE id = $4
    #     RETURNING *;
    #     """
    #     result = await PGConnectionManager.fetch_one(query, TurnStatus.COMMITTED.value, now, message, turn_id)
    #     if result is None:
    #         raise ValueError(f"Turn {turn_id} not found")
    #     return Turn(**dict(result))
        
        
    # @classmethod
    # async def _update_version_params(cls, namespace: "PostgresNamespace", data: Dict[str, Any], turn_id: Optional[int] = None, branch_id: Optional[int] = None):
    #     # if namespace.is_repo and "id" not in data and "main_branch_id" not in data:
    #     #     # Create a new main branch
    #     #     branch = await cls.create_branch(name="main")
    #     #     # Add main_branch_id to the data
    #     #     data["main_branch_id"] = branch.id
            
    #     if namespace.repo_namespace:
    #         if branch_id is None:
    #             raise ValueError("Versioned model cannot be saved without a branch_id")
    #         if turn_id is None:
    #             raise ValueError("Versioned model cannot be saved without a turn_id")
            
            
        
    #     # If branch_id is provided, get the current turn
    #     if branch_id is not None and turn_id is not None:        
    #         data["branch_id"] = branch_id
    #         data["turn_id"] = turn_id
    #         turn = await cls.get_turn(turn_id)
    #         if turn.status != TurnStatus.STAGED:
    #             raise ValueError(f"Turn {turn_id} is {turn.status.value}, not STAGED.")
    #     return data
    
    
    @classmethod
    async def insert(cls, namespace: "PostgresNamespace", data: Dict[str, Any], turn_id: Optional[int] = None, branch_id: Optional[int] = None) -> Dict[str, Any]:
        """Insert a new record"""
        # Prepare data for insertion
        keys = []
        placeholders = []
        values = []
        
        data = await ArtifactLog.update_version_params(namespace, data, turn_id, branch_id)
        for key, value in data.items():
            if key == "id" and value is None:
                continue
            
            if key == "branch_id" or key == "turn_id":
                keys.append(f'"{key}"')
                placeholders.append(f'${len(values) + 1}')
                values.append(value)
                continue
            
            field_info = namespace.get_field(key)
            if field_info.validate_value(value):
                placeholder = field_info.get_placeholder(len(values) + 1)
                processed_value = field_info.serialize(value)
                
                keys.append(f'"{key}"')
                placeholders.append(placeholder)
                values.append(processed_value)
            else:
                raise ValueError(f"Field {key} is not valid")
            
        # Build SQL query
        sql = f"""
        INSERT INTO "{namespace.table_name}" ({", ".join(keys)})
        VALUES ({", ".join(placeholders)})
        RETURNING *;
        """
        
        # Execute query
        try:
            result = await PGConnectionManager.fetch_one(sql, *values)
        except Exception as e:
            print_error_sql(sql, values, e)
            raise e
        
        # Convert result to dictionary
        return dict(result) if result else {"id": 1, **data}
    
    @classmethod
    async def update(cls, namespace: "PostgresNamespace", id: Any, data: Dict[str, Any], turn_id: Optional[int] = None, branch_id: Optional[int] = None) -> Dict[str, Any]:
        """Update an existing record"""
        # Prepare data for update
        set_parts = []
        values = []
        
        data = await ArtifactLog.update_version_params(namespace, data, turn_id, branch_id)
        
        for key, value in data.items():
            if key == "id":
                continue
            
            if key == "branch_id" or key == "turn_id":
                continue
            
            field_info = namespace.get_field(key)
            if field_info.validate_value(value):
                placeholder = field_info.get_placeholder(len(values) + 1)
                processed_value = field_info.serialize(value)
                
                set_parts.append(f'"{key}" = {placeholder}')
                values.append(processed_value)
            else:
                raise ValueError(f"Field {key} is not valid")

        # Add ID to values
        values.append(id)
        
        # Build SQL query
        sql = f"""
        UPDATE "{namespace.table_name}"
        SET {", ".join(set_parts)}
        WHERE id = ${len(values)}
        RETURNING *;
        """
        
        # Execute query
        result = await PGConnectionManager.fetch_one(sql, *values)
        
        # Convert result to dictionary
        return dict(result) if result else {"id": data["id"], **data}        
    
    @classmethod
    async def delete(cls, namespace: "PostgresNamespace", id: Any) -> Dict[str, Any]:
        """Delete an existing record"""
        # Prepare data for deletion
        sql = f"""
        DELETE FROM "{namespace.table_name}" WHERE id = $1
        RETURNING *;
        """
        
        # Execute query
        result = await PGConnectionManager.fetch_one(sql, id)
        
        # Convert result to dictionary
        return dict(result) if result else {"id": id}
    
    
    @classmethod
    async def get(cls, namespace: "PostgresNamespace", id: Any) -> Optional[Dict[str, Any]]:
        """
        Get a record by ID.
        
        Args:
            namespace: The namespace to get from
            id: The ID of the record to get
            
        Returns:
            The record if found, None otherwise
        """
        # Build SQL query
        sql = f'SELECT * FROM "{namespace.table_name}" WHERE id = $1;'
        
        # Execute query
        result = await PGConnectionManager.fetch_one(sql, id)
        
        # Convert result to dictionary
        return dict(result) if result else None
    
    @classmethod
    async def get_artifact(cls, namespace: "PostgresNamespace", artifact_id: uuid.UUID, version: int | None = None) -> Optional[Dict[str, Any]]:
        """
        Get an artifact by artifact ID and version.
        """
        if version is not None:
            sql = f'SELECT * FROM "{namespace.table_name}" WHERE artifact_id = $1 AND version = $2;'
            values = [artifact_id, version]
        else:
            sql = f'SELECT DISTINCT ON (artifact_id) * FROM "{namespace.table_name}" WHERE artifact_id = $1 ORDER BY artifact_id, version DESC;'                
            values = [artifact_id]
        result = await PGConnectionManager.fetch_one(sql, *values)
        return dict(result) if result else None
    
    
    @classmethod
    async def query(
        cls, 
        namespace: "PostgresNamespace", 
        partition_id: int | None = None, 
        branch_id: int | None = None, 
        select: list[SelectType] | None = None,
        filters: Dict[str, Any] | None = None, 
        limit: int | None = None,
        order_by: str | None = None,
        offset: int | None = None,
        joins: list[JoinType] | None = None,
        filter_proxy: QueryProxy | None = None,
        include_reverted_turns: bool = False,
        is_event_source: bool = True,
        turn_limit: int | None = None,
        turn_direction: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Query records with versioning support.
        
        Args:
            namespace: The namespace to query
            partition_id: The partition ID to query from (optional)
            branch_id: The branch ID to query from (optional)
            filters: Filters to apply to the query
            limit: Maximum number of records to return
            order_by: Field and direction to order by (e.g., "created_at desc")
            offset: Number of records to skip
            joins: List of joins to apply to the query
            
            
        Returns:
            A list of records matching the query
        """
        
            # Regular query without versioning
        select_clause = "*"
        
        if select:
            select_parts = []
            for select_type in select:                                
                select_parts.append(', '.join([f"{select_type['namespace']}.{f}" for f in select_type['fields']]))
            select_clause = ",".join(select_parts)
        

        sql = f'SELECT {select_clause} FROM "{namespace.table_name}"'
        
        if joins:
            for join in joins:
                # sql += f" JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}"
                sql += f" JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}"
        
        # Add filters
        values = []
        if filters or filter_proxy:
            filters_sql = " WHERE "
            if filters:
                where_parts = []
                for key, value in filters.items():
                    where_parts.append(f'"{key}" = ${len(values) + 1}')
                    values.append(value)
                    
                filters_sql += ' AND '.join(where_parts)
            if filter_proxy: 
                filters_sql += " AND " if filters else ""           
                filters_sql += build_where_clause(filter_proxy)
            sql += filters_sql
        
        # Add order by
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        # Add limit
        if limit:
            sql += f" LIMIT {limit}"
        
        # Add offset
        if offset:
            sql += f" OFFSET {offset}"
                
                
        if namespace.is_versioned:
            filtered_alias = f"filtered_{namespace.table_name}"
            # sql = sql.replace(namespace.table_name, "filtered_records")
            sql = sql.replace(namespace.table_name, filtered_alias)
            # if not include_reverted_turns:
            #     reverted_turns_clause = f""" AND t.status != '{TurnStatus.REVERTED.value}'"""
            # else:
            #     reverted_turns_clause = ""
                
            # partition_clause = f"AND t.partition_id = {partition_id}" if partition_id else ""
            # event_source_select_clause = " DISTINCT ON (m.artifact_id)" if is_event_source else ""
            # event_source_order_by_clause = "AND m.deleted_at IS NULL ORDER BY m.artifact_id, m.version DESC" if is_event_source else ""
            
            # sql = f"""
            # WITH RECURSIVE branch_hierarchy AS (
            #     SELECT
            #         id,
            #         name,
            #         forked_from_turn_index,
            #         forked_from_branch_id,
            #         current_index AS start_turn_index
            #     FROM branches
            #     WHERE id = {branch_id}
                
            #     UNION ALL
                
            #     SELECT
            #         b.id,
            #         b.name,
            #         b.forked_from_turn_index,
            #         b.forked_from_branch_id,
            #         bh.forked_from_turn_index AS start_turn_index
            #     FROM branches b
            #     JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
            # ),
            # {filtered_alias} AS (
            #     SELECT{event_source_select_clause}
            #         m.*
            #     FROM branch_hierarchy bh
            #     JOIN turns t ON bh.id = t.branch_id
            #     JOIN "{namespace.table_name}" m ON t.id = m.turn_id
            #     WHERE t.index <= bh.start_turn_index {partition_clause}{reverted_turns_clause}
            #     {event_source_order_by_clause}
            # )
            # {sql}
            # """
            if turn_limit:
                turn_order_by_clause = f"ORDER BY t.index {turn_direction} LIMIT {turn_limit}"
            else:
                turn_order_by_clause = ""
            
            
            
            turn_where_clause = []
            if partition_id is not None:
                turn_where_clause.append(f"t.partition_id = {partition_id}")
            if is_event_source:
                turn_where_clause.append("m.deleted_at IS NULL")
            turn_where_clause = " AND ".join(turn_where_clause)
            
            event_source_select_clause = " DISTINCT ON (m.artifact_id)" if is_event_source else ""
            event_source_order_by_clause = "ORDER BY m.artifact_id, m.version DESC" if is_event_source else ""
            
            sql = f"""
            WITH RECURSIVE branch_hierarchy AS (
                SELECT
                    id,
                    name,
                    forked_from_turn_index,
                    forked_from_branch_id,
                    current_index AS start_turn_index
                FROM branches
                WHERE id = {branch_id}
                
                UNION ALL
                
                SELECT
                    b.id,
                    b.name,
                    b.forked_from_turn_index,
                    b.forked_from_branch_id,
                    bh.forked_from_turn_index AS start_turn_index
                FROM branches b
                JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
            ),
            turn_hierarchy AS (
                SELECT t.* 
                FROM branch_hierarchy bh
                JOIN turns t ON bh.id = t.branch_id
                WHERE t.index <= bh.start_turn_index AND t.status != 'reverted'
                {turn_order_by_clause}
            ),
            {filtered_alias} AS (
                SELECT{event_source_select_clause}
                    m.*
                FROM turn_hierarchy t               
                JOIN "{namespace.table_name}" m ON t.id = m.turn_id
                WHERE {turn_where_clause}
                {event_source_order_by_clause}
            )
            {sql}
            """
            
                
            # Execute query
        results = await cls.fetch(sql, *values)
        
        # Convert results to dictionaries
        return [dict(row) for row in results]

    
    
    @classmethod
    async def query2(
        cls, 
        namespace: "PostgresNamespace", 
        partition_id: int | None = None, 
        branch_id: int | None = None, 
        select: list[SelectType] | None = None,
        filters: Dict[str, Any] | None = None, 
        limit: int | None = None,
        order_by: str | None = None,
        offset: int | None = None,
        joins: list[JoinType] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Query records with versioning support.
        
        Args:
            namespace: The namespace to query
            partition_id: The partition ID to query from (optional)
            branch_id: The branch ID to query from (optional)
            filters: Filters to apply to the query
            limit: Maximum number of records to return
            order_by: Field and direction to order by (e.g., "created_at desc")
            offset: Number of records to skip
            
        Returns:
            A list of records matching the query
        """
        # If branch_id is provided, use versioning query
        if partition_id is not None and branch_id is None:
            branch_id = 1
        if branch_id is not None and namespace.is_versioned:
            # Get the branch
            # branch = await cls.get_branch(branch_id)
            
            # Build filter clause
            filter_clause = ""
            values = []
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    filter_parts.append(f'm."{key}" = ${len(values) + 1}')
                    values.append(value)
                
                if filter_parts:
                    filter_clause = f"WHERE {' AND '.join(filter_parts)}"
            
            # Build limit clause
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            # Build offset clause
            offset_clause = f"OFFSET {offset}" if offset else ""
            
            # Build order by clause
            order_by_clause = ""
            if order_by:
                order_by_clause = f"ORDER BY {order_by}"
            else:
                # Default ordering by id and turn_id
                order_by_clause = "ORDER BY id, turn_id DESC"
            
            partition_clause = f"AND t.partition_id = {partition_id}" if partition_id else ""
            
            # Build the recursive CTE query
            sql = f"""
            WITH RECURSIVE branch_hierarchy AS (
                SELECT
                    id,
                    name,
                    forked_from_turn_index,
                    forked_from_branch_id,
                    current_index AS start_turn_index
                FROM branches
                WHERE id = {branch_id}
                
                UNION ALL
                
                SELECT
                    b.id,
                    b.name,
                    b.forked_from_turn_index,
                    b.forked_from_branch_id,
                    bh.forked_from_turn_index AS start_turn_index
                FROM branches b
                JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
            ),
            filtered_records AS (
                SELECT
                    m.*
                FROM branch_hierarchy bh
                JOIN turns t ON bh.id = t.branch_id
                JOIN "{namespace.table_name}" m ON t.id = m.turn_id
                WHERE t.index <= bh.start_turn_index {partition_clause}
            )
            SELECT *
            FROM filtered_records
            {filter_clause}
            {order_by_clause}
            {limit_clause}
            {offset_clause}
            """
            
            # Execute query
            results = await PGConnectionManager.fetch(sql, *values)
            
            # Convert results to dictionaries
            return [dict(row) for row in results]
        else:
            # Regular query without versioning
            select_clause = "*"
            
            if select:
                select_parts = []
                for select_type in select:
                    select_parts.append(f"{select_type['namespace']}.{', '.join(select_type['fields'])}")
                select_clause = ",".join(select_parts)
            
            
            sql = f'SELECT {select_clause} FROM "{namespace.table_name}"'
            
            if joins:
                for join in joins:
                    # sql += f" JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}"
                    sql += f" JOIN {join['foreign_table']} ON {join['primary_table']}.{join['primary_key']} = {join['foreign_table']}.{join['foreign_key']}"
            
            # Add filters
            values = []
            if filters:
                where_parts = []
                for key, value in filters.items():
                    where_parts.append(f'"{key}" = ${len(values) + 1}')
                    values.append(value)
                    
                sql += f" WHERE {' AND '.join(where_parts)}"
            
            # Add order by
            if order_by:
                sql += f" ORDER BY {order_by}"
            
            # Add limit
            if limit:
                sql += f" LIMIT {limit}"
            
            # Add offset
            if offset:
                sql += f" OFFSET {offset}"
                
                
            
                
            # Execute query
            results = await cls.fetch(sql, *values)
            
            # Convert results to dictionaries
            return [dict(row) for row in results]
    
    
    
    
    
    async def execute_query(self, sql: str, *values: Any):
        """Execute a raw SQL query"""
        return await PGConnectionManager.fetch(sql, *values)
    



    