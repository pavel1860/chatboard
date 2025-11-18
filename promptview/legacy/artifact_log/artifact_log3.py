from contextvars import ContextVar
import json
import os
import asyncpg
import enum
from datetime import datetime, timezone
from typing import List, Optional, Type, Any, Dict, Tuple

from pydantic import BaseModel

from promptview.utils.db_connections import PGConnectionManager


"""
A framework for version-controlling Model instances using a git-like architecture.
Models can opt-in to versioning by setting versioned=True in their Config class.
This implementation uses asyncpg directly without SQLAlchemy.
"""


class TurnStatus(enum.Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    REVERTED = "reverted"




# class PGConnectionManager:
#     _pool: Optional[asyncpg.Pool] = None

#     @classmethod
#     async def initialize(cls, url: Optional[str] = None) -> None:
#         if cls._pool is None:
#             url = url or os.environ.get("POSTGRES_URL", "postgresql://snack:Aa123456@localhost:5432/promptview_test")
#             cls._pool = await asyncpg.create_pool(dsn=url)

#     @classmethod
#     async def close(cls) -> None:
#         if cls._pool:
#             await cls._pool.close()
#             cls._pool = None

#     @classmethod
#     async def execute(cls, query: str, *args) -> str:
#         if cls._pool is None:
#             await cls.initialize()
#         assert cls._pool is not None, "Pool must be initialized"
#         async with cls._pool.acquire() as conn:
#             return await conn.execute(query, *args)

#     @classmethod
#     async def fetch(cls, query: str, *args) -> List[asyncpg.Record]:
#         if cls._pool is None:
#             await cls.initialize()
#         assert cls._pool is not None, "Pool must be initialized"
#         async with cls._pool.acquire() as conn:
#             return await conn.fetch(query, *args)
        
        
#     @classmethod
#     async def fetch_one(cls, query: str, *args) -> asyncpg.Record:
#         if cls._pool is None:
#             await cls.initialize()
#         assert cls._pool is not None, "Pool must be initialized"
#         async with cls._pool.acquire() as conn:
#             return await conn.fetchrow(query, *args)
        



def model_to_table_name(model_cls: Type[Any]) -> str:
    """Utility function to get table name from a model class."""
    return model_cls.__name__.lower()


class Turn(BaseModel):
    id: int
    created_at: datetime
    ended_at: datetime | None = None    
    index: int
    status: TurnStatus    
    message: str | None = None
    score: int | None = None
    metadata: dict | None = None
    branch_id: int
    forked_branches: List[dict] | None = None
    local_state: dict | None = None
    
    
    def __init__(
        self, 
        forked_branches: List[dict] | str | None = None, 
        metadata: dict | str | None = None,
        **kwargs
    ):
        if forked_branches is None:
            forked_branches = []
        elif isinstance(forked_branches, str):
            forked_branches = json.loads(forked_branches)
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            metadata = json.loads(metadata)            
        
        super().__init__(**kwargs)
        self.forked_branches = forked_branches
        self.metadata = metadata
    

class Branch(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    forked_from_turn_index: int | None = None
    forked_from_branch_id: int | None = None
    last_turn: Turn | None = None
    


class Head(BaseModel):
    id: int
    main_branch_id: int
    branch_id: int
    turn_id: int
    created_at: datetime
    is_detached: bool



class TestCase(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    title: str
    description: str
    inputs: dict
    targets: dict
    branch_id: int
    turn_id: int


class TestRun(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    score: int
    message: str
    test_case_id: int
    branch_id: int
    turn_id: int


ARTIFCAT_LOG_CTX = ContextVar("ARTIFCAT_LOG_CTX", default=None)


class ArtifactLog:
    """
    Main class for managing versioned models and their artifacts using asyncpg.
    """
    def __init__(self, head_id: Optional[int] = None, branch_id: Optional[int] = None, turn_id: Optional[int] = None) -> None:        
        self._artifact_tables: Dict[str, str] = {}
        self._head: Optional[Dict[str, Any]] = None
        self._token = None
        self._head_id = head_id
        self._branch_id = branch_id
        self._turn_id = turn_id
        self._branch = None
        self._turn = None
        
    @classmethod
    def get_current(cls) -> "ArtifactLog":
        artifact_log = ARTIFCAT_LOG_CTX.get()
        if artifact_log is None:
            raise ValueError("No artifact log found")
        return artifact_log
    
    def set_context(self):
        self._token = ARTIFCAT_LOG_CTX.set(self)
        
    def reset_context(self):
        if self._token is not None:
            ARTIFCAT_LOG_CTX.reset(self._token)
            self._token=None
    
    
    async def __aenter__(self):
        if self._head_id is not None:
            self._head = await self.get_head(self._head_id)
        if self._branch_id is not None:
            await self.checkout_branch(self._branch_id, self._turn_id, update_db=False)
        if self._head is None:
            self._head = await self.create_head()
            
            
        # if self._head_id is not None:
        #     await self.init_head(self._head_id, self._branch_id)
        # elif self._branch_id is not None:
        #     await self.checkout_branch(self._branch_id, self._turn_id, update_db=False)
        self.set_context()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        self.reset_context()
        
    @property
    def is_initialized(self) -> bool:
        return self._head is not None
        # return self._head is not None and self._token is not None
    
    @property
    def is_in_context(self) -> bool:
        return self._token is not None

    async def initialize_tables(self) -> None:
        # Initialize the asyncpg pool if not already done
        await PGConnectionManager.initialize()
        # Create required extensions and tables
        await PGConnectionManager.execute("CREATE EXTENSION IF NOT EXISTS ltree;")
        
        
        await PGConnectionManager.execute("""                                          
CREATE TABLE IF NOT EXISTS branches (
	id SERIAL PRIMARY KEY,
	name TEXT,
	created_at TIMESTAMP DEFAULT NOW(),
	updated_at TIMESTAMP DEFAULT NOW(),
	forked_from_turn_index INTEGER,
	forked_from_branch_id INTEGER,
	FOREIGN KEY (forked_from_branch_id) REFERENCES branches(id)
);

CREATE TABLE IF NOT EXISTS turns (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP DEFAULT NOW(),
	ended_at TIMESTAMP,                
	index INTEGER NOT NULL,
	status TEXT NOT NULL,                
	message TEXT,
	score INTEGER,
	metadata JSONB DEFAULT '{}',
	branch_id INTEGER NOT NULL,
	FOREIGN KEY (branch_id) REFERENCES branches(id),
	local_state JSONB
);
							  
CREATE TABLE IF NOT EXISTS heads (
	id SERIAL PRIMARY KEY,
	created_at TIMESTAMP DEFAULT NOW(),
	updated_at TIMESTAMP DEFAULT NOW(),
	main_branch_id INTEGER,
	FOREIGN KEY (main_branch_id) REFERENCES branches(id),
	branch_id INTEGER,
	FOREIGN KEY (branch_id) REFERENCES branches(id),
	turn_id INTEGER,
	FOREIGN KEY (turn_id) REFERENCES turns(id)
);	
""")

    async def drop_tables(self, extra_tables: list[str] | None = None) -> None:
        # Drop all main tables and dynamic artifact tables
        await PGConnectionManager.execute("DROP TABLE IF EXISTS heads CASCADE;")
        await PGConnectionManager.execute("DROP TABLE IF EXISTS turns CASCADE;")
        await PGConnectionManager.execute("DROP TABLE IF EXISTS branches CASCADE;")
        if extra_tables:
            for table in extra_tables:
                await PGConnectionManager.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        # await PGConnectionManager.execute("DROP TABLE IF EXISTS base_artifacts CASCADE;")
        for table in self._artifact_tables.values():
            await PGConnectionManager.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        self._artifact_tables = {}

    async def register_model(self, model_cls: Type[Any]) -> str:
        """
        Register a versioned model and create its artifact table.
        """
        table_name = f"{model_to_table_name(model_cls)}_artifacts"
        if table_name in self._artifact_tables:
            return self._artifact_tables[table_name]

        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY,
                FOREIGN KEY (id) REFERENCES base_artifacts(id) ON DELETE CASCADE
            );
        """
        await PGConnectionManager.execute(query)
        self._artifact_tables[table_name] = table_name
        return table_name

    async def init_head(self, head_id: Optional[int] = None, branch_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Initialize a new head with a main branch and initial turn.
        If head_id is provided, load the existing head.
        """
        if head_id is None:
            head = await self.create_head()
            self._head = head
            return head
        else:
            self._head = await self.get_head(head_id)
            if branch_id is not None:
                await self.checkout_branch(branch_id)
            return self._head
            
            
    async def checkout_head(self, head_id: int) -> Dict[str, Any]:
        if not self.is_initialized:
            raise ValueError("Artifact log is not initialized")
        
        head = await PGConnectionManager.fetch_one(
            f"""
            UPDATE heads AS uh
            SET 
                main_branch_id = th.main_branch_id,
                branch_id      = th.branch_id,
                turn_id        = th.turn_id,
                updated_at     = NOW()
            FROM heads AS th
            WHERE uh.id = {self.head["id"]}
            AND th.id = {head_id}
            RETURNING uh.*;
            """
        )
        if head is None:
            raise ValueError("Head not found")
        self._head = dict(head)
        return self._head
          
            
    async def create_head(self, init_repo: bool = True) -> Dict[str, Any]:
        # Create head with null placeholders for branch_id and turn_id                
        
        if init_repo:
            branch = await self.create_branch()
            branch_id = branch.id
            turn_id = branch.last_turn.id
        else:
            branch_id = None
            turn_id = None
        
        query = f"INSERT INTO heads (branch_id, turn_id, main_branch_id) VALUES ($1, $2, $3) RETURNING *;"
        head_rows = await PGConnectionManager.fetch_one(query, branch_id, turn_id, branch_id)
        if head_rows is None:
            raise ValueError("Failed to create head")
        new_head = dict(head_rows)
        return {
            "is_detached": False if branch_id is not None else True, 
            **new_head
        }
        
        
    
        
        
    
    
    async def update_head(self, branch_id: int | None = None, turn_id: int | None = None, main_branch_id: int | None = None):
        if self._head is None:
            raise ValueError("No active head found")
        update_values = {}
        if branch_id is not None:
            update_values["branch_id"] = branch_id
        if turn_id is not None:
            update_values["turn_id"] = turn_id
        if main_branch_id is not None:
            update_values["main_branch_id"] = main_branch_id
        query = "UPDATE heads SET " + ", ".join([f"{k}=${i + 1}" for i,k in enumerate(update_values.keys())]) + " WHERE id=$" + str(len(update_values) + 1) + " RETURNING *;"
        res = await PGConnectionManager.fetch_one(query, *update_values.values(), self.head["id"])        
        return dict(res)
        

    @property
    def head(self) -> Dict[str, Any]:
        if self._head is None:
            raise ValueError("No active head found")
        return self._head
    
    
    def get_upsert_values(self, model_type: Type[Any]):
        """
        type, turn_id
        """



    async def commit_turn(self, message: Optional[str] = None) -> int:
        """
        Commit the current turn and create a new one.
        """
        if self._head is None:
            raise ValueError("No active head found")

        head = self._head
        # Convert aware datetime to naive by removing tzinfo
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        query = "UPDATE turns SET status = $1, ended_at = $2, message = $3 WHERE id = $4 RETURNING branch_id, index;"
        rows = await PGConnectionManager.fetch(query, TurnStatus.COMMITTED.value, now, message, head['turn_id'])
        branch_id = rows[0]['branch_id']
        current_index = rows[0]['index']

        new_index = current_index + 1
        query = "INSERT INTO turns (branch_id, index, status) VALUES ($1, $2, $3) RETURNING id;"
        new_turn_rows = await PGConnectionManager.fetch(query, branch_id, new_index, TurnStatus.STAGED.value)
        new_turn_id = new_turn_rows[0]['id']

        query = "UPDATE heads SET turn_id = $1 WHERE id = $2;"
        await PGConnectionManager.execute(query, new_turn_id, head['id'])
        head['turn_id'] = new_turn_id
        return new_turn_id


    async def branch_from(self, turn_id: int, name: Optional[str] = None, check_out: bool = False) -> int:
        """
        Create a new branch from a specific turn.
        """
        # query = "SELECT * FROM turns WHERE id = $1;"
        # rows = await PGConnectionManager.fetch(query, turn_id)
        # if not rows:
        #     raise ValueError(f"Turn {turn_id} not found")
        # source_turn = dict(rows[0])
        source_turn = await self.get_turn(turn_id)

        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")

        new_branch_name = name if name is not None else f"branch_from_{turn_id}"

        branch = await self.create_branch(turn_id=source_turn.id, name=new_branch_name)
        new_branch_id = branch.id
        
        # query = "INSERT INTO branches (name, forked_from_turn_index, forked_from_branch_id) VALUES ($1, $2, $3) RETURNING id;"
        # new_branch_rows = await PGConnectionManager.fetch(query, new_branch_name, source_turn['index'], source_turn['branch_id'])
        # new_branch_id = new_branch_rows[0]['id']

        if check_out:
            await self.checkout_branch(new_branch_id)

        return new_branch_id
    
    
    async def create_branch(self, turn_id: int | None = None, name: str | None = None):
        if name is None:
            if turn_id is None:
                name = "main"
            else:
                name = f"branch_from_{turn_id}"
        turn = None    
        if turn_id is not None:
            turn = await self.get_turn(turn_id)
            turn_index = turn.index
            turn_branch_id = turn.branch_id
        else:
            turn_index = 1
            turn_branch_id = None
        
        query = "INSERT INTO branches (name, forked_from_turn_index, forked_from_branch_id) VALUES ($1, $2, $3) RETURNING *;"
        new_branch_rows = await PGConnectionManager.fetch(query, name, turn_index, turn_branch_id)
        if not new_branch_rows:
            raise ValueError("Failed to create branch")
        branch = new_branch_rows[0]
        turn = await self.create_turn(branch['id'], turn.index + 1 if turn is not None else 1, TurnStatus.STAGED)        
        branch = Branch(**branch, last_turn=turn)
        return branch
        
        
    
    
    async def checkout_branch(self, branch_id: int, turn_id: int | None = None, update_db: bool = True):
        # if self._head is None:
            # raise ValueError("Artifact log is not initialized")
        if turn_id is None:
            branch = await self.get_branch(branch_id=branch_id)
            new_head = {
                "id": self._head["id"] if self._head is not None else None,
                "main_branch_id": self._head["main_branch_id"] if self._head is not None else None,
                "is_detached": False,
                "branch_id": branch.id,
                "turn_id": branch.last_turn.id
            }
            turn = branch.last_turn
        else:
            branch, selected_turn, is_detached = await self.get_branch_with_turn(branch_id=branch_id, turn_id=turn_id)
            new_head = {
                "id": self._head["id"] if self._head is not None else None,
                "main_branch_id": self._head["main_branch_id"] if self._head is not None else None,
                "is_detached": is_detached,
                "branch_id": branch.id,
                "turn_id": selected_turn.id
            }
            turn = selected_turn
        
        self._branch = branch
        self._turn = turn
        self._head = new_head
        if update_db:
            await self.update_head(branch_id=new_head["branch_id"], turn_id=new_head["turn_id"])
            
        return new_head
    
    
    @classmethod
    async def get_head(cls, head_id: int) -> Dict[str, Any]:
        query = "SELECT * FROM heads WHERE id = $1;"
        res = await PGConnectionManager.fetch_one(query, head_id)
        if res is None:
            raise ValueError(f"Head {head_id} not found")
        return dict(res)
    
    
    async def delete_head(self, head_id: int):
        query = "DELETE FROM heads WHERE id = $1;"
        return await PGConnectionManager.execute(query, head_id)
    
    
    async def get_branch_with_turn(self, branch_id: int, turn_id: int):
        res = await self._get_branch_with_turn(branch_id, turn_id)
        branch = Branch(**res["branch"], last_turn=Turn(**res["last_turn"]))
        selected_turn = Turn(**res["selected_turn"])
        return branch, selected_turn, res["is_detached"]
    
    async def get_branch(self, branch_id: int):
        res = await self._get_branch_with_turn(branch_id)
        return Branch(**res["branch"], last_turn=Turn(**res["last_turn"]))
        
    async def _get_branch_with_turn(self, branch_id: int, turn_id: int | None = None):    
        query = f"""
            SELECT 
                (
                    SELECT to_json(b) FROM branches b WHERE b.id = {branch_id}
                ) AS branch,
                (
                    SELECT to_json(t)
                    FROM turns t
                    WHERE t.branch_id = {branch_id}
                    ORDER BY t.created_at DESC
                    LIMIT 1
                ) AS last_turn               
        """
        if turn_id is not None:
            query += ","
            query += f"""
             (
                SELECT to_json(t)
                FROM turns t
                WHERE t.branch_id = {branch_id} AND t.id = {turn_id}
                ORDER BY t.created_at DESC
                LIMIT 1
            ) AS selected_turn
            """
        query += ";"        
        res = await PGConnectionManager.fetch_one(query)
        if res is None:
            raise ValueError("No active head found")
        last_turn = res.get('last_turn')
        if last_turn is None:
            raise ValueError("No last turn found")
        last_turn = json.loads(last_turn)
        branch = res.get('branch')
        if branch is None:
            raise ValueError("No branch found")
        branch = json.loads(branch)   
        
        result = {
            "branch": branch, 
            "last_turn": last_turn, 
            "is_detached": False
        }
        
        if turn_id is not None:
            selected_turn = res.get('selected_turn')
            if selected_turn is None:
                raise ValueError("No selected turn found on branch")
            selected_turn = json.loads(selected_turn)
            if last_turn["id"] != selected_turn["id"]:
                result["is_detached"] = True
            result["selected_turn"] = selected_turn
        return result
     

    # async def checkout_branch2(self, branch_id: int, turn_id: int | None = None) -> None:
    #     """
    #     Switch HEAD to a different branch.
    #     """
    #     query = "SELECT * FROM branches WHERE id = $1;"
    #     rows = await PGConnectionManager.fetch(query, branch_id)
    #     if not rows:
    #         raise ValueError(f"Branch {branch_id} not found")

    #     current_head = self.head
    #     if not current_head:
    #         raise ValueError("No active head found")

    #     query = "SELECT * FROM turns WHERE branch_id = $1 AND status = $2 ORDER BY index DESC LIMIT 1;"
    #     turn_rows = await PGConnectionManager.fetch(query, branch_id, TurnStatus.STAGED.value)
    #     if turn_rows:
    #         turn_id = turn_rows[0]['id']
    #     else:
    #         query = "INSERT INTO turns (branch_id, index, status) VALUES ($1, $2, $3) RETURNING id;"
    #         turn_row = await PGConnectionManager.fetch(query, branch_id, 1, TurnStatus.STAGED.value)
    #         turn_id = turn_row[0]['id']

    #     query = "UPDATE heads SET branch_id = $1, turn_id = $2 WHERE id = $3;"
    #     await PGConnectionManager.execute(query, branch_id, turn_id, current_head['id'])
    #     current_head['branch_id'] = branch_id
    #     current_head['turn_id'] = turn_id

    async def revert_turn(self) -> int:
        """
        Revert the current turn and create a new one.
        """
        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")

        now = datetime.utcnow()
        query = "UPDATE turns SET status = $1, ended_at = $2 WHERE id = $3;"
        await PGConnectionManager.execute(query, TurnStatus.REVERTED.value, now, current_head['turn_id'])

        query = "SELECT branch_id, index FROM turns WHERE id = $1;"
        row = await PGConnectionManager.fetch(query, current_head['turn_id'])
        branch_id = row[0]['branch_id']
        new_index = row[0]['index'] + 1

        query = "INSERT INTO turns (branch_id, index, status) VALUES ($1, $2, $3) RETURNING id;"
        new_turn_row = await PGConnectionManager.fetch(query, branch_id, new_index, TurnStatus.STAGED.value)
        new_turn_id = new_turn_row[0]['id']

        query = "UPDATE heads SET turn_id = $1 WHERE id = $2;"
        await PGConnectionManager.execute(query, new_turn_id, current_head['id'])
        current_head['turn_id'] = new_turn_id
        return new_turn_id

    async def revert_to_turn(self, turn_id: int) -> int:
        """
        Revert to a specific turn, marking all later turns as reverted and creating a new turn.
        """
        query = "SELECT * FROM turns WHERE id = $1;"
        rows = await PGConnectionManager.fetch(query, turn_id)
        if not rows:
            raise ValueError(f"Turn {turn_id} not found")
        target_turn = dict(rows[0])

        current_head = self.head
        if not current_head:
            raise ValueError("No active head found")

        query = "UPDATE turns SET status = $1, ended_at = $2 WHERE branch_id = $3 AND index > $4 AND status != $5;"
        now = datetime.utcnow()
        await PGConnectionManager.execute(query, TurnStatus.REVERTED.value, now, target_turn['branch_id'], target_turn['index'], TurnStatus.REVERTED.value)

        new_index = target_turn['index'] + 1
        query = "INSERT INTO turns (branch_id, index, status) VALUES ($1, $2, $3) RETURNING id;"
        new_turn_row = await PGConnectionManager.fetch(query, target_turn['branch_id'], new_index, TurnStatus.STAGED.value)
        new_turn_id = new_turn_row[0]['id']

        query = "UPDATE heads SET branch_id = $1, turn_id = $2 WHERE id = $3;"
        await PGConnectionManager.execute(query, target_turn['branch_id'], new_turn_id, current_head['id'])
        current_head['branch_id'] = target_turn['branch_id']
        current_head['turn_id'] = new_turn_id
        return new_turn_id


    async def get_turn(self, turn_id: int) -> Turn:
        query = "SELECT * FROM turns WHERE id = $1;"
        rows = await PGConnectionManager.fetch(query, turn_id)
        if not rows:
            raise ValueError(f"Turn {turn_id} not found")
        return Turn(**dict(rows[0]))
    
    
    async def create_turn(self, branch_id: int, index: int, status: TurnStatus):
        query = "INSERT INTO turns (branch_id, index, status) VALUES ($1, $2, $3) RETURNING *;"
        new_turn_row = await PGConnectionManager.fetch(query, branch_id, index, status.value)
        if not new_turn_row:
            raise ValueError("Failed to create turn")
        return Turn(**dict(new_turn_row[0]))

    # async def get_branch(self, branch_id: int) -> Dict[str, Any]:
    #     query = "SELECT * FROM branches WHERE id = $1;"
    #     rows = await PGConnectionManager.fetch(query, branch_id)
    #     if not rows:
    #         raise ValueError(f"Branch {branch_id} not found")
    #     return dict(rows[0])
    
    
    
    
    
    async def get_artifact_list(self, artifact_table: str, limit: int = 10, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[dict]:
        
        turn = await self.get_turn(self.head["turn_id"])
        query = self._artifact_cte_query(
            turn,
            artifact_table=artifact_table,
            order_by=order_by,
            order_direction=order_direction,
            limit=limit,
            offset=offset
        )
        rows = await PGConnectionManager.fetch(query)        
        return [dict(row) for row in rows]
    
    
    
    async def get_all_turns(self) -> List[Turn]:
        query = """
        SELECT 
            t.*
        FROM heads h
        JOIN branches b ON b.head_id=h.id
        JOIN turns t ON t.branch_id=b.id
        """
        rows = await PGConnectionManager.fetch(query)
        return [Turn(**dict(row)) for row in rows]
    
    @classmethod
    async def get_head_list(cls) -> List[Head]:
        query = """
            SELECT * FROM heads
        """
        rows = await PGConnectionManager.fetch(query)
        return [Head(**dict(row)) for row in rows]
    
    
    async def get_branch_turns(self, branch_id: int, limit: int = 10, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[Turn]:
        query = f"""
            SELECT 
                t.id,
                t.branch_id,
                t.index,
                t.status,
                t.created_at,
                t.ended_at,
                t.message,
                t.local_state,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'id', b.id,
                            'name', b.name,
                            'forked_from_turn_index', b.forked_from_turn_index,
                            'forked_from_branch_id', b.forked_from_branch_id,
                            'created_at', b.created_at,
                            'updated_at', b.updated_at
                        ) ORDER BY b.created_at  -- ordering the aggregated forked branches by creation time
                    ) FILTER (WHERE b.id IS NOT NULL),
                    '[]'
                ) AS forked_branches
            FROM turns t
            LEFT JOIN branches b 
                ON b.forked_from_branch_id = t.branch_id 
            AND b.forked_from_turn_index = t.index
            WHERE t.branch_id = {branch_id}  -- Parameter for the specific branch_id
            GROUP BY t.id
            ORDER BY t.index ASC     -- Ordering the turns by their index
            LIMIT {limit} OFFSET {offset};      -- Limit and offset parameters

        """
        rows = await PGConnectionManager.fetch(query)
        return [Turn(**dict(row)) for row in rows]
    
    
    async def get_turn_list(self, limit: int = 10, status: Optional[TurnStatus] = None, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[Turn]:
        turn = await self.get_turn(self.head["turn_id"])
        query = self._turn_cte_query(
            turn,
            order_by=f"t.{order_by}",
            order_direction=order_direction,
            limit=limit,
            offset=offset,
            filter_query=f"t.status = '{status.value}'" if status else ""
        )
        rows = await PGConnectionManager.fetch(query)        
        return [Turn(**dict(row)) for row in rows]
    
    
    async def get_branch_list(self, limit: int = 10, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[Branch]:
        turn = await self.get_turn(self.head["turn_id"])
        query = self._branch_cte_query(turn, select_query=f"SELECT * FROM branch_hierarchy ORDER BY {order_by} {order_direction} LIMIT {limit} OFFSET {offset}", minimal_fields=False)
        rows = await PGConnectionManager.fetch(query)
        return [Branch(**dict(row)) for row in rows]
    
    async def execute_query(self, query: str) -> List[dict]:
        rows = await PGConnectionManager.fetch(query)        
        return [dict(row) for row in rows]
    
    def _branch_cte_query(self, turn: Turn, select_query: str="SELECT * FROM branch_hierarchy", minimal_fields: bool=True) -> str:
        if minimal_fields:
            fields_base = """
                id,
                name,
                forked_from_turn_index,
                forked_from_branch_id,
            """
            fields_step = """
                b.id,
                b.name,
                b.forked_from_turn_index,
                b.forked_from_branch_id,
            """
        else:
            fields_base = """
                *,
            """
            fields_step = """
                b.*,
            """
                
        query = f"""
        WITH RECURSIVE branch_hierarchy AS (
            SELECT 
                {fields_base}
                {turn.index} as start_turn_index
            FROM branches
            WHERE id={turn.branch_id}

            UNION ALL

            SELECT
                {fields_step}
                bh.forked_from_turn_index as start_turn_index
            FROM branches b
            JOIN branch_hierarchy bh ON b.id=bh.forked_from_branch_id
        )
        {select_query}     
        """
        return query
    
    def _turn_cte_query(
        self, 
        turn: Turn,
        select_query: str="t.*",
        filter_query: str="", 
        join_query: str="", 
        order_by: str="t.created_at", 
        order_direction: str="DESC", 
        limit: int=10, 
        offset: int=0
    ):
        
        turn_select_query = f"""
        SELECT 
            {select_query}
        FROM branch_hierarchy bh 
        LEFT JOIN turns t ON bh.id = t.branch_id
        {join_query}
        WHERE t.index <= bh.start_turn_index
        {"AND " + filter_query if filter_query else ""}
        ORDER BY {order_by} {order_direction}
        LIMIT {limit} OFFSET {offset};
        """
        return self._branch_cte_query(turn, select_query=turn_select_query)
    
    
    
    def _artifact_cte_query(
        self,
        turn: Turn,
        artifact_table: str,
        filter_query: str="",
        order_by: str="m.created_at",
        order_direction: str="DESC",
        limit: int=10,
        offset: int=0
    ) -> str:
        query = self._turn_cte_query(
            turn, 
            select_query=f"m.*", 
            join_query=f"RIGHT JOIN {artifact_table} m on t.id=m.turn_id", 
            filter_query=filter_query,
            order_by=order_by,
            order_direction=order_direction,
            limit=limit,
            offset=offset
        )
        return query
    
    
    
    
    async def artifact_cte_raw_query(
        self, 
        artifact_table: str,
        artifact_query: str,        
    ):
        
        turn = await self.get_turn(self.head["turn_id"])
        artifact_query = artifact_query.replace(artifact_table, "filtered_artifacts")
        query = f"""
        WITH RECURSIVE branch_hierarchy AS (
            SELECT 
                id,
                name,
                forked_from_turn_index,
                forked_from_branch_id,
                {turn.index} AS start_turn_index
            FROM branches
            WHERE id = {turn.branch_id}

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
        filtered_artifacts AS (
            SELECT 
                m.*
            FROM branch_hierarchy bh 
            JOIN turns t ON bh.id = t.branch_id
            JOIN "{artifact_table}" m ON t.id = m.turn_id
            WHERE t.index <= bh.start_turn_index
        )
        {artifact_query}
        """
        res = await PGConnectionManager.fetch(query)
        return [dict(row) for row in res]