import enum
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Generic, List, Optional, Any, Self, Type, Union, final
from typing_extensions import TypeVar
from pydantic import BaseModel, Field


from promptview.utils.db_connections import PGConnectionManager
if TYPE_CHECKING:
    from promptview.model.postgres.namespace import PostgresNamespace, PgFieldInfo





class TurnStatus(str, enum.Enum):
    """Status of a turn"""
    STAGED = "staged"
    COMMITTED = "committed"
    REVERTED = "reverted"


class Turn(BaseModel):
    """A turn represents a point in time in a branch"""
    id: int
    created_at: datetime
    ended_at: Optional[datetime] = None
    index: int
    status: TurnStatus
    message: Optional[str] = None
    branch_id: int
    trace_id: Optional[str] = None
    state: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    forked_branches: List[dict] | None = None
    
    def __init__(
        self, 
        metadata: Union[Dict[str, Any], str, None] = None,
        **kwargs
    ):
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        super().__init__(metadata=metadata, **kwargs)


class Branch(BaseModel):
    """A branch represents a line of development"""
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    forked_from_turn_index: Optional[int] = None
    forked_from_branch_id: Optional[int] = None


# We don't need the Repo class anymore since we're using branch_id directly

class Partition(BaseModel):
    """A partition represents a group of turns"""
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    participants: List["PartitionParticipant"]
    
    def is_participant(self, user_id: int) -> bool:
        return any(participant.user_id == user_id for participant in self.participants)
    
    
    
class PartitionParticipant(BaseModel):
    """A participant in a partition"""
    id: int
    user_id: int
    





@final
class DontUpdateType:
    """A type used as a sentinel for undefined values."""

    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any) -> Self: ...

DontUpdate: DontUpdateType = DontUpdateType()


TURN_MODEL = TypeVar("TURN_MODEL", bound=Turn)

class ArtifactLog(Generic[TURN_MODEL]):
    
    _turn_model: Optional[Type[TURN_MODEL]] = Turn
    
    
    @classmethod
    def register_turn_model(cls, turn_model: Type[TURN_MODEL]):
        cls._turn_model = turn_model
        
        
    @classmethod
    def get_extra_turn_fields(cls) -> "list[PgFieldInfo]":
        """Get the extra fields for the turn model"""
        from promptview.model.postgres.namespace import PgFieldInfo
        if cls._turn_model is None:
            cls._turn_model = Turn
            return []
        base_fields = dict(Turn.model_fields.items())
        extra_fields = []
        for field, field_info in cls._turn_model.model_fields.items():            
            if not field_info.annotation:
                raise ValueError(f"Field {field} has no annotation")            
            if field in base_fields:
                bf = base_fields[field]
                if field_info.annotation != bf.annotation:
                    raise ValueError(f"Field {field} has a different annotation in the turn model")
                continue            
            extra_json = field_info.json_schema_extra
            extra: Dict[str, Any] = {}
            if extra_json is not None:
                if isinstance(extra_json, dict):
                    extra = dict(extra_json)
            pg_field = PgFieldInfo(field, field_info.annotation,  extra)
            extra_fields.append(pg_field)
        return extra_fields

                
            
    
    @classmethod
    async def initialize_versioning(cls):
        """Initialize versioning tables"""
        await PGConnectionManager.initialize()
        # Create required tables
        await PGConnectionManager.execute("""
        CREATE TABLE IF NOT EXISTS branches (
            id SERIAL PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            forked_from_turn_index INTEGER,
            forked_from_branch_id INTEGER,
            current_index INTEGER DEFAULT 0,
            FOREIGN KEY (forked_from_branch_id) REFERENCES branches(id)
        );

        CREATE TABLE IF NOT EXISTS turns (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW(),
            ended_at TIMESTAMP,
            index INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            state JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}',            
            branch_id INTEGER NOT NULL,
            trace_id TEXT,
            FOREIGN KEY (branch_id) REFERENCES branches(id)
        );
        
        
        
                
        CREATE INDEX IF NOT EXISTS idx_turns_branch_id ON turns (branch_id);
        CREATE INDEX IF NOT EXISTS idx_turns_index ON turns (index DESC);
        """)

    
    @classmethod
    async def add_partition_id_to_turns(cls, partition_table: str, key: str):
        """Add a partition_id column to the table"""
        await PGConnectionManager.execute(f"""
        ALTER TABLE turns ADD COLUMN IF NOT EXISTS partition_id INTEGER NOT NULL REFERENCES "{partition_table}" ({key}) ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS idx_turns_partition_id ON turns (partition_id);
        """)
        
        
    @classmethod
    async def add_field_to_table(cls, table: str, field: str, type: str, default: str | None = None, index: bool = False):
        sql = f"""
        ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {field} {type}
        """
        if default is not None:
            sql += f" DEFAULT {default}"
        
        sql += ";"
        if index:
            sql += f"CREATE INDEX IF NOT EXISTS idx_{table}_{field} ON {table} ({field});"
        await PGConnectionManager.execute(sql)
    
    # @classmethod
    # async def create_partition(cls, name: str, participants: List[int]) -> Partition:
    #     """Create a new partition and associated partition participant record in a single transaction"""
    #     query = "INSERT INTO partitions (name) VALUES ($1) RETURNING *;"
    #     partition_row = await PGConnectionManager.fetch_one(query, name)
    #     if partition_row is None:
    #         raise ValueError("Failed to create partition and participant")
    #     for participant in participants:
    #         await cls.add_participant_to_partition(partition_row["id"], participant)
    #     partition_row = await cls.get_partition(partition_row["id"])
    #     return Partition(**dict(partition_row))
    
    # @classmethod
    # async def add_participant_to_partition(cls, partition_id: int, user_id: int) -> PartitionParticipant:
    #     """Add a participant to a partition"""
    #     query = "INSERT INTO partition_participants (partition_id, user_id) VALUES ($1, $2) RETURNING *;"
    #     participant_row = await PGConnectionManager.fetch_one(query, partition_id, user_id)
    #     return PartitionParticipant(**dict(participant_row))
    
    # @classmethod
    # def _pack_participant(cls, record: Any) -> Partition:
    #     partition_row = dict(record)
    #     partition_row["participants"] = json.loads(partition_row["participants"])
    #     return Partition(**partition_row)
    
    # @classmethod
    # async def get_partition(cls, partition_id: int) -> Partition:
    #     """Get a partition by ID"""
    #     query = """
    #     SELECT 
    #         p.*,
    #         json_agg(
    #             json_build_object(
    #                 'id', pp.id,
    #                 'user_id', pp.user_id
    #             )
    #         ) AS participants
    #     FROM partitions p
    #     LEFT JOIN partition_participants pp ON p.id = pp.partition_id
    #     WHERE p.id = $1
    #     GROUP BY p.id, p.name, p.created_at, p.updated_at;
    #     """
    #     partition_row = await PGConnectionManager.fetch_one(query, partition_id)
    #     return cls._pack_participant(partition_row)
    
    # @classmethod
    # async def list_partitions(cls, user_id: int, name: str | None = None, limit: int = 100, offset: int = 0) -> List[Partition]:
    #     """Get all partitions for a user"""

    #     query = f"""
    #     SELECT 
    #         p.*,
    #         json_agg(
    #             json_build_object(
    #                 'id', pp.id,
    #                 'user_id', pp.user_id
    #             )
    #         ) AS participants
    #     FROM partitions p
    #     LEFT JOIN partition_participants pp ON p.id = pp.partition_id
    #     WHERE pp.user_id = $1 {f"AND p.name = $4" if name is not None else ""}
    #     GROUP BY p.id, p.name, p.created_at, p.updated_at
    #     ORDER BY p.created_at DESC
    #     LIMIT $2 OFFSET $3;
    #     """
    #     params = [user_id, limit, offset]
    #     if name is not None:
    #         params.append(name)
    #     partitions = await PGConnectionManager.fetch(query, *params)
    #     return [cls._pack_participant(partition) for partition in partitions]
    
    # @classmethod
    # async def last_partition(cls, user_id: int, name: str | None = None) -> Partition | None:
    #     """Get the last partition for a user"""
    #     partitions = await cls.list_partitions(user_id, name, limit=1, offset=0)
    #     if len(partitions) == 0:
    #         return None
    #         # raise ValueError("No partitions found for user")
    #     return partitions[0]

    
    @classmethod
    async def create_branch(cls, name: Optional[str] = None, forked_from_turn_id: Optional[int] = None) -> Branch:
        """Create a new branch"""
        # Initialize versioning if needed
        # await cls.initialize_versioning()
        
        turn = None
        if forked_from_turn_id is not None:
            turn = await cls.get_turn(forked_from_turn_id)
            turn_index = turn.index
            turn_branch_id = turn.branch_id
        else:
            turn_index = 1
            turn_branch_id = None
        
        if name is None:
            if forked_from_turn_id is None:
                name = "main"
            else:
                name = f"branch_from_{forked_from_turn_id}"
        
        # Create the branch
        query = "INSERT INTO branches (name, forked_from_turn_index, forked_from_branch_id) VALUES ($1, $2, $3) RETURNING *;"
        branch_row = await PGConnectionManager.fetch_one(query, name, turn_index, turn_branch_id)
        if branch_row is None:
            raise ValueError("Failed to create branch")
        
        
        return Branch(**dict(branch_row))
    
    @classmethod
    def _pack_turn(cls, record: Any) -> TURN_MODEL:
        data = dict(record)
        data["state"] = json.loads(data["state"]) if data["state"] is not None else None
        data["metadata"] = json.loads(data["metadata"])
        data["forked_branches"] = json.loads(data["forked_branches"]) if "forked_branches" in data else None
        if cls._turn_model is None:
            cls.register_turn_model(Turn)
        return cls._turn_model(**data)
    
    
    @classmethod
    async def create_turn(cls, branch_id: int = 1, status: TurnStatus=TurnStatus.STAGED, state: Optional[Any] = None, **kwargs) -> Turn:
        """
        Create a new turn
        
        Args:
            partition_id: The partition ID
            branch_id: The branch ID
            status: The status of the turn
        """
    
        
        query = f"""
        WITH updated_branch AS (
            UPDATE branches
            SET current_index = current_index + 1
            WHERE id = $1
            RETURNING id, current_index
        ),
        new_turn AS (
            INSERT INTO turns (branch_id, index, status, state{"".join([", " + k for k in kwargs.keys()])})
            SELECT id, current_index, $2, $3{"".join([", $" + str(i) for i in range(4, len(kwargs) + 4)])}
            FROM updated_branch
            RETURNING *
        )
        SELECT * FROM new_turn;
        """
        state_json = "{}"
        if state is not None:
            if isinstance(state, BaseModel):
                state_json = state.model_dump_json()
            else:
                state_json = json.dumps(state)
        # Use a transaction to ensure atomicity
        async with PGConnectionManager.transaction() as tx:
            turn_row = await tx.fetch_one(query, branch_id, status.value, state_json, *[kwargs[k] for k in kwargs.keys()])
            if turn_row is None:
                raise ValueError(f"Failed to create turn for branch {branch_id}")
        
        return cls._pack_turn(turn_row)
    
    

    
    @classmethod
    async def get_branch_turns(cls, branch_id: int, limit: int = 10, offset: int = 0, order_by: str = "index", order_direction: str = "DESC") -> List[Turn]:
        query = f"""
            SELECT 
                t.id,
                t.branch_id,
                t.index,
                t.status,
                t.created_at,
                t.ended_at,
                t.message,
                t.metadata,
                t.state,
                t.trace_id,
                t.partition_id,
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
            ORDER BY t.{order_by} {order_direction}     -- Ordering the turns by their index
            LIMIT {limit} OFFSET {offset};      -- Limit and offset parameters            
        """
        rows = await PGConnectionManager.fetch(query)
        # turns = []
        # for row in rows:
        #     data = dict(row)
        #     data["user_context"] = UserContext(**json.loads(data["user_context"])) if data["user_context"] is not None else None
        #     data["forked_branches"] = json.loads(data["forked_branches"])
        #     data["metadata"] = json.loads(data["metadata"])
        #     turns.append(Turn(**data))
        return [cls._pack_turn(row) for row in reversed(rows)]
    
    @classmethod
    def build_turn_query(cls, branch_id: int | None = None, filters: dict[str, Any] = {}, limit: int = 10, offset: int = 0, order_by: str = "index", order_direction: str = "DESC", ) -> str:
        query = f"""
        SELECT * FROM turns
        """
        where_clause = []
        if branch_id is not None:
            where_clause.append(f"branch_id = {branch_id}")
        if filters:
            where_clause.append(" AND ".join([f"{k} = {v}" for k, v in filters.items()]))
        if where_clause:
            query += " WHERE " + " AND ".join(where_clause)
        query += f" ORDER BY {order_by} {order_direction} LIMIT {limit} OFFSET {offset};"
        return query
    
    @classmethod
    async def get_turn(cls, turn_id: int) -> Turn:
        """Get a turn by ID"""
        query = "SELECT * FROM turns WHERE id = $1;"
        turn_row = await PGConnectionManager.fetch_one(query, turn_id)
        if turn_row is None:
            raise ValueError(f"Turn {turn_id} not found")
        
        return cls._pack_turn(turn_row)
        
    
    @classmethod
    async def get_last_turn(cls, branch_id: int, filters: dict[str, Any] = {}) -> Turn | None:
        """Get the last turn for a partition and branch"""
        query = cls.build_turn_query(branch_id=branch_id, filters=filters, limit=1, offset=0, order_by="index", order_direction="DESC")
        turn_row = await PGConnectionManager.fetch_one(query)
        return cls._pack_turn(turn_row) if turn_row is not None else None
    
    @classmethod
    async def list_turns(cls, branch_id: int | None = None, filters: dict[str, Any] = {}, limit: int = 100, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> List[Turn]:
        query = cls.build_turn_query(branch_id=branch_id, filters=filters, limit=limit, offset=offset, order_by=order_by, order_direction=order_direction)
        turns = await PGConnectionManager.fetch(query)
        return [cls._pack_turn(turn) for turn in turns]
    
    
    @classmethod
    async def get_branch_or_none(cls, branch_id: int) -> Branch | None:
        """Get a branch by ID"""
        # Get the branch
        query = "SELECT * FROM branches WHERE id = $1;"
        branch_row = await PGConnectionManager.fetch_one(query, branch_id)
        if branch_row is None:
            return None
        return Branch(**dict(branch_row))
    
    
    
    @classmethod
    async def get_branch(cls, branch_id: int) -> Branch:
        """Get a branch by ID"""
        # Get the branch
        branch = await cls.get_branch_or_none(branch_id)
        if branch is None:
            raise ValueError(f"Branch {branch_id} not found")        
        return branch  
    
    
    
    @classmethod
    async def list_branches(cls, limit: int = 100, offset: int = 0, order_by: str = "created_at", order_direction: str = "DESC") -> list[Branch]:
        """List all branches"""
        query = f"SELECT * FROM branches ORDER BY {order_by} {order_direction} LIMIT $1 OFFSET $2;"
        branches = await PGConnectionManager.fetch(query, limit, offset)
        return [Branch(**dict(branch)) for branch in branches]
    
    @classmethod
    async def commit_turn(cls, turn_id: int, message: Optional[str] = None, trace_id: Optional[str] = None) -> Turn:
        """Commit the current turn and create a new one"""        
        # Update the turn
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        query = """
        UPDATE turns
        SET status = $1, ended_at = $2, message = $3, trace_id = $4
        WHERE id = $5
        RETURNING *;
        """
        result = await PGConnectionManager.fetch_one(query, TurnStatus.COMMITTED.value, now, message, trace_id, turn_id)
        if result is None:
            raise ValueError(f"Turn {turn_id} not found")
        return cls._pack_turn(result)
    
    
    @classmethod
    async def update_turn(
        cls, 
        turn_id: int, 
        status: TurnStatus | DontUpdateType = DontUpdate, 
        message: str | None | DontUpdateType = DontUpdate, 
        trace_id: str | None | DontUpdateType = DontUpdate
    ) -> Turn:
        """Update a turn"""
        query = """
        UPDATE turns
        SET
        """
        
        
        def build_update_query(params: list[str], values: list[Any]):
            idx = 1
            filtered_values = []
            update_params = []
            for param, value in zip(params, values):
                if value is DontUpdate:
                    continue
                update_params.append(f"{param} = ${idx}")
                filtered_values.append(value)
                idx += 1
            return ", ".join(update_params), filtered_values
        
        update_query, values = build_update_query(["status", "message", "trace_id"], [status, message, trace_id])        
        query += update_query
        query += f"""
        WHERE id = ${len(values) + 1}
        RETURNING *;
        """
        res = await PGConnectionManager.fetch_one(query, *values, turn_id)
        if res is None:
            raise ValueError(f"Turn {turn_id} not found")
        return cls._pack_turn(res)
        
    
    @classmethod
    async def revert_turn(cls, turn_id: int, message: Optional[str] = None, trace_id: Optional[str] = None) -> Turn:
        """Revert the current turn"""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        query = """
        UPDATE turns
        SET status = $1, ended_at = $2, message = $3, trace_id = $4
        WHERE id = $5
        RETURNING *;
        """
        result = await PGConnectionManager.fetch_one(query, TurnStatus.REVERTED.value, now, message, trace_id, turn_id)
        if result is None:
            raise ValueError(f"Turn {turn_id} not found")
        return cls._pack_turn(result)
    
    
    @classmethod
    async def update_version_params(cls, namespace: "PostgresNamespace", data: Dict[str, Any], turn_id: Optional[int] = None, branch_id: Optional[int] = None):
        # if namespace.is_repo and "id" not in data and "main_branch_id" not in data:
        #     # Create a new main branch
        #     branch = await cls.create_branch(name="main")
        #     # Add main_branch_id to the data
        #     data["main_branch_id"] = branch.id
            
        if namespace.repo_namespace:
            if branch_id is None:
                raise ValueError("Versioned model cannot be saved without a branch_id")
            if turn_id is None:
                raise ValueError("Versioned model cannot be saved without a turn_id")
            
            
        
        # If branch_id is provided, get the current turn
        if branch_id is not None and turn_id is not None:        
            data["branch_id"] = branch_id
            data["turn_id"] = turn_id
            turn = await cls.get_turn(turn_id)
            if turn.status != TurnStatus.STAGED:
                raise ValueError(f"Turn {turn_id} is {turn.status.value}, not STAGED.")
        return data


    @classmethod
    async def fetch(
        cls, 
        table_name: str,
        sql: str, 
        values: list[Any], 
        branch_id: int = 1, 
        turn_limit: int | None = None,
        turn_direction: str | None = None,
        is_event_source: bool = True,
    ) -> list[Any]: 
        
        filtered_alias = f"filtered_{table_name}"
        sql = sql.replace(table_name, filtered_alias)       
        if turn_limit:
            turn_order_by_clause = f"ORDER BY t.index {turn_direction} LIMIT {turn_limit}"
        else:
            turn_order_by_clause = ""

        
        turn_where_clause = []
        # if partition_id is not None:
        #     turn_where_clause.append(f"t.partition_id = {partition_id}")
        if is_event_source:
            turn_where_clause.append("m.deleted_at IS NULL")
        turn_where_clause = " AND ".join(turn_where_clause)
        
        event_source_select_clause = " DISTINCT ON (m.artifact_id)" if is_event_source else ""
        event_source_order_by_clause = "ORDER BY m.artifact_id, m.version DESC" if is_event_source else ""

        versioned_sql = f"""
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
                JOIN "{table_name}" m ON t.id = m.turn_id
                WHERE {turn_where_clause}
                {event_source_order_by_clause}
            )
            {sql}
            """

        results = await PGConnectionManager.fetch(versioned_sql, *values)
        return [dict(row) for row in results]