from contextlib import asynccontextmanager
import enum
import uuid
import datetime as dt
import contextvars
from typing import TYPE_CHECKING, AsyncGenerator, Callable, List, Literal, Type, TypeVar, Self, Any

from promptview.model.base.types import ArtifactKind



from .. import Model
from ..fields import KeyField, ModelField, RelationField
from ..postgres2.pg_query_set import PgSelectQuerySet
from ..postgres2.rowset import RowsetNode
from ..sql.queries import CTENode, RawSQL
from ..sql.expressions import RawValue
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block import Block

# ContextVars for current branch/turn
_curr_branch = contextvars.ContextVar("curr_branch", default=None)
_curr_turn = contextvars.ContextVar("curr_turn", default=None)



SpanTypeEnum = Literal["component", "stream", "llm"]
ArtifactKindEnum = Literal["block", "span", "log", "model", "literal"]

class TurnStatus(enum.StrEnum):
    """Status of a turn in the version history."""
    STAGED = "staged"
    COMMITTED = "committed"
    REVERTED = "reverted"
    BRANCH_CREATED = "branch_created"


class Branch(Model):
    _namespace_name = "branches"
    id: int = KeyField(primary_key=True)
    name: str | None = ModelField(default=None)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    forked_from_index: int | None = ModelField(default=None)
    forked_from_turn_id: int | None = ModelField(default=None, foreign_key=True)
    forked_from_branch_id: int | None = ModelField(default=None, foreign_key=True)
    current_index: int = ModelField(default=0)

    turns: List["Turn"] = RelationField(foreign_key="branch_id")
    children: List["Branch"] = RelationField(foreign_key="forked_from_branch_id")
    
    async def fork_branch(self, turn: "Turn", name: str | None = None):
        branch = await Branch(
            forked_from_index=turn.index,
            forked_from_turn_id=turn.id,
            current_index=turn.index + 1,
            forked_from_branch_id=self.id,
            name=name,
        ).save()
        return branch
    
    
    @classmethod
    async def get_main(cls):
        branch = await cls.get_or_none(1)
        if branch is None:
            branch = await cls(name="main").save()
        return branch
    
    
    @asynccontextmanager
    async def fork(self, turn: "Turn", name: str | None = None) -> AsyncGenerator["Branch", None]:
        branch = await self.fork_branch(turn, name)
        try:
            with branch as b:
                yield b
        except Exception as e:
            # await turn.revert()
            raise e
        
        
    
    @asynccontextmanager  
    async def start_turn(
        self, 
        message: str | None = None, 
        status: TurnStatus = TurnStatus.STAGED,
        raise_on_error: bool = True,
        auto_commit: bool = True,
        **kwargs
    ) -> AsyncGenerator["Turn", None]:
        turn = await self.create_turn(message, status, auto_commit, **kwargs)
        turn._raise_on_error = raise_on_error
        async with turn as t:
            yield t
        # try:
        #     with turn as t:
        #         yield t
        # except Exception as e:
        #     await turn.revert(str(e))
        #     raise e
        # finally:
        #     await turn.commit()
    

    async def create_turn(self, message: str | None = None, status: TurnStatus = TurnStatus.STAGED, auto_commit: bool = True, **kwargs):
        query = f"""
            WITH updated_branch AS (
                UPDATE branches
                SET current_index = current_index + 1
                WHERE id = $1
                RETURNING id, current_index
            ),
            new_turn AS (
                INSERT INTO turns (branch_id, index, created_at, status{"".join([", " + k for k in kwargs.keys()])})
                SELECT id, current_index - 1, current_timestamp, $2{"".join([", $" + str(i) for i in range(3, len(kwargs) + 3)])}
                FROM updated_branch
                RETURNING *
            )
            SELECT * FROM new_turn;
        """   

        if not self.id:
            raise ValueError("Branch ID is not set")
        
        turn_ns = Turn.get_namespace()
        row = await PGConnectionManager.fetch(query, self.id, status.value, *[kwargs[k] for k in kwargs.keys()])
        
        
        # row = await PGConnectionManager.fetch_one(sql, self.id, status.value, message, metadata)
        if not row:
            raise ValueError("Failed to create turn")
        if turn_ns._model_cls is None:
            raise ValueError("Turn namespace is not initialized")
        # turn = Turn(**row[0])
        turn = turn_ns._model_cls(**row[0])
        turn.branch_id = self.id
        turn._auto_commit = auto_commit
        return turn

        
        

        # finally:
            # await turn.commit()
        

    
    @classmethod
    def recursive_query(cls, branch_id: int) -> PgSelectQuerySet["Branch"]:
        sql = f"""
            SELECT
                id,
                name,
                forked_from_index,
                forked_from_branch_id,
                current_index AS start_turn_index
            FROM branches
            WHERE id = {branch_id}

            UNION ALL

            SELECT
                b.id,
                b.name,
                b.forked_from_index,
                b.forked_from_branch_id,
                bh.forked_from_index AS start_turn_index
            FROM branches b
            JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
        """
        return PgSelectQuerySet(Branch, alias="branch_hierarchy", recursive=True).raw_sql(sql, [
            "id", 
            "name", 
            "forked_from_index", 
            "forked_from_branch_id", 
            ("current_index", "start_turn_index")
        ])
        # return RowsetNode("branch_hierarchy", RawSQL(sql), model=Branch, key="id", recursive=True)

    


class Turn(Model):
    # _is_base: bool = True
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    ended_at: dt.datetime | None = ModelField(default=None)
    index: int = ModelField(order_by=True)
    status: TurnStatus = ModelField(default=TurnStatus.STAGED)
    message: str | None = ModelField(default=None)
    branch_id: int = ModelField(foreign_key=True, foreign_cls=Branch)
    trace_id: str | None = ModelField(default=None)
    metadata: dict | None = ModelField(default=None)
    artifacts: List["Artifact"] = RelationField(foreign_key="turn_id")
    spans: List["ExecutionSpan"] = RelationField(foreign_key="turn_id")
    block_trees: List["BlockTree"] = RelationField(foreign_key="turn_id")
    
    _auto_commit: bool = True
    _raise_on_error: bool = True

    forked_branches: List["Branch"] = RelationField("Branch", foreign_key="forked_from_turn_id")
    
    @classmethod
    def blocks(cls):
        from ..block_models.block_log import parse_block_tree_turn
        return cls.query().include(
            BlockTree.query(alias="bt").select("*").include(
                BlockNode.query(alias="bn").select("*").include(
                    BlockModel.query(alias="bm").select("*")
                )
            )
        ).parse(parse_block_tree_turn)
        
        
    async def add_block(self, block: "Block", span_id: int | None = None):
        from ..block_models.block_log import insert_block
        return await insert_block(block, self.branch_id, self.id, span_id)
        
        
    async def commit(self):
        """Mark this turn as committed."""
        self.status = TurnStatus.COMMITTED
        self.ended_at = dt.datetime.now()
        return await self.save()

    async def revert(self, reason: str | None = None):
        """Mark this turn as reverted with an optional reason."""
        self.status = TurnStatus.REVERTED
        self.ended_at = dt.datetime.now()
        if reason:
            self.message = reason
        return await self.save()
    
    
    
    async def __aenter__(self):
        if self.status != TurnStatus.STAGED:
            raise ValueError("Turn is not staged")
        ns = self.get_namespace()
        self._ctx_token = ns.set_ctx(self)
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        ns = self.get_namespace()
        ns.set_ctx(None)
        self._ctx_token = None
        if exc_type is not None:
            if self._auto_commit:
                await self.revert(str(exc_value))
            else:
                return False
        elif not self._auto_commit:
            # await self.revert()
            return True
        else:
            await self.commit()
        if self._raise_on_error:
            return False
        return True
    
    
    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        to_select: bool = True,
        **kwargs
    ) -> "PgSelectQuerySet[Self]":
        from ..postgres2.pg_query_set import PgSelectQuerySet
        query = PgSelectQuerySet(cls, alias=alias)           
            # .where(lambda t: (t.index <= branch_cte.get_field("start_turn_index")))
        branch_id = Branch.resolve_target_id_or_none(branch)
        if branch_id is not None:
            branch_cte = Branch.recursive_query(branch_id)
            col = branch_cte.get_field("start_turn_index")
            query = (
                query 
                .use_cte(branch_cte, name="branch_hierarchy", alias="bh", on=("branch_id", "id"))
                .where(lambda t: (t.index <= RawValue[int]("bh.start_turn_index - 1")))
            )
        if to_select:
            query = query.select(*fields if fields else "*")
        return query
        # return cls.query_extra(query, **kwargs)

    
    @classmethod
    def vquery(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        branch: Branch | None = None, 
        **kwargs
    ) -> "PgSelectQuerySet[Self]":
        from ..postgres2.pg_query_set import PgSelectQuerySet
        branch_id = Branch.resolve_target_id(branch)
        branch_cte = Branch.recursive_query(branch_id)
        col = branch_cte.get_field("start_turn_index")
        query = (
            PgSelectQuerySet(cls) \
            .use_cte(branch_cte, name="branch_hierarchy", alias="bh", on=("branch_id", "id"))    
            .where(lambda t: (t.index <= RawValue[int]("bh.start_turn_index")))
            # .where(lambda t: (t.index <= branch_cte.get_field("start_turn_index")))
        )
        return cls.query_extra(query, **kwargs)
    
    @classmethod    
    def query_extra(cls: Type[Self], query: "PgSelectQuerySet[Self]", **kwargs) -> "PgSelectQuerySet[Self]":
        return query
    
    
    @asynccontextmanager
    async def start_span(self, name: str, span_type: SpanTypeEnum):
        span = await ExecutionSpan(
            name=name,
            span_type=span_type,
            index=0,
            status="running",
        ).save()
        try:
            with span as s:
                yield span
        except Exception as e:
            raise e
        finally:
            span.end_time = dt.datetime.now()
            span.status = "completed"
            await span.save()
        


class Artifact(Model):
    """
    Hub table for all versionable objects.
    Stores lineage metadata for blocks, spans, logs, and user models.
    """
    id: int = KeyField(primary_key=True)
    kind: ArtifactKindEnum = ModelField()  # 'block', 'span', 'log', 'model'
    model_name: str | None = ModelField(None)  # 'meal_plan', 'workout', etc.

    # Lineage
    branch_id: int = ModelField(foreign_key=True, foreign_cls=Branch)
    turn_id: int = ModelField(foreign_key=True, foreign_cls=Turn)
    turn: "Turn | None" = RelationField("Turn", primary_key="turn_id", foreign_key="id")
    branch: "Branch | None" = RelationField("Branch", foreign_key="id")
    # span_id: int | None = ModelField()

    # Event sourcing
    version: int = ModelField(default=1)
    # parent_artifact_id: int | None = ModelField(foreign_key=True, self_ref=True)

    # Timeline
    # seq: int = ModelField()
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    # deleted_at: dt.datetime | None = ModelField()

    _namespace_name = "artifacts"
    _is_artifact_hub = True
    
    @classmethod
    def from_model(
        cls, 
        model: "VersionedModel | ArtifactModel", 
        branch: Branch | int | None = None,
        turn: Turn | int | None = None
    ) -> "Artifact":
        # if isinstance(model, VersionedModel):
        branch_id = Branch.resolve_target_id(branch)
        turn_id = Turn.resolve_target_id(turn)
        return cls(
            kind=model._artifact_kind,
            model_name=model.get_namespace_name(),
            branch_id=branch_id,
            turn_id=turn_id
        )
        

        


class VersionedModel(Model):
    """Mixin for models tied to a specific branch & turn."""
    _is_base = True
    _artifact_kind: ArtifactKind = "model"
    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    artifact: "Artifact | None" = RelationField(primary_key="artifact_id", foreign_key="id")
    
    
    def _build_insert_query(self, branch: Branch | int | None = None, turn: Turn | int | None = None, version: int = 1):
        ns = self.get_namespace()
        branch_id = Branch.resolve_target_id(branch)
        turn_id = Turn.resolve_target_id(turn)
        art_query = Artifact(
            kind=self._artifact_kind,
            model_name=self.get_namespace_name(),
            version=version,
            branch_id=branch_id,
            turn_id=turn_id
        ).insert()
        dump = self.model_dump()
        dump["artifact_id"] = art_query.col("id")
        return ns.insert(dump).select("*").one().json()                        

    async def save(self, *, branch: Branch | int | None = None, turn: Turn | int | None = None):
        ns = self.get_namespace()
        # if not self._should_save_to_db(branch, turn):
        #     print(f"WARNING: {self.__class__.__name__} is not saved to database")
        #     if not ns.has_primary_key(self):
        #         ns.set_primary_key(self, ns.generate_fake_key())
        #     return self
        # return await super().save()
        
        pk_value = self.primary_id
        self._load_context_vars()
        if pk_value is None:
            result = await self._build_insert_query(branch, turn)
            # branch_id = Branch.resolve_target_id(branch)
            # turn_id = Turn.resolve_target_id(turn)
            # art_query = Artifact(
            #     kind=self._artifact_kind,
            #     model_name=self.get_namespace_name(),
            #     branch_id=branch_id,
            #     turn_id=turn_id
            # ).insert()
            # dump = self.model_dump()
            # dump["artifact_id"] = art_query.col("id")
            # result = await ns.insert(dump).select("*").one().json()                        
            # result = await ns.insert(self.model_dump()).select("*").one().json()
        else:
            result = await ns.update(pk_value, self.model_dump()).select("*").one().json()
        for key, value in result.items():
            setattr(self, key, value)
            
        if self.artifact is None:
            await self.include("artifact")
        return self
    
    def _should_save_to_db(self, branch: Branch | int | None = None, turn: Turn | int | None = None) -> bool:        
        branch_id = self._resolve_branch_id(branch)        
        turn_id = self._resolve_turn_id(turn)
        return branch_id is not None and turn_id is not None

    def _resolve_branch_id(self, branch):
        if branch is None:
            branch = Branch.current()
        return branch.id if isinstance(branch, Branch) else branch

    def _resolve_turn_id(self, turn):
        if turn is None:
            turn = Turn.current()
        return turn.id if isinstance(turn, Turn) else turn
    
    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        turn_cte: "PgSelectQuerySet[Turn] | None" = None,
        limit: int | None = None, 
        offset: int | None = None, 
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        direction: Literal["asc", "desc"] = "desc",
        **kwargs
    ) -> "PgSelectQuerySet[Self]":
        from ..postgres2.pg_query_set import PgSelectQuerySet
        
        if turn_cte is None:
            turn_cte = Turn.query(branch=branch, to_select=True)
            if statuses:
                turn_cte = turn_cte.where(lambda t: t.status.isin(statuses))
            if limit:
                turn_cte = turn_cte.limit(limit)
                turn_cte = turn_cte.order_by(f"-index" if direction == "desc" else "index")
            if offset:
                turn_cte = turn_cte.offset(offset)
        
        art_cte = Artifact.query().join(turn_cte, on=("turn_id", "id")).use_cte(turn_cte, name="committed_turns", alias="ct")
        return (
            PgSelectQuerySet(cls, alias=alias) \
            .use_cte(
                art_cte,
                name="artifact_cte",
                alias="ac",
            )
            .select(*fields if fields else "*")
        )
        
        
    
    @classmethod
    def vquery(
        cls, 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        limit: int | None = None, offset: int | None = None, 
        direction: Literal["asc", "desc"] = "desc",
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        **kwargs
    ):
        from ..postgres2.pg_query_set import PgSelectQuerySet
        turn_cte = Turn.vquery().select(*fields or "*")
        # .where(lambda t: t.status.isin([TurnStatus.COMMITTED, TurnStatus.STAGED]))
        if statuses:
            turn_cte = turn_cte.where(lambda t: t.status.isin(statuses))
        if limit:
            turn_cte = turn_cte.limit(limit)
            turn_cte = turn_cte.order_by(f"-index" if direction == "desc" else "index")
        if offset:
            turn_cte = turn_cte.offset(offset)
        
        art_cte = Artifact.query().join(turn_cte, on=("turn_id", "id")).use_cte(turn_cte, name="committed_turns", alias="ct")
        return (
            PgSelectQuerySet(cls, alias=alias) \
            .use_cte(
                art_cte,
                name="artifact_cte",
                alias="ac",
            )
        )
        
        
    def __repr__(self):
        fields_str = ""
        artifact_str = ""
        for k, v in self.model_dump().items():
            if k == "artifact":
                artifact_str = f"artifact={v}"
            else:
                fields_str += f"{k}={v}, "
        return f"{self.__class__.__name__}({fields_str}) {artifact_str}"



class BlockModel(Model):
    _namespace_name: str = "blocks"
    id: str = KeyField(primary_key=True)
    # created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    content: str | None = ModelField(default=None)
    json_content: dict | None = ModelField(default=None) 
    block_nodes: list["BlockNode"] = RelationField(foreign_key="block_id")   
    

class BlockNode(Model):
    id: int = KeyField(primary_key=True)
    tree_id: int = ModelField(foreign_key=True)
    path: str = ModelField(db_type="LTREE")
    block_id: str = ModelField(foreign_key=True)
    styles: list[str] | None = ModelField(default=None)
    role: str | None = ModelField(default=None)
    tags: list[str] | None = ModelField(default=None)
    attrs: dict | None = ModelField(default=None)
    block: "BlockModel" = RelationField(primary_key="block_id", foreign_key="id")
    tree: "BlockTree" = RelationField(primary_key="tree_id", foreign_key="id")
    
    @classmethod
    async def block_query(cls, cte):
        from ..block_models.block_log import pack_block
        from ..sql.queries import Column
        records = await cls.query([
            Column("styles", "bn"),
            Column("role", "bn"),
            Column("tags", "bn"),
            Column("path", "bn"),
            Column("attrs", "bn"),
            Column("type", "bn"),
            Column("content", "bsm"),
            Column("json_content", "bsm"),            
        ], alias="bn") \
        .use_cte(cte,"tree_cte", alias="btc") \
        .join(BlockModel.query(["content", "json_content"], alias="bsm"), on=("block_id", "id")) \
        .where(lambda b: (b.tree_id == RawValue("btc.id"))).print().json()
        return pack_block(records)
     
        
class BlockTree(VersionedModel):
    _artifact_kind: ArtifactKind = "block"
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    nodes: List[BlockNode] = RelationField(foreign_key="tree_id")
    span_id: int | None = ModelField(foreign_key=True)
    


class ArtifactModel(VersionedModel):
    """VersionedModel with artifact tracking."""
    _is_base = True
    _dirty_fields: dict[str, Any] = {}
    # id: int = KeyField(primary_key=True)
    # artifact_id: uuid.UUID = KeyField(
    #         default_factory=uuid.uuid4, 
    #         type="uuid"
    #     )
    # version: int = ModelField(default=1)
    # version: int = KeyField(default=1)

    @classmethod
    async def latest(cls, artifact_id: uuid.UUID) -> Self | None:
        """Backend-specific: get latest version per artifact."""
        # Placeholder — will be overridden in backend manager
        return await cls.query().filter(artifact_id=artifact_id).order_by("-version").first()

    
    async def _super_save(self):
        return await super().save()

    async def save(self, *, branch: Branch | int | None = None, turn: Turn | int | None = None):        
        ns = self.get_namespace()
        if primary_key:= ns.get_primary_key(self):
            result = await self._build_insert_query(branch, turn, self.artifact.version + 1)
            # obj = self.model_copy(update={"turn_id": None, "branch_id": None})
            # obj.version += 1
            # return await obj._super_save()
        else:
            result = await super().save()
        self._dirty_fields = {}
        return result
    
    # @classmethod
    # def query(
    #     cls: Type[Self], 
    #     fields: list[str] | None = None, 
    #     alias: str | None = None, 
    #     use_ctx: bool = True,
    #     **kwargs
    # ) -> "PgSelectQuerySet[Self]":  
    #     query = (
    #         PgSelectQuerySet(cls, alias=alias) \
    #         .distinct_on("artifact_id")
    #         .order_by("-artifact_id", "-version")
    #     ) 
    #     return query
    
        
    @classmethod
    def vquery(
        cls,
        fields: list[str] | None = None,
        alias: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        direction: Literal["asc", "desc"] = "desc",
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        **kwargs
    ) -> "PgSelectQuerySet[Self]":
        from ..postgres2.pg_query_set import PgSelectQuerySet

        # Build turn CTE with versioning context
        turn_cte = Turn.vquery().select(*fields or "*")
        if statuses:
            turn_cte = turn_cte.where(lambda t: t.status.isin(statuses))
        if limit:
            turn_cte = turn_cte.limit(limit)
            turn_cte = turn_cte.order_by(f"-index" if direction == "desc" else "index")
        if offset:
            turn_cte = turn_cte.offset(offset)

        # Join Artifact with turn CTE to get event-sourced versions
        art_cte = (
            Artifact.query()
            .join(turn_cte, on=("turn_id", "id"))
            .use_cte(turn_cte, name="committed_turns", alias="ct")
        )

        # Query the model using the artifact CTE
        query = (
            PgSelectQuerySet(cls, alias=alias) \
            .use_cte(
                art_cte,
                name="artifact_cte",
                alias="ac",
            )
            .select(*fields or "*")
            .join(art_cte, on=("artifact_id", "id"))
            .distinct_on("artifact_id")
            .include(Artifact)
            .order_by("-artifact_id", "-artifact_id")
        )
        return query

    # def __getattr__(self, name: str):
    #     print(f"__getattr__ {name}")
    #     return super().__getattr__(name)
    # def __getattribute__(self, name):
    #     print(f"Accessing attribute: {name}")
    #     # Call the default implementation to avoid recursion
    #     value = super().__getattribute__(name)
    
    def __setattr__(self, name, value):
        print(f"Setting {name} = {value}")
        if name != "_dirty_fields":
            curr_value = super().__getattribute__(name)
            if curr_value != value and name not in self._dirty_fields:
                self._dirty_fields[name] = curr_value
        # Prevent recursion by calling the base implementation
        super().__setattr__(name, value)
        





class Log(VersionedModel):
    _artifact_kind: ArtifactKind = "log"
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    message: str = ModelField()
    level: Literal["info", "warning", "error"] = ModelField()
    



ValueIOKind = Literal["input", "output"]

class SpanValue(Model):
    id: int = KeyField(primary_key=True)    
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    kind: ArtifactKindEnum = ModelField()
    io_kind: ValueIOKind = ModelField()
    # index: int = ModelField()
    span_id: int = ModelField(foreign_key=True)
    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    artifact: "Artifact | None" = RelationField( primary_key="artifact_id", foreign_key="id")
    
    

class ExecutionSpan(VersionedModel):
    """Represents a single execution unit (component call, stream, etc.)"""
    _artifact_kind: ArtifactKind = "span"
    id: int = KeyField(primary_key=True)
    name: str = ModelField()  # Function/component name
    path: str = ModelField(db_type="LTREE")
    span_type: SpanTypeEnum = ModelField()
    parent_span_id: int | None = ModelField(foreign_key=True, self_ref=True)
    start_time: dt.datetime = ModelField(default_factory=dt.datetime.now)
    end_time: dt.datetime | None = ModelField(default=None)
    tags: list[str] | None = ModelField(default=None)
    depth: int = ModelField(default=0)  # Nesting level
    metadata: dict[str, Any] = ModelField(default={})
    status: Literal["running", "completed", "failed"] = ModelField(default="running")
    
    # Relations
    values: List["SpanValue"] = RelationField(foreign_key="span_id")
    # events: List[Event] = RelationField(foreign_key="execution_span_id")
    block_trees: List[BlockTree] = RelationField(foreign_key="span_id")
    
    # def _resolve_branch_id(self, branch: Branch | None = None) -> int:
    #     branch_id = super()._resolve_branch_id(branch)
    #     return branch_id or 1
    
    # def _resolve_turn_id(self, turn: Turn | None = None) -> int:
    #     turn_id = super()._resolve_turn_id(turn)
    #     return turn_id or 1
    
    def _get_target_meta(self, target: Any) -> tuple[ArtifactKindEnum, int | None]:
        from ...block import Block
        if isinstance(target, Block):
            return "block", None
        elif isinstance(target, Log):
            return "log", target.artifact_id
        elif isinstance(target, ExecutionSpan):
            if target == self:
                print(f"target == self {target.id} {self.id}")
            return "span", target.artifact_id
        elif isinstance(target, VersionedModel):
            return "model", target.artifact_id
        else:
            return "literal", None

    
    async def log_value(self, target: Any, io_kind: ValueIOKind= "output"):
        kind, artifact_id = self._get_target_meta(target)
        if kind == "block":
            return await self.add_block_event(target, io_kind)
        try:
            value = await self.add(SpanValue(
                span_id=self.id,
                kind=kind,
                io_kind=io_kind,            
                artifact_id=artifact_id,
            ))
            value.artifact = target
            return value
        except Exception as e:
            print(f"Error logging value: {e}")
            raise e
    
    async def add_block_event(self, block: "Block", io_kind: ValueIOKind= "output"):
        from ..block_models.block_log import insert_block
        from ..namespace_manager2 import NamespaceManager
        # if self._should_save_to_db():
        #     tree_id = await insert_block(block, self.artifact.branch_id, self.artifact.turn_id, self.id)
        # else:
        #     tree_id = str(uuid.uuid4())
        block_tree = await insert_block(block, self.artifact.branch_id, self.artifact.turn_id, self.id)
            
        value = await self.add(SpanValue(
            span_id=self.id,
            kind="block",
            io_kind=io_kind,
            artifact_id=block_tree.artifact.id,
        ))
        return value
    
    # async def add_stream(self, index: int):
    #     return await SpanValue(
    #         span_id=self.id,
    #         kind="stream",
    #         artifact_id="",
    #         index=index
    #     ).save()
    
    
    # async def add_span_event(self, span: "ExecutionSpan", index: int):
    #     return await SpanValue(
    #         span_id=self.id,
    #         kind="span",
    #         artifact_id=str(span.id),
    #         index=index
    #     ).save()
    
    # async def add_log_event(self, log: "Log", index: int):
    #     return await SpanValue(
    #         span_id=self.id,
    #         kind="log",
    #         artifact_id=str(log.id),
    #         index=index
    #     ).save()
        
        
    # async def add_model_event(self, model: "Model", index: int):
    #     return await SpanValue(
    #         span_id=self.id,
    #         kind="model",
    #         artifact_id=str(model.id),
    #         table=model._namespace_name,
    #         index=index
    #     ).save()
        
        
    
    