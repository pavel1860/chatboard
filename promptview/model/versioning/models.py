from contextlib import asynccontextmanager
import enum
import uuid
import datetime as dt
import contextvars
from typing import TYPE_CHECKING, AsyncGenerator, Callable, List, Literal, Type, TypeVar, Self, Any

from pydantic import BaseModel, Field

from promptview.model.base.types import ArtifactKind
from promptview.model.db_types import Tree
from promptview.model.sql2.expressions import Raw
from promptview.model.sql2.relations import RawRelation

from promptview.utils.type_utils import SerializableType, UnknownType, deserialize_value, serialize_value, type_to_str, str_to_type, type_to_str_or_none



from .. import Model
from ..fields import KeyField, ModelField, RelationField
from ..postgres2.pg_query_set import PgSelectQuerySet
from ..postgres2.rowset import RowsetNode
from ..sql.queries import CTENode, RawSQL
from ..sql.expressions import RawValue
from ...utils.db_connections import PGConnectionManager

if TYPE_CHECKING:
    from ...block import Block
    from promptview.model.sql2.relational_queries import SelectQuerySet
    from promptview.model.sql2.pg_query_builder import PgQueryBuilder
    from promptview.model.versioning.artifact_log import ArtifactLog
    from ...prompt.context import Context

# ContextVars for current branch/turn
_curr_branch = contextvars.ContextVar("curr_branch", default=None)
_curr_turn = contextvars.ContextVar("curr_turn", default=None)



SpanType = Literal["component", "stream", "llm", "evaluator"]
ArtifactKindEnum = Literal["block", "span", "log", "model", "parameter", "list"]

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
    kind: str | None = ModelField(default=None)
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
    
    
    @classmethod
    async def get_or_create(cls, id: int, name: str | None = None, kind: str | None = None):
        branch = await cls.get_or_none(id)
        if branch is None:
            branch = await cls(name=name, kind=kind).save()
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
        

    
    # @classmethod
    # def recursive_query(cls, branch_id: int) -> PgSelectQuerySet["Branch"]:
    #     sql = f"""
    #         SELECT
    #             id,
    #             name,
    #             forked_from_index,
    #             forked_from_branch_id,
    #             current_index AS start_turn_index
    #         FROM branches
    #         WHERE id = {branch_id}

    #         UNION ALL

    #         SELECT
    #             b.id,
    #             b.name,
    #             b.forked_from_index,
    #             b.forked_from_branch_id,
    #             bh.forked_from_index AS start_turn_index
    #         FROM branches b
    #         JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
    #     """
    #     return PgSelectQuerySet(Branch, alias="branch_hierarchy", recursive=True).raw_sql(sql, [
    #         "id", 
    #         "name", 
    #         "forked_from_index", 
    #         "forked_from_branch_id", 
    #         ("current_index", "start_turn_index")
    #     ])
        # return RowsetNode("branch_hierarchy", RawSQL(sql), model=Branch, key="id", recursive=True)

    @classmethod
    def recursive_query(cls, branch_id: int) -> "SelectQuerySet":
        from promptview.model.sql2.pg_query_builder import PgQueryBuilder, select
        from promptview.model.sql2.relational_queries import SelectQuerySet
        
        return PgQueryBuilder().raw(
            sql=f"""
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
            """,
            name="branch_hierarchy",
            namespace=cls.get_namespace()            
        )
        # rel = RawRelation(
        #     sql=f"""
        #         SELECT
        #             id,
        #             name,
        #             forked_from_index,
        #             forked_from_branch_id,
        #             current_index AS start_turn_index
        #         FROM branches
        #         WHERE id = {branch_id}

        #         UNION ALL

        #         SELECT
        #             b.id,
        #             b.name,
        #             b.forked_from_index,
        #             b.forked_from_branch_id,
        #             bh.forked_from_index AS start_turn_index
        #         FROM branches b
        #         JOIN branch_hierarchy bh ON b.id = bh.forked_from_branch_id
        #     """,
        #     name="branch_hierarchy",
        #     namespace=cls.get_namespace()
        # )
        # return SelectQuerySet(rel)

    


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
    spans: Tree["ExecutionSpan"] = RelationField(foreign_key="turn_id")
    data: Tree["DataFlowNode"] = RelationField(foreign_key="turn_id")
    test_turns: List["TestTurn"] = RelationField(foreign_key="turn_id")    

    _auto_commit: bool = True
    _raise_on_error: bool = True

    forked_branches: List["Branch"] = RelationField( foreign_key="forked_from_turn_id")
    
    
    @property
    def inputs(self) -> list["DataFlowNode"]:
        return self.data["1.0.*"]
    
    
    def get_kwargs(self) -> dict[str, Any]:        
        kwargs = {}
        for d in self.data["1.0.*"]:
            path = d.path.split(".")[-1]
            kwargs[path] = d.value
        return kwargs
    
    def get_args(self) -> list[Any]:
        return [d.value for d in self.data["1.0.*"]]
        
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
    
    # @classmethod
    # async def _parse_executions(cls, turns: List["Turn"]):
    #     from collections import defaultdict
    #     from ..namespace_manager2 import NamespaceManager
    #     def kind2table(k: str):
    #         if k == "parameter":
    #             return "parameters"
    #         return k

    #     models_to_load = defaultdict(list)

    #     for turn in turns:
    #         for span in turn.spans:
    #             print(span.id, span.name)
    #             for value in span.values:
    #                 if value.kind != "span":
    #                     print(value.path, value.kind, value.artifact_id)
    #                     models_to_load[value.kind].append(value.artifact_id)
                    
    #     model_lookup = {"span": {s.artifact_id: s for turn in turns for s in turn.spans}}
    #     for k in models_to_load:
    #         if k == "list":
    #             models = await Artifact.query(include_branch_turn=True).where(Artifact.id.isin(models_to_load[k]))
    #             model_lookup["list"] = {m.id: m for m in models}
    #         elif k == "block_trees":
    #             models = await get_blocks(models_to_load[k], dump_models=False, include_branch_turn=True)
    #             model_lookup[k] = models
    #         # elif k == "execution_spans":
    #         #     value_dict[k] = {s.artifact_id: s for s in spans}
    #         else:
    #             ns = NamespaceManager.get_namespace(kind2table(k))
    #             models = await ns._model_cls.query(include_branch_turn=True).where(ns._model_cls.artifact_id.isin(models_to_load[k]))
    #             model_lookup[k] = {m.artifact_id: m for m in models}

    #     for turn in turns:
    #         for span in turn.spans:
    #             for value in span.values:
    #                 value._value = model_lookup[value.kind][value.artifact_id]
                    
    #     return turns
    
    
    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        to_select: bool = True,
        include_branch_turn: bool = False,
        include_executions: bool = False,
        **kwargs
    ) -> "PgQueryBuilder[Self]":
        from ..postgres2.pg_query_set import PgSelectQuerySet
        from ..sql2.pg_query_builder import PgQueryBuilder, select
        from ..versioning.artifact_log import ArtifactLog
        
        query = PgQueryBuilder().select(cls)       
        if fields:
            query.select(*fields)
        if alias:
            query.alias(alias)
            # .where(lambda t: (t.index <= branch_cte.get_field("start_turn_index")))
        branch_id = Branch.resolve_target_id_or_none(branch)
        if branch_id is not None:
            branch_cte = Branch.recursive_query(branch_id)
            query = (
                query 
                .where(Raw(f"turns.index <= bh.start_turn_index{' - 1' if not include_branch_turn else ''}"))
                .join_cte(branch_cte, "branch_hierarchy", on=("branch_id", "id"), alias="bh", recursive=True)
            )
        if include_executions:
            # query.include(
            #     ExecutionSpan.query()
            #         .include(Artifact)
            #         .include(
            #             DataFlowNode.query()
            #             .include(DataArtifact)
            #         )
            # )
                    
            query = (
                query
                .include(
                    DataFlowNode.query()
                    .include(DataArtifact)            
                )
            )
            query.parse(ArtifactLog.populate_turns, target="models")
        # if to_select:
            # query = query.select(*fields if fields else "*")
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
    async def start_span(self, name: str, span_type: SpanType):
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
    span_id: int | None = ModelField(foreign_key=True)
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
        
    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        turn_cte: "PgSelectQuerySet[Turn] | None" = None,
        use_liniage: bool = True,
        limit: int | None = None, 
        offset: int | None = None, 
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        direction: Literal["asc", "desc"] = "desc",
        include_branch_turn: bool = False,
        **kwargs
    ) -> "PgQueryBuilder[Self]":
        # from ..postgres2.pg_query_set import PgSelectQuerySet
        
        from promptview.model.sql2.pg_query_builder import PgQueryBuilder, select
        query = PgQueryBuilder().select(cls)
        
        if fields:
            query.select(*fields)
        
        if turn_cte is None and use_liniage:
            turn_cte = Turn.query(branch=branch, to_select=True, include_branch_turn=include_branch_turn)
            if statuses:
                # turn_cte = turn_cte.where(lambda t: t.status.isin(statuses))
                turn_cte = turn_cte.where(Turn.status.isin(statuses))
            if limit:
                turn_cte = turn_cte.limit(limit)
                turn_cte = turn_cte.order_by(f"-index" if direction == "desc" else "index")
            if offset:
                turn_cte = turn_cte.offset(offset) 
        if turn_cte is not None:
            query.join_cte(
                turn_cte,
                "turn_liniage",
                # alias="tl",
            )            
        return query
                       
        



class VersionedModel(Model):
    """Mixin for models tied to a specific branch & turn."""
    _is_base = True
    _artifact_kind: ArtifactKind = "model"
    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    artifact: "Artifact | None" = RelationField(primary_key="artifact_id", foreign_key="id")
    
    
    def _build_insert_query(
        self, 
        branch: Branch | int | None = None, turn: Turn | int | None = None, 
        version: int = 1,
        exclude: set[str] | None = None
    ):
        # Import here to avoid circular dependency        
        from ...prompt.context import Context, ContextError
        ctx = Context.current_or_none()
        if ctx is None:            
            raise ContextError("Context not found")
        
        branch_id = Branch.resolve_target_id(branch)
        turn_id = Turn.resolve_target_id(turn)
        
        span_id = ctx.current_span.id if ctx.current_span else None

        # If we still don't have branch/turn, this will fail (expected for non-Context usage)
        ns = self.get_namespace()
        art_query = Artifact(
            kind=self._artifact_kind,
            model_name=self.get_namespace_name(),
            version=version,
            branch_id=branch_id,
            turn_id=turn_id,
            span_id=span_id
        ).insert()
        dump = self.model_dump(exclude=exclude)
        dump["artifact_id"] = art_query.col("id")
        return ns.insert(dump).select("*").one().json()                        

    async def save(self, *, branch: Branch | int | None = None, turn: Turn | int | None = None):
        ns = self.get_namespace()

        pk_value = self.primary_id
        self._load_context_vars()
        if pk_value is None:
            result = await self._build_insert_query(branch, turn)

            # If result is None, we're in in-memory mode (Context exists but no branch/turn)
            if result is None:
                # Generate fake ID for in-memory execution
                if not hasattr(self, 'id') or self.id is None:
                    self.id = -1 * (id(self) % 1000000)  # Negative ID to distinguish from DB records
                return self
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
        use_liniage: bool = True,
        limit: int | None = None, 
        offset: int | None = None, 
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        direction: Literal["asc", "desc"] = "desc",
        include_branch_turn: bool = False,
        **kwargs
    ) -> "PgQueryBuilder[Self]":
        # from ..postgres2.pg_query_set import PgSelectQuerySet
        from ..sql2.pg_query_builder import select, PgQueryBuilder
        
        # if turn_cte is None:
        #     turn_cte = Turn.query(branch=branch, to_select=True)
        #     if statuses:
        #         turn_cte = turn_cte.where(lambda t: t.status.isin(statuses))
        #     if limit:
        #         turn_cte = turn_cte.limit(limit)
        #         turn_cte = turn_cte.order_by(f"-index" if direction == "desc" else "index")
        #     if offset:
        #         turn_cte = turn_cte.offset(offset)
        
        # art_cte = Artifact.query(statuses=statuses, limit=limit, offset=offset, direction=direction).join(turn_cte, on=("turn_id", "id")).use_cte(turn_cte, name="committed_turns", alias="ct")
        art_cte = Artifact.query(
            statuses=statuses, 
            limit=limit, 
            offset=offset, 
            direction=direction, 
            use_liniage=use_liniage, 
            turn_cte=turn_cte,
            include_branch_turn=include_branch_turn
        )
        return (
            PgQueryBuilder()
            .select(cls)
            .join_cte(art_cte, "artifact_cte", alias="ac")
        )
        
        
        # return (
        #     PgSelectQuerySet(cls, alias=alias) \
        #     .use_cte(
        #         art_cte,
        #         name="artifact_cte",
        #         alias="ac",
        #     )
        #     .select(*fields if fields else "*")
        # )
        
        
    
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




class Parameter(VersionedModel):
    _artifact_kind: ArtifactKind = "parameter"
    id: int = KeyField(primary_key=True)
    data: dict = ModelField()
    kind: str = ModelField()
    
    @property
    def value(self) -> SerializableType:
        return deserialize_value(self.data["value"], self.kind)
    
    
        
    

# class BlockModel(Model):
#     _namespace_name: str = "blocks"
#     id: str = KeyField(primary_key=True)
#     # created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
#     content: str | None = ModelField(default=None)
#     json_content: dict | None = ModelField(default=None)
#     signatures: list["BlockSignature"] = RelationField(foreign_key="block_id")


# class BlockSignature(Model):
#     """
#     Reusable block signature combining content + styling.
#     Deduplicates blocks with identical content and presentation.
#     """
#     _namespace_name: str = "block_signatures"
#     id: str = KeyField(primary_key=True)  # hash(block_id + styles + role + tags + attrs)
#     block_id: str = ModelField(foreign_key=True)
#     styles: list[str] | None = ModelField(default=None)
#     role: str | None = ModelField(default=None)
#     tags: list[str] | None = ModelField(default=None)
#     attrs: dict | None = ModelField(default=None)
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

#     block: "BlockModel" = RelationField(primary_key="block_id", foreign_key="id")
#     nodes: list["BlockNode"] = RelationField(foreign_key="signature_id")


# class BlockNode(Model):
#     """
#     Lightweight tree structure node referencing block signatures.
#     Only stores tree_id + path + signature reference.
#     """
#     _namespace_name: str = "block_nodes"
#     id: int = KeyField(primary_key=True)
#     tree_id: int = ModelField(foreign_key=True)
#     path: str = ModelField(db_type="LTREE")
#     signature_id: str = ModelField(foreign_key=True)

#     signature: "BlockSignature" = RelationField(primary_key="signature_id", foreign_key="id")
#     tree: "BlockTree" = RelationField(primary_key="tree_id", foreign_key="id")

#     @classmethod
#     async def block_query(cls, cte):
#         from ..block_models.block_log import pack_block
#         from ..sql.queries import Column
#         records = await cls.query([
#             Column("styles", "bs"),
#             Column("role", "bs"),
#             Column("tags", "bs"),
#             Column("path", "bn"),
#             Column("attrs", "bs"),
#             Column("content", "bm"),
#             Column("json_content", "bm"),
#         ], alias="bn") \
#         .use_cte(cte,"tree_cte", alias="btc") \
#         .join(BlockSignature.query(["block_id", "styles", "role", "tags", "attrs"], alias="bs"), on=("signature_id", "id")) \
#         .join(BlockModel.query(["content", "json_content"], alias="bm"), on=("block_id", "id"), prefix="bs") \
#         .where(lambda b: (b.tree_id == RawValue("btc.id"))).json()
#         return pack_block(records)
     
        
# class BlockTree(VersionedModel):
#     _artifact_kind: ArtifactKind = "block"
#     id: int = KeyField(primary_key=True)
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
#     nodes: List[BlockNode] = RelationField(foreign_key="tree_id")
#     span_id: int | None = ModelField(foreign_key=True)
#     _block: "Block | None" = None
    
    
#     @classmethod
#     def query(
#         cls: Type[Self], 
#         include_branch_turn: bool = False,
#         alias: str | None = None, 
#         use_ctx: bool = True,
#         branch: Branch | int | None = None,
#         turn_cte: "PgSelectQuerySet[Turn] | None" = None,
#         use_liniage: bool = True,
#         limit: int | None = None, 
#         offset: int | None = None, 
#         statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
#         direction: Literal["asc", "desc"] = "desc",

#     ):
#         from ..sql2.pg_query_builder import select, PgQueryBuilder
#         from ..block_models.block_log import load_block_dump
#         async def to_block(trees: list[BlockTree]):
#             blocks = []
#             for tree in trees:
#                 dump = tree.model_dump()
#                 blocks.append(load_block_dump(dump["nodes"], artifact_id=tree.artifact_id))
#             return blocks
#         query =(
#             super()
#             .query(
#                 include_branch_turn=include_branch_turn,
#                 alias=alias,
#                 use_ctx=use_ctx,
#                 branch=branch,
#                 turn_cte=turn_cte,
#                 use_liniage=use_liniage,
#                 limit=limit,
#                 offset=offset,
#                 statuses=statuses,
#             )
#             .include(
#                 BlockNode.query(alias="bn")
#                     # .order_by("id")
#                     .include(
#                         BlockSignature.query(alias="bs").include(BlockModel)
#                     )
#                 ).order_by("created_at")
#             .parse(to_block, target="models")
#         )
#         return query
    

class BlockSpan(Model):
    """
    Content-addressed span storage.

    Deduplicated by hash of (prefix + content + postfix + chunks).
    Stores both text (for queries) and structured chunks (for reconstruction).
    """
    _namespace_name: str = "block_spans"

    id: str = KeyField(primary_key=True)  # SHA256 hash
    prefix_text: str = ModelField(default="")
    content_text: str = ModelField(default="")
    postfix_text: str = ModelField(default="")
    prefix_chunks: dict = ModelField(default_factory=list)  # [{content, logprob}, ...]
    content_chunks: dict = ModelField(default_factory=list)
    postfix_chunks: dict = ModelField(default_factory=list)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)


class BlockModel(Model):
    """
    Merkle tree node - content-addressed by span + metadata + children.

    The id is a hash that includes children's IDs, so identical subtrees
    automatically have identical hashes and share storage.
    """
    _namespace_name: str = "blocks"

    id: str = KeyField(primary_key=True)  # Merkle hash
    span_id: str | None = ModelField(default=None, foreign_key=True, foreign_cls=BlockSpan)  # FK to spans
    role: str | None = ModelField(default=None)
    tags: list[str] = ModelField(default_factory=list)
    styles: list[str] = ModelField(default_factory=list)
    name: str | None = ModelField(default=None)  # For BlockSchema
    type_name: str | None = ModelField(default=None)  # For BlockSchema
    attrs: dict = ModelField(default_factory=dict)
    children: list[str] = ModelField(default_factory=list)  # Ordered block IDs
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    span: BlockSpan | None = RelationField(primary_key="span_id", foreign_key="id")


class BlockTree(VersionedModel):
    """
    Root reference for a block tree, linked to versioning system.
    """
    _namespace_name: str = "block_trees"
    _artifact_kind: ArtifactKind = "block"

    id: int = KeyField(primary_key=True)
    root_id: str = ModelField(foreign_key=True, foreign_cls=BlockModel)  # FK to blocks
    span_id: int | None = ModelField(default=None, foreign_key=True)  # Execution span
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    root: BlockModel | None = RelationField(primary_key="root_id", foreign_key="id")


    @classmethod
    def query(
        cls: Type[Self], 
        include_branch_turn: bool = False,
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        turn_cte: "PgSelectQuerySet[Turn] | None" = None,
        use_liniage: bool = True,
        limit: int | None = None, 
        offset: int | None = None, 
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        direction: Literal["asc", "desc"] = "desc",

    ):
        from ..sql2.pg_query_builder import select, PgQueryBuilder
        from ..block_models.block11_storage import load_block
        async def to_block(trees: list[BlockTree]):
            blocks = []
            for tree in trees:
                dump = tree.model_dump()
                # blocks.append(load_block_dump(dump["nodes"], artifact_id=tree.artifact_id))
            return blocks
        query =(
            super()
            .query(
                include_branch_turn=include_branch_turn,
                alias=alias,
                use_ctx=use_ctx,
                branch=branch,
                turn_cte=turn_cte,
                use_liniage=use_liniage,
                limit=limit,
                offset=offset,
                statuses=statuses,
            )
            .include(
                BlockModel.query(alias="bn")
                    # .order_by("id")
                    .include(
                        BlockSpan.query(alias="bs").include(BlockModel)
                    )
                ).order_by("created_at")
            .parse(to_block, target="models")
        )
        return query




   


class ArtifactModel(VersionedModel):
    """VersionedModel with artifact tracking."""
    _is_base = True
    _dirty_fields: dict[str, Any] = {}
    # es_id: int = KeyField(primary_key=True)
    artifact_id: int = KeyField(primary_key=True)
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
            result = await self._build_insert_query(branch, turn, self.artifact.version + 1, exclude={"artifact_id", "artifact"})
            # obj = self.model_copy(update={"turn_id": None, "branch_id": None})
            # obj.version += 1
            # return await obj._super_save()
        else:
            result = await super().save()
        self._dirty_fields = {}
        return result
    

    
    @classmethod
    def query(
        cls: Type[Self], 
        fields: list[str] | None = None, 
        alias: str | None = None, 
        use_ctx: bool = True,
        branch: Branch | int | None = None,
        turn_cte: "PgSelectQuerySet[Turn] | None" = None,
        use_liniage: bool = True,
        limit: int | None = None, 
        offset: int | None = None, 
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
        direction: Literal["asc", "desc"] = "desc",
        include_branch_turn: bool = False,
        **kwargs
    ) -> "PgQueryBuilder[Self]":
        from ..sql2.pg_query_builder import PgQueryBuilder
        
        art_cte = Artifact.query(
            statuses=statuses, 
            limit=limit, 
            offset=offset, 
            direction=direction, 
            use_liniage=use_liniage, 
            turn_cte=turn_cte,
            include_branch_turn=include_branch_turn
        )
        return (
            PgQueryBuilder()
            .select(cls)
            .distinct_on("id")
            .order_by("-id", "-version")
            .join_cte(art_cte, "artifact_cte", alias="ac")
        )
        # return (
        #     PgSelectQuerySet(cls, alias=alias) \
        #     .use_cte(
        #         art_cte,
        #         name="artifact_cte",
        #         alias="ac",
        #     )
        #     .select(*fields if fields else "*")
        # )

    
        
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


class DataArtifact(Model):
    id: int = KeyField(primary_key=True)  # Auto-increment ID
    value_id: int = ModelField(foreign_key=True)
    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    kind: ArtifactKindEnum = ModelField()
    position: int | None = ModelField(default=None)  # For lists/tuples - index in collection
    name: str | None = ModelField(default=None)  # For dicts - key name
    

class DataFlowNode(Model):
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    kind: ArtifactKindEnum = ModelField()
    io_kind: ValueIOKind = ModelField()
    path: str = ModelField(db_type="LTREE")

    # Parent: exactly one of span_id or turn_id must be set
    span_id: int | None = ModelField(foreign_key=True, default=None)
    turn_id: int | None = ModelField(foreign_key=True, foreign_cls=Turn, default=None)

    artifact_id: int = ModelField(foreign_key=True, foreign_cls=Artifact)
    artifacts: list[Artifact] = RelationField(
        primary_key="id",
        junction_keys=["value_id", "artifact_id"],
        foreign_key="id",
        junction_model=DataArtifact,
    )
    artifact_data: list[DataArtifact] = RelationField(
        primary_key="id",
        foreign_key="value_id",
    )
    
    alias: str | None = ModelField(default=None)
    name: str | None = ModelField(default=None)  # Keyword argument name (e.g., "count", "items")
    
    
    _value: Any | None = None
    _container_value: Artifact | None = None
    
    
    @property
    def value(self) -> Any:
        from ...block import Block
        if self._value is None:
            raise ValueError(f"no value for DataFlowNode {self.id}")
        
        def get_value(target: Any) -> Any:
            if isinstance(target, Block):
                return target
            if target.kind == "parameter":
                return target._value.value
            elif target.kind == "block":
                return target._value
            return target._value
        if self.kind == "list":
            return [get_value(v) for v in self._value]
        return get_value(self)
    
    @property
    def list_kind(self) -> list[str] | None:
        if self.kind != "list" or self._value is None:
            return None
        return [v.kind for v in self._value]

    @property
    def value_or_none(self) -> Any | None:
        return self._value

    @property
    def index(self) -> int:
        """Extract index from path (last segment)"""
        return int(self.path.split('.')[-1])
    
    
    
    def model_dump(self, *args, **kwargs) -> dict:
        dump = super().model_dump(*args, **kwargs)
        dump["list_kind"] = None
        if self.kind == "list":
            if self._value is None:
                dump["value"] = []
                dump["list_kind"] = []
            else:
                dump["value"] = [v.model_dump() for v in self.value]
                dump["list_kind"] = self.list_kind
        elif self._value is not None and hasattr(self._value, "model_dump"):     
            dump["value"] = self._value.model_dump()
        elif self.kind == "parameter":
            dump["value"] = self._value
        else:
            dump["value"] = None
        return dump
   

class ExecutionSpan(VersionedModel):
    """Represents a single execution unit (component call, stream, etc.)"""
    _artifact_kind: ArtifactKind = "span"
    id: int = KeyField(primary_key=True)
    name: str = ModelField()  # Function/component name
    path: str = ModelField(db_type="LTREE")
    span_type: SpanType = ModelField()
    parent_span_id: int | None = ModelField(foreign_key=True, self_ref=True)
    start_time: dt.datetime = ModelField(default_factory=dt.datetime.now)
    end_time: dt.datetime | None = ModelField(default=None)
    tags: list[str] | None = ModelField(default=None)
    metadata: dict[str, Any] = ModelField(default={})
    status: Literal["running", "completed", "failed"] = ModelField(default="running")
    turn_id: int = ModelField(foreign_key=True, foreign_cls=Turn)
    
    # Relations
    data: List["DataFlowNode"] = RelationField([], foreign_key="span_id")
    artifacts: List[Artifact] = RelationField(foreign_key="span_id")
    block_trees: List[BlockTree] = RelationField(foreign_key="span_id")
    
    _parent_value: "DataFlowNode | None" = None
    
    @property
    def parent_value(self) -> "DataFlowNode | None":
        return self._parent_value

    @property
    def inputs(self) -> dict[str, "DataFlowNode"]:
        return {v.path.split(".")[-1]: v.value for v in self.data if v.io_kind == "input"}
    
    @property
    def outputs(self) -> list["DataFlowNode"]:
        return [v.value for v in self.data if v.io_kind == "output"]
    
    @property
    def children(self) -> list["ExecutionSpan"]:
        return [v.value for v in self.data if v.kind == "span"]
    
    @property
    def ctx(self) -> "Context":
        from ...prompt.context import Context
        ctx = Context.current_or_none()
        if ctx is None:
            raise ValueError("Context is not set")
        return ctx
    
    
    async def log_value(self, target: Any, alias: str | None = None, io_kind: ValueIOKind = "output", name: str | None = None):
        # from ...prompt.context import Context
        from ...model.versioning.artifact_log import ArtifactLog
        return await ArtifactLog.log_value(target, alias, io_kind, name)
    
    
    
    # async def add_child(self, name: str, span_type: SpanType = "component", tags: list[str] = []):
    #     """
    #     Add a child span to the current span by logging it as a span value.
    #     The child will be accessible via the computed 'children' property.
    #     """
    #     # Compute child path
    #     from ...prompt.context import Context
    #     child_index = len(self.children)
    #     child_path = f"{self.path}.{child_index}"        
        

    #     # Create child ExecutionSpan directly
    #     child_span = await ExecutionSpan(
    #         name=name,
    #         span_type=span_type,
    #         tags=tags,
    #         path=child_path,
    #         parent_span_id=self.id
    #     ).save()        

    #     # Log the SpanTree as a value - this adds it to _values
    #     await self.ctx.artifact_log.log_value(self, child_span, io_kind="output")

    #     return child_span

    






class EvaluationFailure(Exception):
    """Exception raised when an evaluation fails and should stop execution."""
    pass


class EvaluatorConfig(Model):
    """
    Configuration for a value evaluator.

    Evaluators can match values by:
    - path_pattern: LTREE path pattern for SpanValue.path (e.g., "1.*", "*.0", "1.2.3")
    - tags: List of tags that the parent span must have
    - span_name: Exact name match for the parent span
    - value_name: Match values by their name field

    All criteria are AND-ed together (value must match all specified criteria).
    """
    id: int = KeyField(primary_key=True)
    name: str = ModelField(..., description="Evaluator function name")
    path: str = ModelField(..., db_type="LTREE", description="LTREE path pattern for SpanValue.path (e.g., '1.*', '*.0', '1.2.3')")
    tags: list[str] = ModelField(default=[], description="Match values whose parent span has these tags")
    span_name: str | None = ModelField(None, description="Match values whose parent span has this name")
    # value_name: str | None = Field(None, description="Match values by their name field")
    metadata: dict = ModelField(default={}, description="Additional evaluator configuration")
    test_turn_id: int = ModelField(foreign_key=True, description="Parent test turn")


class ValueEval(Model):
    """
    Result of evaluating a single value.

    Each value evaluation records:
    - Which value was evaluated (value_id, path)
    - Which evaluator was used (evaluator name)
    - The score and metadata from the evaluator
    - Link to the parent turn evaluation
    """
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    # What was evaluated
    turn_eval_id: int = ModelField(foreign_key=True, description="Parent turn evaluation")
    value_id: int = ModelField(foreign_key=True, foreign_cls=DataFlowNode, description="The SpanValue that was evaluated")
    path: str = ModelField(db_type="LTREE", description="Value path for querying")

    # Evaluation results
    evaluator: str = ModelField(description="Evaluator function name")
    score: float | None = ModelField(default=None, description="Evaluation score")
    metadata: dict = ModelField(default={}, description="Additional evaluation data")

    # Status
    status: str = ModelField(default="completed", description="Status: completed, failed, skipped")
    error: str | None = ModelField(default=None, description="Error message if evaluation failed")


class TurnEval(Model):
    """
    Evaluation results for a single turn in a test run.

    Links a test turn to a reference turn and stores all value evaluations.
    """
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    test_turn_id: int = ModelField(description="The test turn being evaluated")
    ref_turn_id: int = ModelField(description="The reference turn to compare against")
    test_run_id: int = ModelField(foreign_key=True, description="Parent test run")

    score: float | None = ModelField(default=None, description="Average score across all value evaluations")
    value_evals: Tree[ValueEval] = RelationField(foreign_key="turn_eval_id")
    trace_id: str = ModelField(default="", description="Trace ID for debugging")


class TestRun(Model):
    """
    A single execution of a test case.

    Tracks the overall status and score of running a test case.
    """
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    test_case_id: int = ModelField(foreign_key=True, description="Parent test case")
    branch_id: int = ModelField(default=1, description="Branch this test run belongs to")

    score: float | None = ModelField(default=None, description="Overall score of the test run")
    status: Literal["pending", "running", "success", "failure"] = ModelField(
        default="pending",
        description="Status of the test run"
    )

    turn_evals: List[TurnEval] = RelationField(foreign_key="test_run_id")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.status = "failure"
        else:
            self.status = "success"
        await self.save()


class TestTurn(Model):
    """
    A reference turn with evaluator configurations.

    Maps a reference turn to evaluators that should run during test execution.
    """
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    test_case_id: int = ModelField(foreign_key=True, description="Parent test case")
    turn_id: int = ModelField(foreign_key=True, foreign_cls=Turn, description="Reference turn ID")

    evaluators: Tree[EvaluatorConfig] = RelationField(
        foreign_key="test_turn_id",
        description="Evaluators to run for values in this turn"
    )
    turn: Turn = RelationField(foreign_key="id", primary_key="turn_id")
    
    
    @property
    def inputs(self) -> list[DataFlowNode]:
        return self.turn.inputs
    
    def get_kwargs(self) -> dict[str, Any]:
        return self.turn.get_kwargs()
    
    def get_args(self) -> list[Any]:
        return self.turn.get_args()


class TestCase(Model):
    """
    A test case containing reference turns and evaluator configurations.

    Test cases define what to test and how to evaluate it.
    """
    id: int = KeyField(primary_key=True)
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)
    updated_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

    title: str = ModelField(default="", description="Test case title")
    description: str = ModelField(default="", description="Test case description")
    branch_id: int = ModelField(default=1, foreign_key=True, description="Branch this test case belongs to", foreign_cls=Branch)
    user_id: uuid.UUID = ModelField(description="User who created this test case")

    test_turns: List[TestTurn] = RelationField(foreign_key="test_case_id")
    test_runs: List[TestRun] = RelationField(foreign_key="test_case_id")
