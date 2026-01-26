from contextlib import asynccontextmanager
import enum
import uuid
import datetime as dt
import contextvars
from typing import TYPE_CHECKING, AsyncGenerator, Callable, List, Literal, Type, TypeVar, Self, Any

from pydantic import BaseModel, Field

from promptview.model.base.types import ArtifactKind
from promptview.model.db_types import Tree

from ..model.sql2.expressions import Raw
from ..model.sql2.relations import RawRelation

from promptview.utils.type_utils import SerializableType, UnknownType, deserialize_value, serialize_value, type_to_str, str_to_type, type_to_str_or_none



from ..model.model3 import Model
from ..model.fields import KeyField, ModelField, RelationField
from ..model.postgres2.pg_query_set import PgSelectQuerySet
from ..model.postgres2.rowset import RowsetNode
from ..model.sql.queries import CTENode, RawSQL
from ..model.sql.expressions import RawValue
from ..utils.db_connections import PGConnectionManager



if TYPE_CHECKING:
    from ..block import Block
    from ..model.sql2.relational_queries import SelectQuerySet
    from ..model.sql2.pg_query_builder import PgQueryBuilder
    from .artifact_log import ArtifactLog
    from ..prompt.context import Context
    from .dataflow_models import ExecutionSpan, DataFlowNode

# ContextVars for current branch/turn
_curr_branch = contextvars.ContextVar("curr_branch", default=None)
_curr_turn = contextvars.ContextVar("curr_turn", default=None)



SpanType = Literal["component", "stream", "llm", "evaluator", "turn"]
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
    
    def extract_blocks(self):
        for data in self.data:
            data.extract()
    
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
        from ..model.sql2.pg_query_builder import PgQueryBuilder, select
        from ..versioning.artifact_log import ArtifactLog
        from .dataflow_models import DataFlowNode, DataArtifact
        
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
        from ..model.postgres2.pg_query_set import PgSelectQuerySet
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
        from ..prompt.context import Context, ContextError
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
        from ..model.sql2.pg_query_builder import select, PgQueryBuilder
        
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
            .include(Artifact)
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
        from ..model.postgres2.pg_query_set import PgSelectQuerySet
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
    
    
        
    

    

# class BlockSpan(Model):
#     """
#     Content-addressed text and chunk storage for Block12.

#     Each block's text and chunks are stored together, deduplicated by
#     content hash. Chunk positions are relative to the block's local text.
#     """
#     _namespace_name: str = "block_spans"

#     id: str = KeyField(primary_key=True)  # SHA256 hash of text + chunks
#     text: str = ModelField(default="")
#     chunks: list[dict] = ModelField(default_factory=list)  # [{id, start, end, logprob, style}, ...]
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)



# class BlockTreeBlock(Model):
#     """
#     Junction table for BlockTree ↔ BlockModel many-to-many relationship.

#     Enables:
#     - A BlockTree to contain multiple BlockModels
#     - A BlockModel to appear in multiple BlockTrees (deduplication)
#     - Ordered blocks within a tree via position
#     - Efficient reverse lookups ("which trees contain this block?")
#     """
#     _namespace_name: str = "block_tree_blocks"

#     id: int = KeyField(primary_key=True)
#     tree_id: int = ModelField(foreign_key=True, index="btb_tree_idx")
#     block_id: str = ModelField(foreign_key=True, index="btb_block_idx")
#     position: int = ModelField(default=0, order_by=True)  # Order within the tree
#     is_root: bool = ModelField(default=False)  # Marks the root block of the tree
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

# class BlockModel(Model):
#     """
#     Merkle tree node - content-addressed by span + metadata + children.

#     The id is a hash that includes children's IDs, so identical subtrees
#     automatically have identical hashes and share storage.
#     """
#     _namespace_name: str = "blocks"

#     id: str = KeyField(primary_key=True)  # Merkle hash
#     span_id: str | None = ModelField(default=None, foreign_key=True, foreign_cls=BlockSpan)  # FK to spans
#     role: str | None = ModelField(default=None)
#     tags: list[str] = ModelField(default_factory=list)
#     styles: list[str] = ModelField(default_factory=list)
#     name: str | None = ModelField(default=None)  # For BlockSchema
#     type_name: str | None = ModelField(default=None)  # For BlockSchema
#     attrs: dict = ModelField(default_factory=dict)
#     children: list[str] = ModelField(default_factory=list)  # Ordered block IDs
#     is_rendered: bool = ModelField(default=False)  # Mutator render state
#     block_type: str = ModelField(default="block")  # "block", "schema", "list", "list_schema"
#     item_name: str | None = ModelField(default=None)  # For BlockListSchema
#     key: str | None = ModelField(default=None) # For BlockListSchema
#     path: str = ModelField(default="")  # Index path e.g., "0.1.2"
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)

#     span: BlockSpan | None = RelationField(primary_key="span_id", foreign_key="id")

#     # Reverse relation: which trees contain this block (via junction table)
#     # Note: BlockTreeBlock and BlockTree defined below, uses forward reference
#     trees: List["BlockTree"] = RelationField(
#         primary_key="id",
#         foreign_key="id",
#         junction_keys=["block_id", "tree_id"],
#         junction_model=BlockTreeBlock  # Forward reference as string
#     )



# class BlockTree(VersionedModel):
#     """
#     Root reference for a block tree, linked to versioning system.
#     """
#     _namespace_name: str = "block_trees"
#     _artifact_kind: ArtifactKind = "block"

#     id: int = KeyField(primary_key=True)
#     span_id: int | None = ModelField(default=None, foreign_key=True)  # Execution span
#     created_at: dt.datetime = ModelField(default_factory=dt.datetime.now, order_by=True)

#     blocks: List[BlockModel] = RelationField(
#         primary_key="id",
#         foreign_key="id",
#         junction_keys=["tree_id", "block_id"],
#         junction_model=BlockTreeBlock
#     )


#     @classmethod
#     def query2(
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
#         from ..model.sql2.pg_query_builder import select, PgQueryBuilder
#         async def to_block(trees: list[BlockTree]):
#             blocks = []
#             for tree in trees:
#                 dump = tree.model_dump()
#                 # blocks.append(load_block_dump(dump["nodes"], artifact_id=tree.artifact_id))
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
#                 BlockModel.query(alias="bn")
#                     # .order_by("id")
#                     .include(
#                         BlockSpan.query(alias="bs").include(BlockModel)
#                     )
#                 ).order_by("created_at")
#             # .parse(to_block, target="models")
#         )
#         return query




   


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
        from ..model.sql2.pg_query_builder import PgQueryBuilder
        
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

    
    # @classmethod
    # def query(
    #     cls: Type[Self], 
    #     fields: list[str] | None = None, 
    #     alias: str | None = None, 
    #     use_ctx: bool = True,
    #     branch: Branch | int | None = None,
    #     turn_cte: "PgSelectQuerySet[Turn] | None" = None,
    #     use_liniage: bool = True,
    #     limit: int | None = None, 
    #     offset: int | None = None, 
    #     statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
    #     direction: Literal["asc", "desc"] = "desc",
    #     include_branch_turn: bool = False,
    #     **kwargs
    # ) -> "PgQueryBuilder[Self]":
    #     from ..sql2.pg_query_builder import PgQueryBuilder
        
    #     art_cte = Artifact.query(
    #         statuses=statuses,
    #         limit=limit,
    #         offset=offset,
    #         direction=direction,
    #         use_liniage=use_liniage,
    #         turn_cte=turn_cte,
    #         include_branch_turn=include_branch_turn
    #     )
    #     ns = cls.get_namespace()
    #     table_name = ns.name
    #     pk = ns.primary_key
    #     return (
    #         PgQueryBuilder()
    #         .select(cls)
    #         .distinct_on(f"{table_name}.{pk}")
    #         .order_by(f"-{table_name}.{pk}", "-ac.version")
    #         .join_cte(art_cte, "artifact_cte", alias="ac")
    #     )

    
        
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
        from ..model.postgres2.pg_query_set import PgSelectQuerySet

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


# NOTE: model_rebuild() calls are in __init__.py after all models are imported





    





