from typing import Literal, Self, Type
from .model import ModelField, RelationField
from .versioning import ArtifactModel, TurnStatus, Branch, Turn
from .versioning import BlockLog, StoredBlockModel, compute_block_hash
from .block import BlockSchema, Block
from pydantic import Field

from .model.postgres2.pg_query_set import PgSelectQuerySet
from .model.sql2.pg_query_builder import PgQueryBuilder
from .prompt import Context







class BlockArtifact(ArtifactModel):
    _is_base: bool = True
    block_id: int = ModelField(description="the id of the stored block", foreign_key=True, foreign_cls=StoredBlockModel)
    stored_block: StoredBlockModel = RelationField(description="the stored block model", primary_key="block_id", foreign_key="id")
    content: "Block" = Field(description="the content of the article", default_factory=Block)

    async def save(self, **kwargs) -> Self:
        if not self.block_id:
            stored = await BlockLog.add(self.content)
            self.block_id = stored.id
        return await super().save(**kwargs)

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
        query = super().query(
            fields=fields,
            alias=alias,
            use_ctx=use_ctx,
            branch=branch,
            turn_cte=turn_cte,
            use_liniage=use_liniage,
            limit=limit,
            offset=offset,
            statuses=statuses,
            direction=direction,
            include_branch_turn=include_branch_turn,
            **kwargs
        )

        # async def parse_rows(rows):
        #     for row in rows:
        #         stored_block = row.get("stored_block")
        #         if stored_block:                    
        #             row["content"] = stored_block.to_block()
        #     return rows

        # query.include(StoredBlockModel.query()).parse(parse_rows, target="rows")
        async def parse_models(recs):
            for rec in recs:
                stored_block = rec.stored_block
                if stored_block:                    
                    rec.content = stored_block.to_block()
            return recs

        query.include(StoredBlockModel.query(use_liniage=use_liniage)).parse(parse_models, target="models")

        return query

    
    def to_block(self, exclude: set[str] | None = None, include_artifact: bool = False) -> Block:
        from .block import Block
        from .block.block12.object_helpers import pydantic_to_schema        
        artifact_fields = {"artifact_id", "artifact", "stored_block"} if not include_artifact else set()
        exclude = exclude or set()
        with Block(self.__class__.__name__, style="md") as blk:
            for name, field in self.__class__.model_fields.items():
                if name in artifact_fields or name in exclude or name == "content":
                    continue                
                with blk.view(name, type=field.annotation, style="def") as view:
                        view /= getattr(self, name)
                        
            with blk.view("content", type=Block, style="md") as view:
                view /= self.content
        return blk
