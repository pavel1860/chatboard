from typing import Literal, Self, Type
from .model import ModelField, RelationField
from .versioning import ArtifactModel,TurnStatus, BlockTree, BlockSpan, BlockModel, Branch, Turn
from .versioning import BlockLog
from .versioning.block12_storage import parse_block_tree_query_result, dump_block
from .block import BlockSchema, Block
from pydantic import Field

from .model.postgres2.pg_query_set import PgSelectQuerySet
from .model.sql2.pg_query_builder import PgQueryBuilder
from .prompt import Context







class BlockArtifact(ArtifactModel):
    _is_base: bool = True
    tree_id: int = ModelField(description="the id of the tree that the article belongs to", foreign_key=True, foreign_cls=BlockTree)
    tree: BlockTree = RelationField(description="the tree that the article belongs to", primary_key="tree_id", foreign_key="id")    
    content: "Block" = Field(description="the content of the article", default_factory=Block)
    
    
    async def save(self, **kwargs) -> Self:
        if not self.tree_id:
            tree = await BlockLog.add(self.content)
            self.tree_id = tree.id
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
        
        async def parse_rows(rows):
            trees = [r["tree"] for r in rows]
            blocks = parse_block_tree_query_result(trees)
            for row, block in zip(rows, blocks):
                row["content"] = block
            return rows
        
        query.include(BlockTree.query().include(BlockModel.query().include(BlockSpan))).parse(parse_rows, target="rows")
        return query
