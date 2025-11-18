
import hashlib
import json
import uuid
from typing import Any, Dict, List, Literal, Optional


from ...block import BaseBlock, Block, BlockChunk, BlockList, BlockSent, AttrBlock
from ..versioning.models import Artifact, Branch, DataFlowNode
from ..sql.expressions import RawValue
from ..sql.queries import Column
from ...utils.db_connections import PGConnectionManager
import datetime as dt
from ..versioning.models import BlockTree, BlockNode, BlockModel, BlockSignature, ExecutionSpan, TurnStatus



def block_hash(content: Optional[str] = None, json_content: Optional[dict] = None) -> str:
    if content is not None:
        data = content.encode("utf-8")
    else:
        data = json.dumps(json_content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def signature_hash(
    block_id: str,
    styles: list[str] | None,
    role: str | None,
    tags: list[str] | None,
    attrs: dict | None
) -> str:
    """
    Create a unique hash for a block signature (content + styling).
    """
    data = {
        "block_id": block_id,
        "styles": sorted(styles) if styles else None,
        "role": role,
        "tags": sorted(tags) if tags else None,
        "attrs": attrs
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def flatten_tree(data: Dict[str, Any], base_path: str = "1") -> List[Dict[str, Any]]:
    flat = [{
        "path": base_path,
        "content": data.get("content"),
        "json_content": data.get("json_content"),
        "style": data.get("style", {}),
    }]
    for idx, child in enumerate(data.get("children", []), start=1):
        flat.extend(flatten_tree(child, f"{base_path}.{idx}"))
    return flat


def dump_chunk(chunk: BlockChunk):
    # dump = {
    #     "content": chunk.content,
    # }
    # if chunk.logprob is not None:
    #     dump["logprob"] = chunk.logprob
    # if chunk.prefix is not None:
    #     dump["prefix"] = chunk.prefix
    # if chunk.postfix is not None:
    #     dump["postfix"] = chunk.postfix
    return {
        "content": chunk.content,
        "logprob": chunk.logprob,
        "prefix": chunk.prefix,
        "postfix": chunk.postfix,
    }

def dump_sent(sent: BlockSent):   
    chunks = []    
    for chunk in sent.children:        
        chunks.append(dump_chunk(chunk))
    return {
        # "has_eol": sent.has_eol,
        "content": sent.content,
        "children": chunks,
    }
    
def load_sent_dump(dump: dict):
    sent = BlockSent(
        content=dump["content"],
    )
    for child in dump["children"]:
        sent.append(
            BlockChunk(
                content=child["content"],
                logprob=child["logprob"],
            )       
        )
    return sent

def dump_attrs(attrs: dict[str, str | AttrBlock]):
    return {k: v.model_dump() if isinstance(v, AttrBlock) else v for k, v in attrs.items()}

def load_attrs(attrs: dict[str, str | dict]):
    return {k: AttrBlock.from_dict(v) if isinstance(v, dict) else v for k, v in attrs.items()}

def dump_block(blk: Block):
    dumps = []
    for blk in blk.traverse():
        dump = {}
        dump["content"] = blk.content.render()
        dump["json_content"] = {
            "content": dump_sent(blk.content),
            "prefix": dump_sent(blk.prefix) if blk.prefix is not None else None,
            "postfix": dump_sent(blk.postfix) if blk.postfix is not None else None,
        }
        dump["styles"] = blk.styles
        dump["path"] = ".".join(str(p) for p in [0] +blk.path)
        dump["tags"] = blk.tags
        dump["role"] = blk.role
        dump["attrs"] = dump_attrs(blk.attrs)
        dumps.append(dump)
    return dumps


def load_block_dump(dumps: list[dict], artifact_id: int | None = None):
    """
    Load blocks from new Option 1 schema structure:
    Each dump contains: path, signature -> (styles, role, tags, attrs, block -> (content, json_content))
    """
    block_lookup = {}
    for dump in dumps:
        signature = dump['signature']
        block_data = signature['block']

        blk = Block(
            load_sent_dump(block_data["json_content"]['content']),
            role=signature["role"],
            styles=signature["styles"],
            attrs=load_attrs(signature["attrs"]),
            tags=signature["tags"],
            prefix=load_sent_dump(block_data["json_content"]['prefix']) if block_data["json_content"]['prefix'] is not None else None,
            postfix=load_sent_dump(block_data["json_content"]['postfix']) if block_data["json_content"]['postfix'] is not None else None,
            artifact_id=artifact_id,
        )
        block_lookup[dump["path"]] = blk

    root_blk = block_lookup.pop("0")
    for p in sorted(block_lookup.keys()):
        blk = block_lookup[p]
        path = [int(p_i) for p_i in p.split(".")]
        root_blk.insert(blk, path[1:])

    return root_blk


  

async def insert_block(block: Block, branch_id: int, turn_id: int, span_id: int | None = None) -> BlockTree:
    """
    Insert block using Option 1 (Block Signatures) architecture.
    1. Deduplicate block content in `blocks` table
    2. Deduplicate content+styling in `block_signatures` table
    3. Store only tree structure in `block_nodes` table
    """
    from ..versioning.models import BlockTree
    nodes = dump_block(block)

    async def insert_block_transaction(tx, tree_id: int):
        # --- Step 1: Prepare and insert blocks (content only) ---
        block_rows = []
        seen_blocks = set()
        for node in nodes:
            blk_id = block_hash(node["content"], node["json_content"])
            if blk_id not in seen_blocks:
                block_rows.append((blk_id, node["content"], json.dumps(node["json_content"])))
                seen_blocks.add(blk_id)

        if block_rows:
            await tx.executemany(
                "INSERT INTO blocks (id, content, json_content) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING",
                block_rows
            )

        # --- Step 2: Prepare and insert block signatures (content + styling) ---
        signature_rows = []
        seen_signatures = set()
        node_signatures = {}  # Map node index -> signature_id

        for idx, node in enumerate(nodes):
            blk_id = block_hash(node["content"], node["json_content"])
            sig_id = signature_hash(
                blk_id,
                node["styles"],
                node["role"],
                node["tags"],
                node["attrs"]
            )
            node_signatures[idx] = sig_id

            if sig_id not in seen_signatures:
                styles_array = node["styles"] if node["styles"] is not None else []
                tags_array = node["tags"] if node["tags"] is not None else []
                signature_rows.append((
                    sig_id,
                    blk_id,
                    styles_array,
                    node["role"],
                    tags_array,
                    json.dumps(node["attrs"])
                ))
                seen_signatures.add(sig_id)

        if signature_rows:
            await tx.executemany(
                "INSERT INTO block_signatures (id, block_id, styles, role, tags, attrs) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (id) DO NOTHING",
                signature_rows
            )

        # --- Step 3: Insert block nodes (tree structure only) ---
        node_rows = []
        for idx, node in enumerate(nodes):
            sig_id = node_signatures[idx]
            node_rows.append((
                tree_id,
                node["path"],
                sig_id
            ))

        if node_rows:
            await tx.executemany(
                "INSERT INTO block_nodes (tree_id, path, signature_id) VALUES ($1, $2::ltree, $3)",
                node_rows
            )

        return tree_id

    block_tree = await BlockTree().save()
    async def tx_wrapper(tx):
        return await insert_block_transaction(tx, block_tree.id)
    result = await PGConnectionManager.run_in_transaction(tx_wrapper)
    return block_tree

    
    
    
    
async def get_blocks(art_ids: list[str], dump_models: bool = True, include_branch_turn: bool = False) -> dict[str, Block]:
    """
    Retrieve blocks using the new Option 1 schema:
    BlockTree -> BlockNode -> BlockSignature -> BlockModel
    """
    if not art_ids:
        return {}
    block_trees = await BlockTree.query(alias="bt", include_branch_turn=include_branch_turn).select("*").include(
            BlockNode.query(alias="bn").select("*").include(
                BlockSignature.query(alias="bs").select("*").include(
                    BlockModel.query(alias="bm").select("*")
                )
            )
        ).where(lambda b: b.artifact_id.isin(art_ids)).order_by("-artifact_id").json()
    blocks_lookup = {}
    for tree in block_trees:
        # tree_id = tree["id"]
        art_id = tree["artifact_id"]
        block = load_block_dump(tree["nodes"])
        blocks_lookup[art_id] = block.model_dump() if dump_models else block
    return blocks_lookup







class BlockLogQuery:
    
    def __init__(
        self,
        limit: int | None = None,
        offset: int | None = None,
        direction: Literal["asc", "desc"] = "desc",
        span_name: str | None = None,
        statuses: list[TurnStatus] = [TurnStatus.COMMITTED, TurnStatus.STAGED],
    ):
        self.limit = limit or 5
        self._offset = offset
        self.direction = direction
        self.span_name = span_name
        self.statuses = statuses
        self.include_roles = None
        self.exclude_roles = None
        # self.query = self._build_block_query() if not span_name else self._build_span_query()
        
    def __await__(self):
        return self.execute().__await__()
    
    async def execute(self) -> List[Block]:
        query = self._build_block_query()
        artifact_ids = []
        if self.span_name:
            span_query = self._build_span_query(self.span_name)
            spans = await span_query
            artifact_ids = [a.id for span in spans for sv in span.data for a in sv.artifacts]
            if not artifact_ids:
                return []
        def tree_to_block(tree):
            # print(tree)
            if not tree['nodes']:
                return None
            return load_block_dump(tree['nodes'])
        if artifact_ids:
            query = query.where(lambda x: x.artifact_id.isin(artifact_ids))
        query = query.print().parse(tree_to_block)
        results = await query.json()
        
        if self.include_roles:
            results = [b for b in results if b.role in self.include_roles]
        if self.exclude_roles:
            results = [b for b in results if b.role not in self.exclude_roles]
        return results
    
    async def json(self) -> List[dict]:
        query = self._build_block_query()
        artifact_ids = []
        if self.span_name:
            span_query = self._build_span_query(self.span_name)
            spans = await span_query
            artifact_ids = [a.id for span in spans for sv in span.data for a in sv.artifacts]
        if artifact_ids:
            query = query.where(lambda x: x.artifact_id.isin(artifact_ids))
        results = await query.json()
        return results

    
    
    def _build_block_query(self):
        """Build query using Option 1 schema: BlockTree -> BlockNode -> BlockSignature -> BlockModel"""
        return BlockTree.query(
            alias="bt",
            limit=self.limit,
            offset=self._offset,
            direction=self.direction,
            statuses=self.statuses
        ).include(
            BlockNode.query(alias="bn").order_by("id").include(
                BlockSignature.query(alias="bs").include(BlockModel)
            )
        ).order_by("created_at")
        
    def _build_span_query(self, name: str):
        return ExecutionSpan.query().include(
            DataFlowNode.query().include(
                Artifact
            )
        ).where(name=name)
        

        
    def where(self, span: str | None = None):
        if span:
            self.span_name = span
        return self
    
    def role(self, exclude: set[str] | None = None, include: set[str] | None = None):
        self.include_roles = include
        self.exclude_roles = exclude
        return self
    
    def status(self, statuses: list[TurnStatus]):
        self.statuses = statuses
        return self
    
    def last(self, limit: int):
        self.limit = limit
        return self
    
    def offset(self, offset: int):
        self._offset = offset
        return self
    
    def all(self):
        self.statuses = []
        return self
    
    def span(self, span: str):
        self.span_name = span
        return self
    
    def print(self):
        query = self._build_block_query()
        return query.print()
        

class BlockLog:
    
    def __init__(self):
        self.query = None
        
        
    @classmethod
    def last(cls, limit: int) -> BlockLogQuery:
        return BlockLogQuery(limit=limit)

    @classmethod
    def span(cls, span: str) -> BlockLogQuery:
        return BlockLogQuery(span_name=span)
    
    

    
    @classmethod
    async def add(cls, block: Block, branch_id: int | None = None, turn_id: int | None = None, span_id: int | None = None):
        from ..versioning.models import Turn, Branch
        if branch_id is None:
            branch_id = Branch.current().id
        if turn_id is None:
            turn_id = Turn.current().id
        if span_id is None:
            curr_span = ExecutionSpan.current_or_none()
            if curr_span is not None:
                span_id = curr_span.id
        return await insert_block(block, branch_id, turn_id, span_id)
    
    