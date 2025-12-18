from .block import BlockSchema, BlockListSchema, BlockList, BlockBase, ContentType, BlockChunk, Block
from ...utils.type_utils import UNSET, UnsetType
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .block_transformers import BlockTransformer





class BlockBuilderContext:
    
    def __init__(self, schema: "BlockSchema | None"):        
        self.schema = schema.extract_schema() if schema is not None else None        
        self._stack = []
        self._root = None
        self._block_text = None
        
        
    def init_root(self):
        self._root = self._get_schema(self.schema.name)
        self._root.instantiate()        
        self._stack.append(self._root)

    @property
    def curr_block(self) -> "BlockBase":
        if len(self._stack) == 0:
            raise RuntimeError("No block on top")
        return self._stack[-1].block
    
    @property
    def result(self) -> "Block":
        if self._root is None:
            raise RuntimeError("No block to return")
        return self._root.block
    
    def _push(self, transformer: "BlockTransformer"):
        if self._root is None:
            self._root = transformer
        else:
            transformer = self._top().append_child(transformer)
        self._stack.append(transformer)
        return transformer
        
    def _pop(self):
        if len(self._stack) == 0:
            raise RuntimeError("No block to pop")
        return self._stack.pop()
    
    def _top(self):
        if len(self._stack) == 0:
            raise RuntimeError("No block on top")
        return self._stack[-1]
    
    def _top_or_none(self):
        if len(self._stack) == 0:
            return None
        return self._stack[-1]
    

    
    def _get_schema(self, name: str):
        from .block_transformers import BlockTransformer
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        block_schema = self.schema.get_one(name)
        if block_schema is None:
            raise RuntimeError(f"Schema {name} not found")
        block_transformer = BlockTransformer.from_block_schema(block_schema)
        return block_transformer
    
    
    # def get_item_name(self, name: str) -> str:
    #     if self.schema is None:
    #         raise RuntimeError("Schema not initialized")
    #     block_schema = self.schema.get_one(name)
    #     if block_schema is None:
    #         raise RuntimeError(f"Schema {name} not found")
    #     return block_schema.get_item_name()
    
    def instantiate_list_item(
        self, 
        name: str,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
        force_schema: bool = False,
    ):
        block_schema = self.schema.get_one(name)
        if not block_schema.is_list_item:
            raise RuntimeError(f"Schema {name} is not a list item")
        item_name = block_schema.get_item_name()
        return self.instantiate(item_name, item_name, style=style, role=role, tags=tags, force_schema=force_schema)
    
    def instantiate(
        self, 
        name: str, 
        content: ContentType | None = None, 
        attrs: dict | None = None, 
        style: str | None | UnsetType = UNSET, 
        role: str | None | UnsetType = UNSET, 
        tags: list[str] | None | UnsetType = UNSET,
        force_schema: bool = False,
    ):
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        transformer = self._get_schema(name)
        # if isinstance(transformer.block_schema.parent, BlockListSchema):
        if transformer.is_list_item:
            # check if list is already instantiated
            if not isinstance(self._top_or_none(), BlockList):
                list_transformer = self._get_schema(transformer.block_schema.parent.name)
                list_transformer.instantiate(content=None)
                self._push(list_transformer)
        else:
            if isinstance(self._top_or_none(), BlockList):
                self._pop()
                        
        transformer.instantiate(content=content, style=style, role=role, tags=tags, force_schema=force_schema)
        # if content is not None:
        #     transformer.append(content)
        self._push(transformer)
        return transformer.block
    
    
    
    
        
    
    def commit(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None, force_schema: bool = False):
        if len(self._stack) == 0:
            raise RuntimeError("No block to commit")
        
        transformer = self._stack.pop()
        transformer.commit(content=content, style=style, role=role, tags=tags, force_schema=force_schema)
        return transformer.block
    
    def append(self, chunk: "BlockChunk", force_schema: bool = False):
        if len(self._stack) == 0:
            raise RuntimeError("No block to append to")
        self._top().append(chunk, force_schema=force_schema)
