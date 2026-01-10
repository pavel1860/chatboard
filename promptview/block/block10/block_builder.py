from .block import BlockSchema, BlockListSchema, BlockList, BlockBase, ContentType, BlockChunk, Block
from ...utils.type_utils import UNSET, UnsetType
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .block_transformers import BlockSchemaTransformer



class BlockBuilderError(Exception):
    pass



class BlockBuilderContext:
    
    def __init__(self, schema: "BlockSchema | None"):        
        self.schema = schema.extract_schema() if schema is not None else None  
        self._stack = []
        self._root = None
        self._block_text = None
        if self.schema.is_wrapper:
            self.init_root()      
        
        
        
        
    def init_root(self):
        from .block_transformers import BlockSchemaTransformer
        # self._root = self._get_schema(self.schema.name)
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        self._root = BlockSchemaTransformer.from_block_schema(self.schema)
        content = self._root.block_schema.name if not self._root.block_schema.is_wrapper else None
        self._root.instantiate(content, tags=["root"], force_schema=True)       
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
    
    def _push(self, transformer: "BlockSchemaTransformer"):
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
    
    
    def _is_top_committed(self):
        top = self._top_or_none()
        if top is None:
            return False
        return top._did_commit

    
    # def _get_schema(self, name: str):
    #     from .block_transformers import BlockSchemaTransformer
    #     if self.schema is None:
    #         raise RuntimeError("Schema not initialized")
    #     block_schema = self.schema.get_one(name)
    #     if block_schema is None:
    #         raise RuntimeError(f"Schema {name} not found")
    #     block_transformer = BlockSchemaTransformer.from_block_schema(block_schema)
    #     return block_transformer
    def _get_schema(self, name: str):
        from .block_transformers import BlockSchemaTransformer
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        # block_schema = self._top().get_one_or_none(name)
        block_schema = None
        if top := self._top_or_none():
            block_schema = top.block_schema.get_one_or_none(name)
        elif name in self.schema.tags:
            block_schema = self.schema
        if block_schema is None:
            raise RuntimeError(f"Schema '{name}' not found")
        block_transformer = BlockSchemaTransformer.from_block_schema(block_schema)
        return block_transformer
    
    def inst_list(self, item_name: str):
        from .block_transformers import BlockSchemaTransformer
        list_schemas = self._top().block_schema.get_all_lists()
        if not list_schemas:
            raise BlockBuilderError(f"Did not find any list schemas for '{name}'")
        target_list_schema = None
        for list_schema in list_schemas:
            if list_schema.get_one_or_none(item_name) is not None:
                target_list_schema = list_schema
                break
        if target_list_schema is None:
            raise BlockBuilderError(f"Did not find list schema for '{item_name}'")
        list_transformer = BlockSchemaTransformer.from_block_schema(target_list_schema)
        list_transformer.instantiate(content=None)
        self._push(list_transformer)
        return list_transformer
    
    
    # def inst_and_commit(self, name: str, content: ContentType | None = None, style: str | None | UnsetType = UNSET, role: str | None | UnsetType = UNSET, tags: list[str] | None | UnsetType = UNSET, force_schema: bool = False, inst_key: bool = False):
        

    def instantiate_list_item(
        self, 
        name: str,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
        force_schema: bool = False,
        inst_key: bool = False,
        attrs: dict | None | UnsetType = UNSET,
    ):
        # block_schema = self.schema.get_one(name)
        if self._is_top_committed():
            self._pop()
        item_name = name   
        if not self._top().is_list:
            list_transformer = self.inst_list(name)
        else:
            list_transformer = self._top()
            
        if list_item_name := list_transformer.block_schema.item_name: 
            item_name = list_item_name
        list_item_transformer = self.instantiate(name, item_name, attrs=attrs, style=style, role=role, tags=tags, force_schema=force_schema)
        # if inst_key:
        #     if item_key := list_transformer.block_schema.key:
        #         self.instantiate(item_key, item_key, force_schema=True)
        #         self.curr_block.append_child(name)
        #         self.commit(item_key, force_schema=force_schema)
        
        return list_item_transformer
        # block_schema = self._top().get_one_or_none(name)
        # if not block_schema.is_list_item:
            # raise RuntimeError(f"Schema {name} is not a list item")
        # item_name = block_schema.get_item_name()
        # return self.instantiate(item_name, item_name, style=style, role=role, tags=tags, force_schema=force_schema)
    
    def instantiate(
        self, 
        name: str, 
        content: ContentType | None = None, 
        attrs: dict | None | UnsetType = UNSET, 
        style: str | None | UnsetType = UNSET, 
        role: str | None | UnsetType = UNSET, 
        tags: list[str] | None | UnsetType = UNSET,
        force_schema: bool = False,
    ):
        if self._is_top_committed():
            self._pop()
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        if self._top_or_none() and self._top().is_list:
            if name not in self._top().block_schema.tags:
                raise BlockBuilderError(f"Schema '{name}' not found in list '{self._top().block_schema.name}'")
            if attrs is UNSET or not attrs:
                raise BlockBuilderError("Attributes are required for list item")
            if item_name := attrs.get("name"):
                name = item_name
            else:
                raise BlockBuilderError("Attribute 'name' is required for list item")
        transformer = self._get_schema(name)
        # if isinstance(transformer.block_schema.parent, BlockListSchema):
        if transformer.is_list_item:
            # check if list is already instantiated
            # if not isinstance(self._top_or_none(), BlockList):
            #     list_transformer = self._get_schema(transformer.block_schema.parent.name)
            #     list_transformer.instantiate(content=None)
            #     self._push(list_transformer)
            if not self._top().is_list:
                list_transformer = self.inst_list(name)
                # if list_item_name := list_transformer.block_schema.item_name: 
                    # content = list_item_name
        elif transformer.is_list:
            transformer.instantiate(force_schema=force_schema)
            self._push(transformer)
            if attrs is UNSET or not attrs:
                raise BlockBuilderError("Attributes are required for list item")
            if item_name := attrs.get(transformer.block_schema.key):
                transformer = self._get_schema(item_name)
        else:
            if isinstance(self._top_or_none(), BlockList):
                self._pop()
                        
        transformer.instantiate(content=content, attrs=attrs, style=style, role=role, tags=tags, force_schema=force_schema)
        # if content is not None:
        #     transformer.append(content)
        self._push(transformer)
        return transformer.block
    
    
        
    
    def commit(self, content: ContentType | None = None, style: str | None = None, role: str | None = None, tags: list[str] | None = None, force_schema: bool = False):
        if len(self._stack) == 0:
            raise RuntimeError("No block to commit")
        if self._is_top_committed():
            self._pop()
        # transformer = self._stack.pop()
        transformer = self._top()
        transformer.commit(content=content, style=style, role=role, tags=tags, force_schema=force_schema)
        return transformer.block
    
    def append(self, chunk: "BlockChunk", force_schema: bool = False, as_child: bool = False, start_offset: int | None = None, end_offset: int | None = None):
        if len(self._stack) == 0:
            raise RuntimeError("No block to append to")
        self._top().append(chunk, force_schema=force_schema, as_child=as_child, start_offset=start_offset, end_offset=end_offset)
