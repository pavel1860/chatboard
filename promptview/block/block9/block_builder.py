from .block import Block, BlockSchema, BlockSent, BlockListSchema, BlockChunk
from .base_blocks import BaseContent
from typing import Any



def traverse_dict(target, path: list[int]=[], label_path: list[str] = []):
    for i, (k, v) in enumerate(target.items()):  
        if type(v) is dict:
            yield k, None, [*path, i], [*label_path, k]
            yield from traverse_dict(v, [*path, i], [*label_path, k])
        elif type(v) is list:
            for item in v:
                yield k, item, [*path, i], [*label_path, k]
        else:
            yield k, v, [*path, i], [*label_path, k]



class BlockBuildContext:
    
    def __init__(self, schema: BlockSchema):
        self.schema = schema
        self.block = None
        self.schema_stack = []
        self._did_initialize = False
        self._did_finish = False
        
        
    @property
    def is_list(self) -> bool:
        return type(self.schema) is BlockListSchema
    
    @property
    def did_finish(self) -> bool:
        return self._did_finish
        
    def instantiate(
        self, 
        value: str | list[BlockChunk] | None = None, 
        content: str | None = None,
        attrs: dict[str, str] | None = None, 
        ignore_style: bool = False,
        ignore_tags: bool = False,
    ):
        if not self._did_initialize:
            self.block = self.schema.instantiate(value=value, content=content, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
            self._did_initialize = True
        return self.block
    
    
    def instantiate_item(
        self, 
        value: str | list[BlockChunk] | None = None, 
        content: str | None = None,
        attrs: dict[str, str] | None = None, 
        ignore_style: bool = False,
        ignore_tags: bool = False,
    ):
        if not self._did_initialize:
            raise ValueError("Block not initialized")
        if self.block is None:
            raise ValueError("Block is not initialized")
        if type(self.schema) is not BlockListSchema:
            raise ValueError("Schema is not a list")
        self.block = self.schema.instantiate_item(value=value, content=content, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
        return self.block
    
    
    def append(self, value: Block | BaseContent):
        if not self._did_initialize:
            raise ValueError("Block not initialized")
        if self.block is None:
            raise ValueError("Block is not initialized")
        self.block.append(value)
        return self.block
    
    def inline_append(self, value: BlockChunk | BaseContent):
        if not self._did_initialize:
            raise ValueError("Block not initialized")
        if self.block is None:
            raise ValueError("Block is not initialized")
        if len(self.block) == 0:
            self.block.append(Block())        
        self.block.inline_append(value)
        return self.block
    
    
    def get_schema(self, view_name: str) -> BlockSchema:
        schema = self.schema.get_one(view_name)
        if schema is None:
            raise ValueError(f"Schema {view_name} not found")
        if not isinstance(schema, BlockSchema):
            raise ValueError(f"Schema {view_name} is not a schema")
        return schema
    
    
    def commit(self, postfix: str | None = None):
        if not self._did_initialize:
            raise ValueError("Block not initialized")
        if self.block is None:
            raise ValueError("Block is not initialized")
        if postfix is not None:
            self.block.postfix = BlockSent(postfix)
        self._did_finish = True
        return self.block

    
    
    

class SchemaBuildContext:
    schema: BlockSchema
    inst: BlockBuildContext | None
    stack: list[BlockBuildContext]
    _did_finish: bool
    
    
    def __init__(self, schema: BlockSchema | Block, role="assistant", tags: list[str] | None = None):
        self.schema  = schema.copy_kind(BlockSchema)
        # self.inst = Block(role=role, tags=tags or [])
        self.inst = None
        self.stack = []
        self._did_finish = False
        
    @property
    def result(self):
        return self.inst.block
        
    def _reset_stack(self):
        self.stack = []
        
        
    # def _build_view_inst(self, path, value):
    #     view_schema = self.schema.get_one(path)
    #     block = view_schema.instantiate(value)
    #     return block
    
    def curr_path(self):
        return [b.tags[0] for b in self.stack]
    
    def _push(self, bld_ctx: BlockBuildContext):
        if self.stack:
            self.stack[-1].append(bld_ctx.block)
        else:
            self.inst = bld_ctx
        self.stack.append(bld_ctx)
        return bld_ctx
        
    def _pop(self):
        # while not self.schema_stack[-1] != self.stack[-1].schema:
            # self.schema_stack.pop()
        return self.stack.pop()
    
    def _top(self) -> BlockBuildContext | None:
        if not self.stack:
            return None
        return self.stack[-1]
    
    def _get_schema_build_ctx(self, view_name: str, attrs: dict[str, str] | None = None) -> BlockBuildContext:
        from .block import BlockListSchema
        if not self.stack:
            schema = self.schema.get_one(view_name)
            if schema is None:
                raise ValueError(f"Schema {view_name} not found")
            if not isinstance(schema, BlockSchema):
                raise ValueError(f"Schema {view_name} is not a schema")
            return BlockBuildContext(schema)
            
        schema = self.stack[-1].get_schema(view_name)
        return BlockBuildContext(schema)
    
    
    def _get_list_schema_build_ctx(self, attrs: dict[str, str]) -> BlockBuildContext:
        if not self.stack:
            raise ValueError("Stack is empty")
        bld_ctx = self.stack[-1]
        if not type(bld_ctx.schema) is BlockListSchema or bld_ctx.schema.key is None:
            raise ValueError("Schema is not a list or key is not set")
        schema = bld_ctx.get_schema(attrs[bld_ctx.schema.key])
        return BlockBuildContext(schema)
    
    # def inst_view2(self, label_path, value) -> list[Block]:
    #     curr_path = []
    #     instances = []
    #     for i, label in enumerate(label_path):
    #         is_last = i == len(label_path) - 1
    #         curr_path.append(label)
    #         # if i+1 < len(label_path) and label in self.stack[i+1].tags:
    #             # continue
    #         target_inst = self.inst.get_one_or_none(curr_path) if self.inst else None
    #         if not target_inst:
    #             view_schema = self.schema.get_one(curr_path)
    #             target_inst = view_schema.instantiate(value if is_last else None)    
    #             self.stack[-1].append(target_inst)
    #             instances.append(target_inst)                
    #             self.stack.append(target_inst)
    #     return instances
    
    # def inst_view(self, view_name: str, value, attrs: dict[str, str] | None = None) -> list[Block]:
    #     from .block import BlockListSchema
    #     # view_schema = self.schema.get_one(label_path)
    #     view_schema = self._get_schema(view_name, attrs)
    #     block = view_schema.instantiate(content=value, attrs=attrs, ignore_style=True)
    #     self._push(view_schema, block)
    #     if isinstance(view_schema, BlockListSchema):
    #         if not attrs:
    #             raise ValueError("Attribute 'name' is required for list item")
    #         item_schema = view_schema.get(attrs["name"])
    #         if item_schema is None:
    #             raise ValueError(f"List view '{attrs["name"]}' not found")
    #         self.schema_stack.append(item_schema)
    #     return [block]
    
    def inst_view(self, view_name: str, value, attrs: dict[str, str] | None = None) -> list[Block]:
        bld_ctx = self._get_schema_build_ctx(view_name, attrs) 
        if bld_ctx.is_list:
            if not bld_ctx._did_initialize:
                block = bld_ctx.instantiate(None)
                self._push(bld_ctx)  
            
            if not attrs:
                raise ValueError("Attribute 'name' is required for list item")
            item_bld_ctx = self._get_list_schema_build_ctx(attrs)
            block = item_bld_ctx.instantiate(content=value, attrs=attrs, ignore_style=True)
            self._push(item_bld_ctx)        
        else:
            block = bld_ctx.instantiate(content=value, attrs=attrs, ignore_style=True)
            self._push(bld_ctx)
        if block is None:
            raise ValueError("Block is not initialized")
        return [block]
            
    def append(self, value: BlockChunk | BaseContent):
        if not self.stack:
            raise ValueError("Stack is empty")
        return self.stack[-1].inline_append(value)
            
    def commit_view(self, value: str | list[BlockChunk] | None = None):
        
        bld_ctx = self._pop()
        if value is not None:
            bld_ctx.block.postfix = BlockSent(value)        
        bld_ctx._did_finish = True
        # view = view.strip()
        return bld_ctx.block
    
    def inst_dict(self, payload: dict[str, Any]):
        from .block import BlockListSchema
        from pydantic import BaseModel
        from ...utils.string_utils import camel_to_snake
        for key, value, path, label_path in traverse_dict(payload):
                       
            # if self._top() and self._top().is_list:
            #     if isinstance(value, BaseModel):
            #         model_name = camel_to_snake(value.__class__.__name__)
            #         attr = {self._top().schema.key: model_name}                    
            #         self.inst_view(key, key, attrs=attr)
            # else:
            #     self.inst_view(key, key)
            attr = None 
            if isinstance(value, BaseModel):
                model_name = camel_to_snake(value.__class__.__name__)
                attr = {self._top().schema.key: model_name}                    
            self.inst_view(key, key, attrs=attr)
            if value is not None:
                self.stack[-1].append(value)
            self.commit_view()
        return self.inst
    
    def inst_dict3(self, payload):
        from .block import BlockListSchema
        for key, value, path, label_path in traverse_dict(payload):
            view_schema = self.schema.get_one(label_path)
            if isinstance(view_schema, BlockListSchema):
                if self.inst is None:
                    raise ValueError("inst is not set")
                view_list = self.inst.get_one_or_none(label_path)
                if view_list is None:
                    view_list = view_schema.instantiate(is_wrapper=True)
                    parent = self.inst.get_one(label_path[:-1])
                    parent.append(view_list)
                block = view_schema.instantiate_item(value)
                view_list.append(block)
                # block = view_schema.instantiate()
            else:
                block = view_schema.instantiate(value=value)                
                if self.inst is None:
                    self.inst = block
                else:
                    parent = self.inst.get_one(label_path[:-1])
                    parent.append(block)
        return self.inst
        
        
    def inst_dict2(self, payload):
        for key, value, path, label_path in traverse_dict(payload):
            curr_path = []
            for i, label in enumerate(label_path):
                is_last = i == len(label_path) - 1
                curr_path.append(label)
                target_inst = self.inst.get_one_or_none(curr_path)
                if not target_inst:
                    view_schema = self.schema.get_one(curr_path)
                    target_inst = view_schema.instantiate(value if is_last else None)    
                    self.stack[-1].append(target_inst)           
                self.stack.append(target_inst)
            self._reset_stack()
        return self.inst
