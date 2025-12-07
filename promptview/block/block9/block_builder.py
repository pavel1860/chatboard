from .block import Block, BlockSchema, BlockSent



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



class SchemaBuildContext:
    
    
    def __init__(self, schema, role="assistant", tags: list[str] | None = None):
        self.schema  = schema.copy_kind(BlockSchema)
        # self.inst = Block(role=role, tags=tags or [])
        self.inst = None
        self.stack = []
        self.schema_stack = []
        self._did_finish = False
        
    @property
    def result(self):
        return self.inst
        
    def _reset_stack(self):
        self.stack = []
        
        
    def _build_view_inst(self, path, value):
        view_schema = self.schema.get_one(path)
        block = view_schema.instantiate(value)
        return block
    
    def curr_path(self):
        return [b.tags[0] for b in self.stack]
    
    def _push(self, schema: BlockSchema, block: Block):
        if self.stack:
            self.stack[-1].append(block)
        else:
            self.inst = block
        self.stack.append(block)
        self.schema_stack.append(schema)
        return block
        
    def _pop(self):
        # while not self.schema_stack[-1] != self.stack[-1].schema:
            # self.schema_stack.pop()
        return self.stack.pop(), self.schema_stack.pop()
    
    
    def _get_schema(self, view_name: str, attrs: dict[str, str] | None = None) -> BlockSchema:
        from .block import BlockListSchema
        if not self.schema_stack:
            return self.schema.get_one(view_name)
        # if isinstance(self.schema_stack[-1], BlockListSchema):
        #     if not attrs:
        #         raise ValueError("Attribute 'name' is required for list item")
        #     list_tag = attrs["name"]
        #     list_view = self.schema_stack[-1].get(list_tag)
        #     if list_view is None:
        #         raise ValueError(f"List view {list_tag} not found")
        #     # if not isinstance(list_view, Block):
        #     #     raise ValueError(f"List view {list_tag} is not a block")
        #     return list_view.get_one(view_name)
            
        return self.schema_stack[-1].get(view_name)
    
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
    
    def inst_view(self, view_name: str, value, attrs: dict[str, str] | None = None) -> list[Block]:
        from .block import BlockListSchema
        # view_schema = self.schema.get_one(label_path)
        view_schema = self._get_schema(view_name, attrs)
        block = view_schema.instantiate(content=value, attrs=attrs, ignore_style=True)
        self._push(view_schema, block)
        if isinstance(view_schema, BlockListSchema):
            if not attrs:
                raise ValueError("Attribute 'name' is required for list item")
            item_schema = view_schema.get(attrs["name"])
            if item_schema is None:
                raise ValueError(f"List view '{attrs["name"]}' not found")
            self.schema_stack.append(item_schema)
        return [block]
            
    def append(self, value):
        if len(self.stack[-1]) == 0:
            self.stack[-1].append(Block())
        return self.stack[-1].inline_append(value)
            
    def commit_view(self, value = None):
        
        view, schema = self._pop()
        if value is not None:
            view.postfix = BlockSent(value)        
        # view = view.strip()
        return view
    
    
    def inst_dict(self, payload):
        from .block import BlockListSchema
        for key, value, path, label_path in traverse_dict(payload):            
            view_schema = self.schema.get_one(label_path)
            if isinstance(view_schema, BlockListSchema):
                if self.inst is None:
                    raise ValueError("inst is not set")                 
                view_list = self.inst.get_one_or_none(label_path)
                if view_list is None:
                    view_list = view_schema.instantiate()
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
                    # print("build block for", curr_path)               
                self.stack.append(target_inst)
            self._reset_stack()
        return self.inst
