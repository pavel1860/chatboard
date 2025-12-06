from .block import Block, BlockSchema, BlockSent



def traverse_dict(target, path: list[int]=[], label_path: list[str] = []):
    for i, (k, v) in enumerate(target.items()):  
        if type(v) is dict:
            yield from traverse_dict(v, [*path, i], [*label_path, k])
        else:
            yield k, v, [*path, i], [*label_path, k]



class SchemaBuildContext:
    
    
    def __init__(self, schema, role="assistant", tags: list[str] | None = None):
        self.schema  = schema.copy_kind(BlockSchema)
        # self.inst = Block(role=role, tags=tags or [])
        self.inst = None
        self.stack = []
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
    
    def _push(self, block: Block):
        if self.stack:
            self.stack[-1].append(block)
        else:
            self.inst = block
        self.stack.append(block)
        return block
        
    def _pop(self):
        return self.stack.pop()
    
    def inst_view2(self, label_path, value) -> list[Block]:
        curr_path = []
        instances = []
        for i, label in enumerate(label_path):
            is_last = i == len(label_path) - 1
            curr_path.append(label)
            # if i+1 < len(label_path) and label in self.stack[i+1].tags:
                # continue
            target_inst = self.inst.get_one_or_none(curr_path) if self.inst else None
            if not target_inst:
                view_schema = self.schema.get_one(curr_path)
                target_inst = view_schema.instantiate(value if is_last else None)    
                self.stack[-1].append(target_inst)
                instances.append(target_inst)                
                self.stack.append(target_inst)
        return instances
    
    def inst_view(self, label_path, value) -> list[Block]:
        view_schema = self.schema.get_one(label_path)
        block = view_schema.instantiate(value)
        self._push(block)
        return [block]
            
    def append(self, value):
        if len(self.stack[-1]) == 0:
            self.stack[-1].append(Block())
        return self.stack[-1].inline_append(value)
            
    def commit_view(self, value = None):
        
        view = self.stack.pop()
        if value is not None:
            view.postfix = BlockSent(value)        
        # view = view.strip()
        return view
        
        
    def inst_dict(self, payload):
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
