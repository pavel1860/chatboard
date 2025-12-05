from .block import Block, BlockSchema



def traverse_dict(target, path: list[int]=[], label_path: list[str] = []):
    for i, (k, v) in enumerate(target.items()):  
        if type(v) is dict:
            yield from traverse_dict(v, [*path, i], [*label_path, k])
        else:
            yield k, v, [*path, i], [*label_path, k]



class SchemaBuildContext:
    
    
    def __init__(self, schema, role="assistant", tags: list[str] | None = None):
        self.schema  = schema.copy_kind(BlockSchema)
        self.inst = Block(role=role, tags=tags or [])
        self.stack = [self.inst]
        self._did_finish = False
        
    @property
    def result(self):
        return self.inst
        
    def _reset_stack(self):
        self.stack = [self.inst]
        
        
    def _build_view_inst(self, path, value):
        view_schema = self.schema.get_one(path)
        block = view_schema.instantiate(value)
        return block
    
    def inst_view(self, label_path, value) -> list[Block]:
        curr_path = []
        instances = []
        for i, label in enumerate(label_path):
            is_last = i == len(label_path) - 1
            curr_path.append(label)
            target_inst = self.inst.get_one_or_none(curr_path)
            if not target_inst:
                view_schema = self.schema.get_one(curr_path)
                target_inst = view_schema.instantiate(value if is_last else None)    
                self.stack[-1].append(target_inst)
                instances.append(target_inst)
                # print("build block for", curr_path)               
            self.stack.append(target_inst)
        return instances
            
    def append(self, value):
        if len(self.stack[-1]) == 0:
            self.stack[-1].append(Block())
        return self.stack[-1].inline_append(value)
            
    def commit_view(self, value = None):
        view = self.stack.pop()
        if value is not None:
            view.postfix = value
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
