from queue import SimpleQueue

from .block import Block, BlockChunk, BlockSchema, AttrBlock, BlockSent
from .base_blocks import BaseBlock
from typing import Type



class BlockBuilderError(Exception):
    pass



class BlockBuilderContext:
    
    
    def __init__(self, schema: Block):
        self.schema = self.extract_schema(schema)
        self.instance = None
        self.queue = SimpleQueue()
        
        
    def extract_schema(self, schema: Block) -> BlockSchema:
        def _clone_target_node(n: BlockSchema) -> BlockSchema:            
            r = n.copy({"children":[], "parent":None})                    
            return r
        
        def _is_target(node: BaseBlock) -> bool:
            return isinstance(node, BlockSchema)
        
        
        def _connect_target_node(u: BlockSchema, parent: BlockSchema):
            if parent.is_list:
                u.is_list_item = True
            parent.append(u)
        
        res = schema.gather_trees(_is_target, _clone_target_node, _connect_target_node)
        if len(res) == 1:
            return res[0]
        else:
            raise BlockBuilderError("Multiple target nodes found")  
        

    
    def has_events(self) -> bool:
        return not self.queue.empty()
    
    def get_event(self) -> Block:
        if self.queue.empty():
            raise BlockBuilderError("No events to get")
        return self.queue.get()
    
    def _push_event(self, event: Block):
        # self.queue.put(event.model_dump())
        self.queue.put(event)
        
        
    def get_view_info(self, view_name: str, is_last: bool = False) -> tuple[BlockSchema, Block | None]:
        schema = self.schema.get(view_name)
        if schema is None:
            raise BlockBuilderError(f"View {view_name} not found")
        if not isinstance(schema, BlockSchema):
            raise BlockBuilderError(f"View {view_name} is not a schema")
        view = None
        if self.instance is not None:
            if is_last:
                view = self.instance.get_last(view_name)
            else:
                view = self.instance.get(view_name)
        return schema, view
    
    def get_block_path(self, target_schema: BlockSchema | str) -> list[BlockSchema]:        
        if isinstance(target_schema, str):
            view_schema = self.schema.get(target_schema)
            if view_schema is None:
                raise BlockBuilderError(f"View {view_schema} not found")
            if not isinstance(view_schema, BlockSchema):
                raise BlockBuilderError(f"View {view_schema} is not a schema")
        else:
            view_schema = target_schema
        if view_schema is None:
            raise BlockBuilderError(f"View {view_schema} not found")
        path_views = [view_schema]
        while (parent := path_views[0].parent) is not None:
            path_views.insert(0, parent)
        return path_views
    
    
    def _build_response_block(
        self, 
        schema: BlockSchema, 
        content: list[BlockChunk] | BlockChunk | str | None = None, 
        tags: list[str] | None = None, 
        attrs: dict[str, str] | None = None
    ) -> Block:
        view = Block(
            content=content,
            tags=[t for t in schema.tags ] + (tags or []),
            # styles=["stream"]
            styles=["stream-view"]
        )
        if attrs:
            for k, v in attrs.items():
                if k in schema.attrs:
                    value = schema.attrs[k].parse(v)
                    view.attrs[k] = value
                else:
                    raise BlockBuilderError(f'Attribute "{k}" not found in schema')
                
        return view
    

    
    def inst_view(
        self, 
        target_schema: BlockSchema, 
        content: list[BlockChunk] | BlockChunk | str | None = None, 
        tags: list[str] | None = None, 
        attrs: dict[str, str] | None = None
    ) -> Block:
        
        def build_response_block(schema: BlockSchema) -> Block:
            if target_schema.name == schema.name:
                view = Block(
                    content=content,
                    tags=[t for t in schema.tags ] + (tags or []),
                    # styles=["stream"]
                    styles=["stream-view"]
                )
                if attrs:
                    view = self.set_attributes(schema, view, attrs)
                    # for k, v in attrs.items():
                    #     if k in schema.attrs:
                    #         value = schema.attrs[k].parse(v)
                    #         view.attrs[k] = value
                    #     else:
                    #         raise BlockBuilderError(f"Attribute {k} not found in schema")
                    
            else:
                view = Block(
                    content=schema.content.copy(),
                    tags=[t for t in schema.tags],
                    styles=["stream-view"]
                )
            return view
        
        def insert(path: list[int], view: Block):
            path = path[1:]
            if not path:
            # if path == [0]:
                view.role = "assistant"
                self.instance = view
                return view
            else:
                if self.instance is None:
                    raise BlockBuilderError("Instance is not initialized")
                return self.instance.insert(view, path)

        curr_view = self.instance

        view_path = self.get_block_path(target_schema)
        for i, vw_scm in enumerate(view_path):
            pth_view = curr_view.get(vw_scm.name) if curr_view is not None else None
            if pth_view is None:
                pth_view = build_response_block(vw_scm)                
                insert(vw_scm.path, pth_view)
                self._push_event(pth_view)
            curr_view = pth_view            
        return curr_view
    
    
    def set_attributes(self, schema: BlockSchema, view: Block, attrs: dict[str, str]):
        for k, v in attrs.items():
            if k in schema.attrs:
                value = schema.attrs[k].parse(v)
                view.attrs[k] = value
            else:
                raise BlockBuilderError(f"Attribute {k} not found in schema")
        return view
    
    # def set_attributes(self, view_name: str, view: Block, attrs: dict[str, str]):
    #     schema = self.schema.get(view_name)
    #     if not schema:
    #         raise BlockBuilderError(f"Field {view_name} not found")
    #     if not isinstance(schema, BlockSchema):
    #         raise BlockBuilderError(f"Field {view_name} is not a schema")
    #     for k, v in attrs.items():
    #         if k in schema.attrs:
    #             value = schema.attrs[k].parse(v)
    #             view.attrs[k] = value
    #         else:
    #             raise BlockBuilderError(f"Attribute {k} not found in schema")
    #     return view
    
    
    # def append(
    #     self, 
    #     view_name: str, 
    #     content: list[BlockChunk] | BlockChunk | str | None = None,
    #     tags: list[str] | None = None,
    #     attrs: dict[str, str] | None = None,
    # ) -> Block:
    #     schema, view = self.get_view_info(view_name)
    #     if view is not None:
    #         if view.is_last_eol():
    #             blk = view.append(content)
    #         else:
    #             blk = view.inline_append(content)
            
    #         if attrs:
    #             self.set_attributes(view_name, blk, attrs)
    #         self._push_event(blk)
    #         return blk                
    #     else:             
    #         return self.inst_view(schema, content, tags, attrs)
    
    def append(
        self, 
        view_name: str, 
        content: list[BlockChunk] | BlockChunk | str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, str] | None = None,
    ) -> Block:
        schema, view = self.get_view_info(view_name, is_last=True)
        if view is None:
            raise BlockBuilderError(f"View {view_name} is not instantiated")
        if view.is_last_eol():
            blk = view.append(content)                
            blk.styles += ["stream"]
        else:
            blk = view.inline_append(content)
        if attrs:
            self.set_attributes(view_name, blk, attrs)
        self._push_event(blk)
        return blk  
        
            
    
    def instantiate(self, view_name: str, content: list[BlockChunk] | BlockChunk | str | None = None, tags: list[str] | None = None, attrs: dict[str, str] | None = None) -> tuple[Block, BlockSchema]:
        schema, view = self.get_view_info(view_name)
        if view is not None:
            if schema.is_list_item:
                raise BlockBuilderError(f"previous view '{view_name}' is not closed")
            else:
                raise BlockBuilderError(f"View '{view_name}' is already instantiated")
        if schema.is_list:
            if view is None:
                view = self.inst_view(schema)
                if attrs is None or "name" not in attrs:
                    raise BlockBuilderError(f"Attribute 'name' is required for list item")
                item_name = attrs["name"]
            return self.inst_list_item(view, schema, item_name, content, tags, attrs)
        else:
            return self.inst_view(schema, content, tags, attrs), schema
    
    # def append_list_item(
    #     self, 
    #     list_name: str, 
    #     view_name: str, 
    #     content: list[BlockChunk] | BlockChunk | str | None = None, 
    #     tags: list[str] | None = None, 
    #     attrs: dict[str, str] | None = None
    # ) -> tuple[Block, BlockSchema]:
    #     list_schema, list_view = self.get_view_info(list_name)
    #     if list_view is None:
    #         raise ValueError(f"List {list_name} is not instantiated")
    #     last_view = list_view.last_child
    #     if last_view is None:
    #         raise ValueError(f"List {list_name} has no children")
    #     if last_view.is_last_eol():
    #         blk = last_view.append(content)
    #         blk.styles += ["stream"]
    #     else:
    #         blk = last_view.inline_append(content)
    #     if attrs:
    #         self.set_attributes(view_name, blk, attrs)
    #     self._push_event(blk)
    #     return blk, schema
    
    
    def inst_list_item(
        self, 
        list_view: Block,
        list_schema: BlockSchema,
        view_name: str, 
        content: list[BlockChunk] | BlockChunk | str | None = None, 
        tags: list[str] | None = None, 
        attrs: dict[str, str] | None = None
    ) -> tuple[Block, BlockSchema]:
        schema, _ = self.get_view_info(view_name, is_last=True)
        item_view = self._build_response_block(schema, content, tags, attrs)
        list_view.append(item_view)
        item_view = self.set_attributes(schema, item_view, attrs)
        self._push_event(item_view)
        return item_view, schema
            
            
            
        
        
    
    def instantiate_list_item(
        self, 
        list_name: str, 
        view_name: str, 
        content: list[BlockChunk] | BlockChunk | str | None = None, 
        tags: list[str] | None = None, 
        attrs: dict[str, str] | None = None
    ) -> tuple[Block, BlockSchema]:
        list_schema, list_view = self.get_view_info(list_name)
        if list_view is None:
            raise BlockBuilderError(f"List {list_name} is not instantiated")
        schema, _ = self.get_view_info(view_name, is_last=True)
        item_view = self._build_response_block(schema, content, tags, attrs)
        list_view.append(item_view)
        item_view = self.set_attributes(schema, item_view, attrs)
        self._push_event(item_view)
        return item_view, schema
    
    
    
    def print_schema_tree(self, verbose: bool = False):
        if self.schema is None:
            raise BlockBuilderError("Schema is not initialized")
        return self.schema.print_tree(verbose=verbose)
    
    def print_tree(self, verbose: bool = False):
        if self.instance is None:
            return "None"
        return self.instance.print_tree(verbose=verbose)
            
    
    
    def set_view_attr(
        self,
        view_name: str,
        postfix: list[BlockChunk] | None = None,
    ):
        schema, view = self.get_view_info(view_name, is_last=True)
        if postfix is not None:            
            if view is None:
                raise BlockBuilderError(f"View {view_name} not found")
            view.postfix.extend(postfix)
        
        return view
    
    
    
    def commit_view(
        self,
        view_name: str,
    ):
        schema, view = self.get_view_info(view_name, is_last=True)
        if view is None:
            raise BlockBuilderError(f"View {view_name} not found")
        # view.commit()
        return view