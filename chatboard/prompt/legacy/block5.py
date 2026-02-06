from collections import defaultdict
from abc import abstractmethod
from typing import Any, List, Type, Dict, Optional
from chatboard.block.style import InlineStyle, style_manager






class TagRegistry:
    
    tags: defaultdict[str, list["BaseBlock"]]
    
    def __init__(self):
        self.tags = defaultdict(list)

    def add(self, block: "BaseBlock"):
        for tag in block.tags:
            self.tags[tag].append(block)        
    
    def get(self, tag: str | list[str]) -> list["BaseBlock"]:
        if isinstance(tag, str):
            return self.tags[tag]
        else:
            return [block for t in tag for block in self.tags[t]]






# class MetaBlock(type):
    
#     def __new__(cls, name, bases, attrs):
#         attrs["__call__"] = BlockContext(cls)
#         return super().__new__(cls, name, bases, attrs)

     
        
        
class BaseBlock:
    
    tags: list[str]
    items: list["BaseBlock"]
    
    
    def __init__(self, tags: list[str] | None = None, depth: int = 0):
        self.tags = tags or []
        self.items = []
        self.depth = depth or 0
        self.inline_style: InlineStyle = {}
        self.computed_style: InlineStyle = {}
        self.parent: Optional["BaseBlock"] = None
        
        
    def append(self, item: "BaseBlock | Any"):
        # if not isinstance(item, Block):
            # item = self._build_instance(item)
        self.items.append(item)
        
        # Set parent reference for nested selector support
        if hasattr(item, 'parent'):
            item.parent = self
            
        return item
        
    def __add__(self, other: "BaseBlock | Any"):
        item = self.append(other)
        return item
    
    def __iadd__(self, other: "BaseBlock | Any"):
        self.append(other)
        return self
       
    # def _build_instance(self, *args, **kwargs):
    #     print(self.__class__)     
    #     return self.__class__(*args, **kwargs)
    
    @classmethod
    def _build_instance(cls, *args, **kwargs): 
        return cls(*args, **kwargs)
    
    def add_style(self, **style_props: Any) -> "BaseBlock":
        """
        Add inline style properties to this block
        """
        self.inline_style.update(style_props)
        return self
    
    def compute_styles(self) -> InlineStyle:
        """
        Compute the final styles for this block by applying the style manager rules
        """
        self.computed_style = style_manager.apply_styles(self)
        
        # Recursively compute styles for all child blocks
        for item in self.items:
            if hasattr(item, 'compute_styles'):
                item.compute_styles()
                
        return self.computed_style
    
    def get_style(self, property_name: str, default: Any = None) -> Any:
        """
        Get a computed style property value
        """
        if not self.computed_style:
            self.compute_styles()
        return self.computed_style.get(property_name, default)
        
    @abstractmethod
    def render(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def parse(self, text: str):
        raise NotImplementedError("Subclasses must implement this method")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        


    
      
    
    
    
class BlockContext(object):
    
    _block_type_registry: dict[Type, Type[BaseBlock]]
    _ctx_stack: "List[BaseBlock]"
    
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_block_type_registry"):
            cls._block_type_registry = {}
        return super().__new__(cls)

        
    def __init__(
        self,
        value: Any | None = None,
        **kwargs,
        # ctx_stack: "List[Block] | None" = None, 
    ):
        # self._block = block
        # self._staged_inst = self._build_instance(value, *args, **kwargs)
        self._ctx_stack = []
        inst = self._build_instance(value, **kwargs)
        self._main_inst = inst
        # self._ctx_stack.append(inst)
    
    
    def __call__(self, value: Any, **kwargs):       
        self._append(value, **kwargs)
        return self
    
    def _append(self, value: Any, **kwargs):
        inst = self._build_instance(value, **kwargs)        
        self._ctx_stack[-1].append(inst)
        
        # Set parent reference for nested selector support
        if hasattr(inst, 'parent'):
            inst.parent = self._ctx_stack[-1]
            
        return inst

    
    @property
    def top(self):
        if not self._ctx_stack:
            raise ValueError("No context stack")
        return self._ctx_stack[-1]
        
    def __enter__(self):        
        if not self._ctx_stack:
            self._ctx_stack.append(self._main_inst)
        else:
            self._ctx_stack.append(self.top)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if len(self._ctx_stack) > 1:
            self._ctx_stack.pop()

        
    @classmethod
    def register(cls, typ: Type, block_type: Type[BaseBlock]):
        if not hasattr(cls, "_block_type_registry"):
            cls._block_type_registry = {}
        cls._block_type_registry[typ] = block_type    
    
        
    def _build_instance(self, value: Any, **kwargs):
        # if "depth" not in kwargs:
            # kwargs["depth"] = depth
        inst = self.__class__._block_type_registry[type(value)](value, **kwargs)
        inst.depth = len(self._ctx_stack) + 1
        return inst
    
    
    def __add__(self, value: Any):
        inst = self._append(value)
        return inst
    
    def __iadd__(self, value: Any):
        self._append(value)
        return self
    
    def render(self):
        # Compute styles before rendering
        self._ctx_stack[-1].compute_styles()
        return self._ctx_stack[-1].render()
    
    def style(self, **style_props: Any):
        """
        Add inline style properties to the current block
        """
        self.top.add_style(**style_props)
        return self
    
    
    
    
    
    


