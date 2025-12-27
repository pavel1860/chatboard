from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import textwrap
from typing import Any, Generator, Literal, Type, TypedDict, TYPE_CHECKING
import contextvars
from ...utils.function_utils import get_if_overridden, is_overridden
from ...utils.type_utils import UNSET, UnsetType

if TYPE_CHECKING:
    from .block import Mutator


style_registry_ctx = contextvars.ContextVar("style_registry_ctx", default={})

TargetType = Literal["content", "block", "children", "tree", "subtree"]



    
    
@dataclass
class MutatorConfig:
    mutator: "Type[Mutator]"
    hidden: "bool" = False
    is_wrapper: "bool" = False
    
    
    def __post_init__(self):
        from .block import Mutator
        if self.mutator is None:
            self.mutator = Mutator
    
    def get(self, key: str, default: "Type[MutatorConfig] | None" = None) -> "Type[MutatorConfig] | None":
        return getattr(self, key)
    
    def __getitem__(self, key: str) -> "Type[MutatorConfig] | None":
        return self.get(key)
    
    def __setitem__(self, key: str, value: "Type[MutatorConfig] | None"):            
        setattr(self, key, value)
        
    def iter_transformers(self):
        if self.is_wrapper:
            order = ["list", "body"]
        else:
            order = ["list", "block", "content", "body", "prefix"]
        for target in order:
            if transformer := self.get(target):
                yield (target, transformer)
        
        
        
    
        
        
class MutatorMeta(type):
    
    def __new__(cls, name, bases, attrs):        
        new_cls = super().__new__(cls, name, bases, attrs)
        if styles := attrs.get("styles"):
            for style in styles:
                style_registry_ctx.get()[style] = new_cls
        return new_cls
    

    
    @classmethod
    def list_styles(cls) -> list[str]:
        current = style_registry_ctx.get()
        return list(current.keys())
    
    @classmethod
    def resolve(
        cls, 
        styles: list[str] | None = None, 
    ) -> "MutatorConfig":
        from .block import Mutator
        current = style_registry_ctx.get()
        mutator_cfg = MutatorConfig(mutator=Mutator)
        if styles is None:
            return mutator_cfg
        for style in styles:
            if style == "hidden":
                mutator_cfg.hidden = True                
            elif style_cls := current.get(style): 
                mutator_cfg["mutator"] = style_cls                          
        return mutator_cfg
