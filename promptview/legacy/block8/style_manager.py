from typing import TYPE_CHECKING, Literal

from promptview.block.types import StyleProps


if TYPE_CHECKING:
    from promptview.block.block7 import BlockChunk



    
class StyleManager:
    """
    Maps styles (list of strings) and paths to rendering properties.
    Supports inheritance + nested selectors.
    """
    def __init__(self):
        self.rules = {}  # list of (selector, props)

    def add_style(self, selector: str | list[str], props: StyleProps):
        if isinstance(selector, str):
            selector = [selector]
        elif isinstance(selector, list):        
            for s in selector:
                self.rules[s.strip()] = props        

    def resolve(self, block) -> dict:        
        resolved = {}

        # inherit parent styles — but NOT format
        if parent := block.parent:
            parent_style = self.resolve(parent)
            for k, v in parent_style.items():
                # if k not in ("format",):  # don't inherit format
                resolved[k] = v

        for s in block.styles:
            if s in self.rules:
                resolved.update(self.rules[s])

        return resolved




    
