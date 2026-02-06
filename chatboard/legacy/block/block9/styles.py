from typing import TYPE_CHECKING, Literal



StyleFormatType = Literal["title-format", "row-format", "list-format", "block-format", "list-layout"]
StyleProps = dict[StyleFormatType, str]
    
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
                
    def resolve_parent(self, block) -> dict:       
        resolved = {}
        if parent := block.parent:
            parent_style = self.resolve(parent)
            for k, v in parent_style.items():
                # if k not in ("format",):  # don't inherit format
                resolved[k] = v
        return resolved

    def resolve(self, block, use_parent: bool = False) -> dict:        
        resolved = {}

        # inherit parent styles — but NOT format
        if use_parent:
            if parent := block.parent:
                parent_style = self.resolve(parent)
                for k, v in parent_style.items():
                    # if k not in ("format",):  # don't inherit format
                    resolved[k] = v
        if hasattr(block, "styles"):
            styles = block.styles
        elif hasattr(block, "parent") and block.parent:
            styles = self.resolve(block.parent)
        else:
            styles = []
        for s in styles:
            if s in self.rules:
                resolved.update(self.rules[s])

        return resolved




    
