




from .block_renderer2 import RendererRegistry
from .style2 import StyleManager


class BlockEngine:
    def __init__(self, style_manager: StyleManager, renderer_registry: RendererRegistry):
        self.styles = style_manager
        self.renderers = renderer_registry

    def render(self, block, depth=0):
        style = self.styles.resolve(block)

        # render the block content
        fmt = style.get("format", "plain")
        renderer = self.renderers.get(fmt)
        output = renderer.render(block, style, depth)

        # render children
        child_fmt = style.get("children-format", None)
        if child_fmt:
            child_renderer = self.renderers.get(child_fmt)
            for child in block.items:
                output += child_renderer.render(child, style, depth+1)
        else:
            for child in block.items:
                output += self.render(child, depth+1)

        return output
