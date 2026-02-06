# block_renderer.py

from .base_renderers import RenderContext, renderer_registry, style_manager
from .block8 import Block, Child


def render(target: Block | str, parent_ctx: RenderContext | None = None, verbose: bool = False) -> str:
    """
    Render a Block (or string).
    Uses style registry if styles are attached, otherwise falls back to default concatenation.
    """
    if isinstance(target, str):
        return target

    if not isinstance(target, Block):
        raise ValueError(f"Invalid render target: {type(target)}")

    # resolve style
    style = style_manager.resolve(target)
    ctx = RenderContext(target, style, parent_ctx, verbose=verbose)

    # pick renderer
    fmt = ctx.get_style("block-format") or ctx.get_style("sentence-format") or None
    if fmt:
        renderer = renderer_registry.get(fmt, "block-format")
        children_content = [render(child.content, ctx) + child.sep for child in target.children]
        return renderer.render(ctx, target, "".join(children_content))

    # fallback: concatenate children
    parts = []
    for child in target.children:
        if isinstance(child.content, Block):
            parts.append(render(child.content, ctx))
        else:
            parts.append(str(child.content))
        parts.append(child.sep)
    return "".join(parts)
