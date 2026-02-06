from .block7 import  BaseBlock, BlockChunk, Block, BlockSent, BlockList, ResponseBlock
from .renderers_base import RenderContext, renderer_registry, style_manager
from .style2 import StyleManager
from .renderers import *



def combine_content(left: str, right: str, sep: str):
    if left and right:
        return left + sep + right
    elif left:
        return left
    elif right:
        return right




def render(target, style=None, parent_ctx: RenderContext | None = None, verbose: bool = False):
    if not isinstance(target, BaseBlock):
        raise ValueError(f"Invalid block type: {type(target)}")
    if isinstance(target, Block):
        style = style_manager.resolve(target)
    ctx = RenderContext(target, style, parent_ctx, verbose=verbose)
    if isinstance(target, Block):
        return render_block(target, ctx)
    elif isinstance(target, BlockSent):
        return render_sentence(target, ctx)
    elif isinstance(target, BlockList):
        return render_list(target, ctx)
    
    else:
        raise ValueError(f"Invalid block type: {type(target)}")


def render_sentence(sentence: BlockSent, ctx: RenderContext):
    fmt = ctx.get_parent_style("sentence-format")
    renderer = renderer_registry.get(fmt, 'sentence-format')
    content = renderer.render(ctx, sentence)
    return content


def render_list(block_list: BlockList, ctx: RenderContext):
    fmt = ctx.get_parent_style("list-format")
    renderer = renderer_registry.get(fmt, 'list-format')
    children_content_list = [render(child, None, ctx) for child in block_list]
    content = renderer.render(ctx, block_list, children_content_list)
    return content
    
def render_block(block: Block, ctx: RenderContext):
    fmt = ctx.get_style("sentence-format")
    title_renderer = renderer_registry.get(fmt, 'sentence-format')
    title_content = title_renderer.render(ctx, block.root)
    
    children_content_list = [render(child, None, ctx) for child in block.children]
    children_fmt = ctx.get_style("list-format")
    children_renderer = renderer_registry.get(children_fmt, 'list-format')
    children_content = children_renderer.render(ctx, block, children_content_list)
    
    postfix_content = None
    if isinstance(block, ResponseBlock):
        if block.postfix is not None:
            postfix_content = render(block.postfix, None, ctx)
            
    block_fmt = ctx.get_style("block-format")
    block_renderer = renderer_registry.get(block_fmt, 'block-format')
    content = block_renderer.render(ctx, block, title_content, children_content, postfix_content)    
    return content







