from .block import Block, Mutator
from .mutator_meta import MutatorMeta, MutatorConfig, TargetType, style_registry_ctx, MutatorMeta





def render(block: Block, depth: int = 0) -> Block:
    config = MutatorMeta.resolve(block.style)
    
    new_block = block.copy_head()
    for child in block.children:
        child = render(child, depth + 1)
        new_block.append_child(child)
    
    tran_block = config["mutator"]().render_and_set(new_block)
    return tran_block
