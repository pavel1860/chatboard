from .block import Block, Mutator
from .mutator_meta import MutatorMeta, MutatorConfig, TargetType, style_registry_ctx, MutatorMeta





def render(block: Block, depth: int = 0) -> Block:
    if block.mutator.is_rendered:
        return block
        
    config = MutatorMeta.resolve(block.style if not block.is_wrapper else [])
    
    new_block = block.copy_head()
    path = block.path
    for child in block.children:
        child = render(child, depth + 1)
        new_block.append_child(child)
        # new_block.mutator.append_child(child)
        # new_block /= child
    mutator = config.mutator()
    tran_block = mutator.call_render(new_block, path)
    return tran_block
