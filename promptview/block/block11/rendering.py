from .block import Block, Mutator
from .mutator_meta import MutatorMeta, MutatorConfig, TargetType, style_registry_ctx, MutatorMeta





# def render(block: Block, depth: int = 0) -> Block:
#     if block.mutator.is_rendered:
#         return block
        
#     config = MutatorMeta.resolve(block.style if not block.is_wrapper else [])
    
#     new_block = block.copy_head()
#     path = block.path
    
#     for child in block.children:
#         child = render(child, depth + 1)
#         # print(child.tags, "text:", child.span.text)
#         new_block.append_child(child)
#         # new_block.mutator._append_child_after()
#         # new_block.mutator.append_child(child)
#         # new_block /= child
#     mutator = config.mutator()
#     # print(block.tags, "text:", block.span.text)
#     tran_block = mutator.call_render(new_block, path)
#     if len(tran_block):
#         tran_block.mutator.join_blocks()
#     return tran_block



def render(block: Block, depth: int = 0) -> Block:
    if block.mutator.is_rendered:
        return block
        
    config = MutatorMeta.resolve(block.style if not block.is_wrapper else [])
    
    # new_block = block.copy_head()
    path = block.path
    
    mutator = config.mutator()
    tran_block = mutator.call_init(block.chunks(), path)
    tran_block.stylizers = [stylizer() for stylizer in config.stylizers]
    
    for child in block.children:
        child = render(child, depth + 1)
        tran_block.append_child(child)
    mutator.call_commit()
    
    # if len(tran_block):
        # tran_block.join("\n")
    return tran_block
