from .block import Block
from .mutator import MutatorConfig, MutatorMeta
from .mutator import BlockMutator




def transform(block: Block, depth: int = 0) -> Block:
    if block.is_rendered:
        return block
    config = MutatorMeta.resolve(block.style, default=BlockMutator)
    tran_block = config.mutator.create_block(block.text, tags=block.tags, role=block.role, style=block.style, attrs=block.attrs)
    
    for child in block.children:
        child = transform(child, depth + 1)
        tran_block.append_child(child)
        
    tran_block.commit()
    return tran_block