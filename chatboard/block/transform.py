from .block import Block
from .mutator import MutatorConfig, MutatorMeta
from .mutator import BlockMutator
from contextvars import ContextVar
from contextlib import contextmanager


_transform_ctx: ContextVar[bool] = ContextVar("transform_ctx", default=False)

@contextmanager
def transform_context(transform: bool = True):
    token = _transform_ctx.set(transform)
    try:
        yield
    finally:
        _transform_ctx.reset(token)
        
        
def is_transforming() -> bool:
    return _transform_ctx.get()


def transform(block: Block) -> Block:    
    return _transform(block)

def _transform(block: Block, depth: int = 0) -> Block:
    if block.is_rendered:
        return block
    config = MutatorMeta.resolve(block.style if not block.is_wrapper or len(block.style) > 0 else [], default=BlockMutator)
    # tran_block = config.mutator.create_block(block.text, tags=block.tags, role=block.role, style=block.style, attrs=block.attrs)
    # tran_block = config.create_block(block.text, tags=block.tags, role=block.role, style=block.style, attrs=block.attrs)
    tran_block = config.build_block(block)
    
    if tran_block.mutator.target != "block":
        for child in block.children:
            child = _transform(child, depth + 1)
            tran_block.append_child(child)
        
    tran_block.commit()
    return tran_block