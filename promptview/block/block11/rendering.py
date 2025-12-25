from .block import Block, Mutator
from .mutator_meta import MutatorMeta, MutatorConfig, TargetType, style_registry_ctx, MutatorMeta


# def render(block: Block) -> str:
#     mutator_config = MutatorMeta.resolve(block.style)
    
    
    
    
    
    
    
#     for target, mutator_cls in mutator_config.iter_transformers():
#         mutator = mutator_cls(block)
#         block = mutator.render(block)
#     return block