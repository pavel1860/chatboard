

from .block7 import Block


def print_block_tree(block: Block, indent: int = 0):
    pad = "  " * indent

    # current block line
    meta = []
    root = block.root + " " if block.root else ""
    if block.role:   meta.append(f"role={block.role}")
    if block.tags:   meta.append(f"labels={block.tags}")
    # if block.dtype:  meta.append(f"dtype={block.dtype.__name__}")
    meta_str = " ".join(meta)
    print(f"{pad}Block('{root}' idx={block.index}{' ' + meta_str if meta_str else ''})")
    
    for child in block.children:
        print_block_tree(child, indent + 1)