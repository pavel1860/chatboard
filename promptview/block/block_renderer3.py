from typing import Callable
from .block7 import Block, BlockSequence, BlockSent, BlockChunk
from .utils import print_block_tree
from sys import getsizeof


b = Block("Hello")
b += "World"
b &= "!"


b /= "one", "subsentence", "number", "1"
b /= "two", "subsentence", "number", "2"




def render_chunk(chunk: BlockChunk):
    return chunk.content



def render_block(block: Block, verbose: bool = False):
    if verbose:
        print(block.path, "Block")
    root_content = render_sequence(block.root, verbose=verbose)
    if verbose:
        print( block.path, "Root content", root_content)
    children_content = render_sequence(block, verbose=verbose) if len(block.children) > 0 else None
    if verbose:
        print( block.path, "Children content", children_content)
    content = root_content
    if children_content:
        content += f"\n{children_content}"
    return content


def get_renderer(block: BlockSequence):
    if isinstance(block, BlockSent):
        return render_chunk
    elif isinstance(block, Block):
        return render_block
    else:
        raise ValueError(f"Unknown block type: {type(block)}")

def render_sequence(block_seq: BlockSequence, renderer: Callable | None = None, verbose: bool = False):
    if verbose:
        print( block_seq.path, "Render sequence")
    content_list= []
    if not renderer:
        renderer = get_renderer(block_seq)
    for sep, block in block_seq.iter_chunks():
        content = renderer(block)
        content_list.append(sep)
        content_list.append(content)  
    if verbose:
        print(block_seq.path,"content_list", content_list)
    return "".join(content_list)