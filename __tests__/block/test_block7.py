import pytest
from chatboard.block import BlockSent, BlockChunk, BlockList, Block
from chatboard.block.style import style_manager






def test_simple_block_with_tags():
    """Test creating a simple block with tags."""
    with Block(tags=["system"]) as b:
        b += "you are a helpful assistant"

    sent = BlockSent()
    sent.append(BlockChunk("<"))
    sent.append(BlockChunk("hello"))
    sent.append(BlockChunk(">\n"))

    s = sent.render()
    s[-1]
    # assert s[-1] == "\n"
    assert s == "<hello>"