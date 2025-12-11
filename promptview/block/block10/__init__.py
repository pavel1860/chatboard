"""
Block10 - Unified Block System

A redesigned block system that unifies user-defined prompts and parsed LLM responses
through clean separation of storage (BlockText) from structure (Block).

Key Components:
- Chunk: Atomic unit of text with optional LLM metadata (logprob)
- BlockText: Linked-list storage for chunks
- Span/VirtualBlockText: Views into BlockText
- Block: Tree structure with styles and tags
- Style: Unified parse + render logic

Example Usage:

    # User-defined prompt
    text = BlockText()
    text.append(Chunk("Hello world"))
    block = Block(text=VirtualBlockText(text), tags=["greeting"], styles=["xml"])
    print(block.render())  # <greeting>Hello world</greeting>

    # Parsed LLM response
    chunks = [Chunk("<response>"), Chunk("The answer is 42"), Chunk("</response>")]
    text = BlockText(chunks)
    parser = SchemaParser(schema, text)
    block = parser.parse()
    block.append_text("!")  # Edit the response
"""

from .chunk import Chunk, BlockText
from .span import SpanAnchor, Span, VirtualBlockText
from .block import BlockBase, Block, BlockSchema, AttrSchema

__all__ = [
    # Core data structures
    "Chunk",
    "BlockText",
    "SpanAnchor",
    "Span",
    "VirtualBlockText",
    # Block classes
    "BlockBase",
    "Block",
    "BlockSchema",
    "AttrSchema",
]
