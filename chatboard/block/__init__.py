"""
Block12 - Simplified block system with local text storage.

This module provides:
- Block: Tree node with local text and chunk metadata
- BlockSchema: Template for creating blocks with structure
- BlockList: List container for multiple items
- BlockListSchema: Schema for defining list structures
- ChunkMeta: Lightweight metadata for text regions
- BlockChunk: Text content with metadata for frontend consumption
- Mutator: Strategy for style-aware block operations
- XmlParser: Streaming XML parser for building blocks from schema
- IndexPath: Block position via indices (e.g., "0.2.1")
- TagPath: Block position via tags (e.g., "response.thinking")
- pydantic_to_schema: Convert Pydantic models to BlockSchema
"""

from .chunk import ChunkMeta, BlockChunk
from .block import Block, ContentType
from .schema import BlockSchema, BlockList, BlockListSchema
from .mutator import (
    Mutator,
    MutatorMeta,
)
from .parsers import XmlParser, ParserEvent, ParserError
from .object_helpers import pydantic_to_schema, block_to_object, block_to_dict
from .path import IndexPath, TagPath
from .diff import (
    diff_blocks,
    BlockDiff,
    NodeDiff,
    FieldChange,
    get_text_diff,
    get_inline_diff,
    format_diff_tree,
    print_diff,
)

__all__ = [
    "Block",
    "BlockSchema",
    "BlockList",
    "BlockListSchema",
    "BlockChunk",
    "ChunkMeta",
    "Mutator",
    "MutatorMeta",
    "XmlParser",
    "ParserEvent",
    "ParserError",
    "IndexPath",
    "TagPath",
    "pydantic_to_schema",
    "block_to_object",
    "block_to_dict",
    "ContentType",
    # Diff
    "diff_blocks",
    "BlockDiff",
    "NodeDiff",
    "FieldChange",
    "get_text_diff",
    "get_inline_diff",
    "format_diff_tree",
    "print_diff",
]
