"""
Block12 - Simplified block system with local text storage.

This module provides:
- Block: Tree node with local text and chunk metadata
- BlockSchema: Template for creating blocks with structure
- BlockList: List container for multiple items
- BlockListSchema: Schema for defining list structures
- ChunkMeta: Lightweight metadata for text regions
- Chunk: Text content with metadata for frontend consumption
- Mutator: Strategy for style-aware block operations
- XmlParser: Streaming XML parser for building blocks from schema
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
    "pydantic_to_schema",
    "block_to_object",
    "block_to_dict",
    "ContentType",
]
