"""
Block12 - Simplified block system with position-based text storage.

This module provides:
- Block: Tree node with text positions and chunk metadata
- BlockSchema: Template for creating blocks with structure
- ChunkMeta: Lightweight metadata for text regions
- Chunk: Text content with metadata for frontend consumption
- Mutator: Strategy for style-aware block operations
- XmlParser: Streaming XML parser for building blocks from schema
"""

from .chunk import ChunkMeta, Chunk
from .block import Block
from .schema import BlockSchema
from .mutator import (
    Mutator,
    MutatorMeta,
)
from .parsers import XmlParser, ParserEvent, ParserError

__all__ = [
    "Block",
    "BlockSchema",
    "Chunk",
    "ChunkMeta",
    "Mutator",
    "MutatorMeta",
    "XmlParser",
    "ParserEvent",
    "ParserError",
]
