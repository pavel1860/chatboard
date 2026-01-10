"""
Block12 - Simplified block system with position-based text storage.

This module provides:
- Block: Tree node with text positions and chunk metadata
- ChunkMeta: Lightweight metadata for text regions
- Chunk: Text content with metadata for frontend consumption
- Mutator: Strategy for style-aware block operations
"""

from .chunk import ChunkMeta, Chunk
from .block import Block
from .mutator import (
    Mutator,
    MutatorMeta,
)

__all__ = [
    "Block",
    "Chunk",
    "ChunkMeta",
    "Mutator",
    "MutatorMeta",
]
