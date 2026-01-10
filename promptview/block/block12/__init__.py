"""
Block12 - Simplified block system with position-based text storage.

This module provides:
- Block: Tree node with text positions and chunk metadata
- ChunkMeta: Lightweight metadata for text regions
- Mutator: Strategy for style-aware block operations
- XmlMutator: XML-style blocks with opening/closing tags
- MarkdownMutator: Markdown headings
- ListMutator: List items with bullet prefix
"""

from .chunk import ChunkMeta
from .block import Block
from .mutator import (
    Mutator,
    MutatorMeta,
    XmlMutator,
    MarkdownMutator,
    ListMutator,
)

__all__ = [
    "Block",
    "ChunkMeta",
    "Mutator",
    "MutatorMeta",
    "XmlMutator",
    "MarkdownMutator",
    "ListMutator",
]
