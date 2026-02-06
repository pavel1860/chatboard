"""
Path - Represents a block's position in the tree.

Block12 separates index paths and tag paths:
- IndexPath: Position via indices (e.g., "0.2.1")
- TagPath: Position via tags (e.g., "response.thinking")

Both are immutable snapshots computed dynamically from a block's position.
They navigate through the "logical" tree structure using mutator.body,
making style-specific wrappers (like XML tags) transparent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, SupportsIndex, overload

if TYPE_CHECKING:
    from .block import Block


@dataclass(frozen=True)
class IndexPath:
    """
    Immutable snapshot of a block's position via indices.

    Uses mutator.body for navigation, making style wrappers transparent.

    Attributes:
        indices: Tuple of child indices from root to block, e.g., (0, 2, 1)

    Example:
        path = block.path
        print(path)              # "0.2.1"
        print(path.depth)        # 3

        if path1 < path2:        # Tree order comparison
            print("path1 comes before path2")

        if path1.is_ancestor_of(path2):
            print("path1 contains path2")
    """

    indices: tuple[int, ...]

    def __init__(self, indices: list[int] | tuple[int, ...]):
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, 'indices', tuple(indices))

    # -------------------------------------------------------------------------
    # Comparisons (lexicographic - tree order)
    # -------------------------------------------------------------------------

    def __lt__(self, other: IndexPath) -> bool:
        """Compare paths in tree order (depth-first, left-to-right)."""
        if not isinstance(other, IndexPath):
            return NotImplemented
        return self.indices < other.indices

    def __le__(self, other: IndexPath) -> bool:
        if not isinstance(other, IndexPath):
            return NotImplemented
        return self.indices <= other.indices

    def __gt__(self, other: IndexPath) -> bool:
        if not isinstance(other, IndexPath):
            return NotImplemented
        return self.indices > other.indices

    def __ge__(self, other: IndexPath) -> bool:
        if not isinstance(other, IndexPath):
            return NotImplemented
        return self.indices >= other.indices

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexPath):
            return NotImplemented
        return self.indices == other.indices

    def __hash__(self) -> int:
        return hash(self.indices)

    # -------------------------------------------------------------------------
    # Indexing and Slicing
    # -------------------------------------------------------------------------

    @overload
    def __getitem__(self, index: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> "IndexPath": ...

    def __getitem__(self, index: SupportsIndex | slice) -> "int | IndexPath":
        """
        Access indices by index or slice.

        Usage:
            path[0]      # First index
            path[-1]     # Last index
            path[1:3]    # Slice -> new IndexPath
            path[:-1]    # All but last -> new IndexPath (same as path.parent)
        """
        if isinstance(index, slice):
            return IndexPath(self.indices[index])
        return self.indices[index]

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    def is_ancestor_of(self, other: IndexPath) -> bool:
        """
        Check if this path is an ancestor of (or equal to) other.

        A path is an ancestor if it's a prefix of the other path.
        Example: (0, 2) is ancestor of (0, 2, 1)
        """
        if len(self.indices) > len(other.indices):
            return False
        return other.indices[:len(self.indices)] == self.indices

    def is_strict_ancestor_of(self, other: IndexPath) -> bool:
        """Check if this path is a strict ancestor (not equal) of other."""
        return self.is_ancestor_of(other) and self != other

    def is_descendant_of(self, other: IndexPath) -> bool:
        """Check if this path is a descendant of (or equal to) other."""
        return other.is_ancestor_of(self)

    def is_strict_descendant_of(self, other: IndexPath) -> bool:
        """Check if this path is a strict descendant (not equal) of other."""
        return other.is_strict_ancestor_of(self)

    def includes(self, other: IndexPath) -> bool:
        """Alias for is_ancestor_of - check if this path includes other."""
        return self.is_ancestor_of(other)

    def is_sibling_of(self, other: IndexPath) -> bool:
        """Check if paths share the same parent."""
        if len(self.indices) != len(other.indices):
            return False
        if len(self.indices) == 0:
            return True  # Both are root
        return self.indices[:-1] == other.indices[:-1]

    def common_ancestor(self, other: IndexPath) -> IndexPath:
        """
        Find the longest common ancestor path.

        Example:
            (0, 2, 1).common_ancestor((0, 2, 3)) -> (0, 2)
            (0, 1).common_ancestor((1, 2)) -> ()
        """
        common_indices = []
        for a, b in zip(self.indices, other.indices):
            if a != b:
                break
            common_indices.append(a)
        return IndexPath(common_indices)

    def __sub__(self, other: IndexPath) -> IndexPath:
        """
        Subtract an ancestor path from this path.

        Returns the relative path from other to self.
        Raises ValueError if other is not an ancestor of self.

        Example:
            IndexPath([0, 2, 1]) - IndexPath([0, 2]) -> IndexPath([1])
            IndexPath([0, 2, 1, 3]) - IndexPath([0]) -> IndexPath([2, 1, 3])
            IndexPath([0, 2]) - IndexPath([0, 2]) -> IndexPath([])  # root
        """
        if not isinstance(other, IndexPath):
            return NotImplemented
        if not other.is_ancestor_of(self):
            raise ValueError(f"Cannot subtract: {other} is not an ancestor of {self}")
        return IndexPath(self.indices[len(other.indices):])

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return depth (number of indices)."""
        return len(self.indices)

    @property
    def depth(self) -> int:
        """Return depth in tree (0 for root)."""
        return len(self.indices)

    @property
    def is_root(self) -> bool:
        """Check if this is the root path (empty indices)."""
        return len(self.indices) == 0

    @property
    def last(self) -> int | None:
        """Get the last index (position in parent's children)."""
        return self.indices[-1] if self.indices else None

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    @property
    def parent(self) -> IndexPath | None:
        """
        Get parent path.

        Returns None for root path.
        """
        if not self.indices:
            return None
        return IndexPath(self.indices[:-1])

    def child(self, index: int) -> IndexPath:
        """
        Create a child path.

        Args:
            index: Child index
        """
        return IndexPath(list(self.indices) + [index])

    def ancestors(self) -> Iterator[IndexPath]:
        """
        Iterate over all ancestor paths, from root to parent.

        Does not include self.
        """
        for i in range(len(self.indices)):
            yield IndexPath(self.indices[:i])

    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """String representation using indices: '0.2.1'"""
        if not self.indices:
            return ""
        return ".".join(str(i) for i in self.indices)

    def __repr__(self) -> str:
        return f"IndexPath({list(self.indices)})"

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    @classmethod
    def from_string(cls, s: str) -> IndexPath:
        """
        Parse path from string.

        Args:
            s: Path string like "0.2.1"

        Returns:
            IndexPath object
        """
        if not s:
            return cls([])
        indices = [int(i) for i in s.split(".")]
        return cls(indices)

    @classmethod
    def root(cls) -> IndexPath:
        """Create an empty root path."""
        return cls([])


@dataclass(frozen=True)
class TagPath:
    """
    Immutable snapshot of a block's position via tags.

    Collects tag names from root to block, useful for semantic navigation.

    Attributes:
        tags: Tuple of tags from root to block, e.g., ("response", "thinking")

    Example:
        tag_path = block.tag_path
        print(tag_path)          # "response.thinking"
        print(tag_path.depth)    # 2
    """

    tags: tuple[str, ...]

    def __init__(self, tags: list[str] | tuple[str, ...]):
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, 'tags', tuple(tags))

    # -------------------------------------------------------------------------
    # Comparisons
    # -------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TagPath):
            return NotImplemented
        return self.tags == other.tags

    def __hash__(self) -> int:
        return hash(self.tags)

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    def is_ancestor_of(self, other: TagPath) -> bool:
        """
        Check if this tag path is an ancestor of (or equal to) other.

        A path is an ancestor if it's a prefix of the other path.
        """
        if len(self.tags) > len(other.tags):
            return False
        return other.tags[:len(self.tags)] == self.tags

    def is_strict_ancestor_of(self, other: TagPath) -> bool:
        """Check if this path is a strict ancestor (not equal) of other."""
        return self.is_ancestor_of(other) and self != other

    def is_descendant_of(self, other: TagPath) -> bool:
        """Check if this path is a descendant of (or equal to) other."""
        return other.is_ancestor_of(self)

    def common_ancestor(self, other: TagPath) -> TagPath:
        """Find the longest common ancestor tag path."""
        common_tags = []
        for a, b in zip(self.tags, other.tags):
            if a != b:
                break
            common_tags.append(a)
        return TagPath(common_tags)

    def __sub__(self, other: TagPath) -> TagPath:
        """
        Subtract an ancestor tag path from this path.

        Returns the relative path from other to self.
        Raises ValueError if other is not an ancestor of self.
        """
        if not isinstance(other, TagPath):
            return NotImplemented
        if not other.is_ancestor_of(self):
            raise ValueError(f"Cannot subtract: {other} is not an ancestor of {self}")
        return TagPath(self.tags[len(other.tags):])

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return depth (number of tags)."""
        return len(self.tags)

    @property
    def depth(self) -> int:
        """Return depth in tree (0 for root)."""
        return len(self.tags)

    @property
    def is_root(self) -> bool:
        """Check if this is the root path (empty tags)."""
        return len(self.tags) == 0

    @property
    def last(self) -> str | None:
        """Get the last tag."""
        return self.tags[-1] if self.tags else None

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    @property
    def parent(self) -> TagPath | None:
        """Get parent tag path. Returns None for root."""
        if not self.tags:
            return None
        return TagPath(self.tags[:-1])

    def child(self, tag: str) -> TagPath:
        """Create a child tag path."""
        return TagPath(list(self.tags) + [tag])

    def ancestors(self) -> Iterator[TagPath]:
        """Iterate over all ancestor tag paths, from root to parent."""
        for i in range(len(self.tags)):
            yield TagPath(self.tags[:i])

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def contains_tag(self, tag: str) -> bool:
        """Check if this path contains the given tag."""
        return tag in self.tags

    def starts_with(self, tag: str) -> bool:
        """Check if path starts with the given tag."""
        return len(self.tags) > 0 and self.tags[0] == tag

    def ends_with(self, tag: str) -> bool:
        """Check if path ends with the given tag."""
        return len(self.tags) > 0 and self.tags[-1] == tag

    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """String representation using tags: 'response.thinking'"""
        return ".".join(self.tags)

    def __repr__(self) -> str:
        return f"TagPath({list(self.tags)})"

    def to_string(self, separator: str = ".") -> str:
        """String representation with custom separator."""
        return separator.join(self.tags)

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    @classmethod
    def from_string(cls, s: str, separator: str = ".") -> TagPath:
        """Parse tag path from string."""
        if not s:
            return cls([])
        return cls(s.split(separator))

    @classmethod
    def root(cls) -> TagPath:
        """Create an empty root tag path."""
        return cls([])


def compute_index_path(block: "Block") -> IndexPath:
    """
    Compute the index path for a block by walking up to the root.

    Uses mutator.body for parent-child relationships, making
    style wrappers (like XML tags) transparent to the path.

    Args:
        block: The block to compute the path for

    Returns:
        IndexPath representing the block's position in the logical tree
    """
    indices: list[int] = []

    current = block
    while current.parent is not None:
        parent = current.parent
        # Use body (through mutator) for logical children
        body = parent.body

        # Find index of current in parent's body using identity
        idx = None
        for i, sibling in enumerate(body):
            if sibling is current:
                idx = i
                break

        if idx is not None:
            indices.append(idx)
        # If not found, current is a transparent wrapper child - skip this level

        current = parent

    # Reverse since we walked from leaf to root
    indices.reverse()

    return IndexPath(indices)


def compute_tag_path(block: "Block") -> TagPath:
    """
    Compute the tag path for a block by walking up to the root.

    Collects the first tag from each block in the path.

    Args:
        block: The block to compute the tag path for

    Returns:
        TagPath representing the block's semantic position
    """
    tags: list[str] = []

    current = block
    while current.parent is not None:
        parent = current.parent
        # Use body (through mutator) for logical children
        body = parent.body

        # Check if current is in parent's body using identity
        in_body = False
        for sibling in body:
            if sibling is current:
                in_body = True
                break

        if in_body:
            # Collect first tag if present
            if current.tags:
                tags.append(current.tags[0])

        current = parent

    # Reverse since we walked from leaf to root
    tags.reverse()

    return TagPath(tags)
