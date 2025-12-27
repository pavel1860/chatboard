"""
Path - Represents a block's position in the tree.

Path is an immutable snapshot computed dynamically from a block's position.
It navigates through the "logical" tree structure using mutator.body,
making style-specific wrappers (like XML tags) transparent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .block import Block


@dataclass(frozen=True)
class Path:
    """
    Immutable snapshot of a block's position in the logical tree.

    Uses mutator.body for navigation, making style wrappers transparent.
    XmlMutator's opening/closing tag structure is invisible to paths.

    Attributes:
        indices: Tuple of child indices from root to block, e.g., (0, 2, 1)
        tags: Tuple of tags from root to block, e.g., ("response", "thinking")

    Example:
        path = block.path
        print(path)              # "0.2.1"
        print(path.tag_str())    # "response.thinking"
        print(path.depth)        # 3

        if path1 < path2:        # Tree order comparison
            print("path1 comes before path2")

        if path1.is_ancestor_of(path2):
            print("path1 contains path2")
    """

    indices: tuple[int, ...]
    tags: tuple[str, ...]

    def __init__(self, indices: list[int] | tuple[int, ...], tags: list[str] | tuple[str, ...] | None = None):
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, 'indices', tuple(indices))
        object.__setattr__(self, 'tags', tuple(tags) if tags else ())

    # -------------------------------------------------------------------------
    # Comparisons (lexicographic - tree order)
    # -------------------------------------------------------------------------

    def __lt__(self, other: Path) -> bool:
        """Compare paths in tree order (depth-first, left-to-right)."""
        if not isinstance(other, Path):
            return NotImplemented
        return self.indices < other.indices

    def __le__(self, other: Path) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self.indices <= other.indices

    def __gt__(self, other: Path) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self.indices > other.indices

    def __ge__(self, other: Path) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self.indices >= other.indices

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self.indices == other.indices

    def __hash__(self) -> int:
        return hash(self.indices)

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    def is_ancestor_of(self, other: Path) -> bool:
        """
        Check if this path is an ancestor of (or equal to) other.

        A path is an ancestor if it's a prefix of the other path.
        Example: (0, 2) is ancestor of (0, 2, 1)
        """
        if len(self.indices) > len(other.indices):
            return False
        return other.indices[:len(self.indices)] == self.indices

    def is_strict_ancestor_of(self, other: Path) -> bool:
        """Check if this path is a strict ancestor (not equal) of other."""
        return self.is_ancestor_of(other) and self != other

    def is_descendant_of(self, other: Path) -> bool:
        """Check if this path is a descendant of (or equal to) other."""
        return other.is_ancestor_of(self)

    def is_strict_descendant_of(self, other: Path) -> bool:
        """Check if this path is a strict descendant (not equal) of other."""
        return other.is_strict_ancestor_of(self)

    def includes(self, other: Path) -> bool:
        """Alias for is_ancestor_of - check if this path includes other."""
        return self.is_ancestor_of(other)

    def is_sibling_of(self, other: Path) -> bool:
        """Check if paths share the same parent."""
        if len(self.indices) != len(other.indices):
            return False
        if len(self.indices) == 0:
            return True  # Both are root
        return self.indices[:-1] == other.indices[:-1]

    def common_ancestor(self, other: Path) -> Path:
        """
        Find the longest common ancestor path.

        Example:
            (0, 2, 1).common_ancestor((0, 2, 3)) -> (0, 2)
            (0, 1).common_ancestor((1, 2)) -> ()
        """
        common_indices = []
        common_tags = []

        for i, (a, b) in enumerate(zip(self.indices, other.indices)):
            if a != b:
                break
            common_indices.append(a)
            if i < len(self.tags) and i < len(other.tags):
                if self.tags[i] == other.tags[i]:
                    common_tags.append(self.tags[i])

        return Path(common_indices, common_tags)

    def __sub__(self, other: Path) -> Path:
        """
        Subtract an ancestor path from this path.

        Returns the relative path from other to self.
        Raises ValueError if other is not an ancestor of self.

        Example:
            Path([0, 2, 1]) - Path([0, 2]) -> Path([1])
            Path([0, 2, 1, 3]) - Path([0]) -> Path([2, 1, 3])
            Path([0, 2]) - Path([0, 2]) -> Path([])  # root
        """
        if not isinstance(other, Path):
            return NotImplemented
        if not other.is_ancestor_of(self):
            raise ValueError(f"Cannot subtract: {other} is not an ancestor of {self}")

        remaining_indices = self.indices[len(other.indices):]
        remaining_tags = self.tags[len(other.tags):] if len(self.tags) > len(other.tags) else ()
        return Path(remaining_indices, remaining_tags)

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
    def last_index(self) -> int | None:
        """Get the last index (position in parent's children)."""
        return self.indices[-1] if self.indices else None

    @property
    def last_tag(self) -> str | None:
        """Get the last tag."""
        return self.tags[-1] if self.tags else None

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    @property
    def parent(self) -> Path | None:
        """
        Get parent path.

        Returns None for root path.
        """
        if not self.indices:
            return None
        return Path(
            self.indices[:-1],
            self.tags[:-1] if self.tags else ()
        )

    def child(self, index: int, tag: str | None = None) -> Path:
        """
        Create a child path.

        Args:
            index: Child index
            tag: Optional tag for the child
        """
        new_tags = list(self.tags)
        if tag:
            new_tags.append(tag)
        return Path(
            list(self.indices) + [index],
            new_tags
        )

    def ancestors(self) -> Iterator[Path]:
        """
        Iterate over all ancestor paths, from root to parent.

        Does not include self.
        """
        for i in range(len(self.indices)):
            yield Path(
                self.indices[:i],
                self.tags[:i] if i <= len(self.tags) else self.tags
            )

    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """String representation using indices: '0.2.1'"""
        if not self.indices:
            return ""
        return ".".join(str(i) for i in self.indices)

    def __repr__(self) -> str:
        return f"Path({list(self.indices)}, {list(self.tags)})"

    def tag_str(self, separator: str = ".") -> str:
        """String representation using tags: 'response.thinking'"""
        return separator.join(self.tags)

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    @classmethod
    def from_string(cls, s: str) -> Path:
        """
        Parse path from string.

        Args:
            s: Path string like "0.2.1"

        Returns:
            Path object
        """
        if not s:
            return cls([], [])
        indices = [int(i) for i in s.split(".")]
        return cls(indices, [])

    @classmethod
    def root(cls) -> Path:
        """Create an empty root path."""
        return cls([], [])


def compute_path(block: "Block") -> Path:
    """
    Compute the path for a block by walking up to the root.

    Uses mutator.body for parent-child relationships, making
    style wrappers (like XML tags) transparent to the path.

    Args:
        block: The block to compute the path for

    Returns:
        Path representing the block's position in the logical tree
    """
    indices: list[int] = []
    tags: list[str] = []

    current = block
    while current.parent is not None:
        parent = current.parent
        # Use body (through mutator) for logical children
        body = parent.body

        # Find index of current in parent's body
        try:
            idx = body.index(current)
            indices.append(idx)
            # Collect first tag if present
            if current.tags:
                tags.append(current.tags[0])
        except ValueError:
            # current is not in parent.body - it's a transparent wrapper child
            # Skip this level (don't add to path)
            pass

        current = parent

    # Reverse since we walked from leaf to root
    indices.reverse()
    tags.reverse()

    return Path(indices, tags)
