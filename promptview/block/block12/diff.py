"""
Block Diff - Compare two block trees and identify changes.

Provides tree-structured diffs with convenience methods for iteration,
text diff generation, and change analysis.

Usage:
    from promptview.block.block12.diff import diff_blocks

    diff = diff_blocks(block_a, block_b)

    if not diff.is_identical:
        print(f"Found {diff.change_count} changes")
        print(diff.get_text_diff())

        for node in diff.iter_changes():
            print(f"{node.path}: {node.status}")
"""

from __future__ import annotations
import difflib
from typing import TYPE_CHECKING, Iterator, Literal, Any
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .block import Block


# =============================================================================
# Diff Models
# =============================================================================

class FieldChange(BaseModel):
    """Represents a change in a single field."""
    field: str
    old_value: Any
    new_value: Any

    def __repr__(self) -> str:
        return f"{self.field}: {self.old_value!r} → {self.new_value!r}"


class TextDiffHunk(BaseModel):
    """A contiguous region of text changes."""
    old_start: int
    old_lines: list[str]
    new_start: int
    new_lines: list[str]


class NodeDiff(BaseModel):
    """
    Diff for a single node in the block tree.

    Mirrors the block tree structure - each NodeDiff corresponds to a block
    at the same path position.
    """
    model_config = {"arbitrary_types_allowed": True}

    path: str = ""
    status: Literal["unchanged", "modified", "added", "removed"] = "unchanged"

    # Field-level changes (only populated if status == "modified")
    text_change: tuple[str, str] | None = None      # (old, new)
    role_change: tuple[str | None, str | None] | None = None
    tags_change: tuple[list[str], list[str]] | None = None
    style_change: tuple[list[str], list[str]] | None = None
    attrs_change: tuple[dict, dict] | None = None

    # For added/removed nodes, store the block data
    block_data: dict | None = None

    # Recursive children diffs
    children: list["NodeDiff"] = Field(default_factory=list)

    @property
    def has_field_changes(self) -> bool:
        """True if any fields changed (excluding children)."""
        return any([
            self.text_change,
            self.role_change,
            self.tags_change,
            self.style_change,
            self.attrs_change,
        ])

    @property
    def field_changes(self) -> list[FieldChange]:
        """Get list of field changes."""
        changes = []
        if self.text_change:
            changes.append(FieldChange(field="text", old_value=self.text_change[0], new_value=self.text_change[1]))
        if self.role_change:
            changes.append(FieldChange(field="role", old_value=self.role_change[0], new_value=self.role_change[1]))
        if self.tags_change:
            changes.append(FieldChange(field="tags", old_value=self.tags_change[0], new_value=self.tags_change[1]))
        if self.style_change:
            changes.append(FieldChange(field="style", old_value=self.style_change[0], new_value=self.style_change[1]))
        if self.attrs_change:
            changes.append(FieldChange(field="attrs", old_value=self.attrs_change[0], new_value=self.attrs_change[1]))
        return changes

    def iter_all(self) -> Iterator["NodeDiff"]:
        """Iterate this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.iter_all()

    def iter_changes(self) -> Iterator["NodeDiff"]:
        """Iterate only changed nodes (skip unchanged)."""
        if self.status != "unchanged":
            yield self
        for child in self.children:
            yield from child.iter_changes()

    def __repr__(self) -> str:
        if self.status == "unchanged":
            return f"NodeDiff(path={self.path!r}, unchanged)"
        elif self.status in ("added", "removed"):
            return f"NodeDiff(path={self.path!r}, {self.status})"
        else:
            fields = [fc.field for fc in self.field_changes]
            return f"NodeDiff(path={self.path!r}, modified: {fields})"


class BlockDiff(BaseModel):
    """
    Complete diff between two block trees.

    Provides tree-structured comparison with convenience methods for
    iteration, text diffing, and change analysis.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Source block hashes for identity
    hash_a: str
    hash_b: str

    # Tree-structured diff
    root: NodeDiff

    @property
    def is_identical(self) -> bool:
        """True if blocks are identical."""
        return self.hash_a == self.hash_b

    @property
    def change_count(self) -> int:
        """Count of changed nodes."""
        return sum(1 for _ in self.root.iter_changes())

    @property
    def has_structural_changes(self) -> bool:
        """True if children were added or removed."""
        for node in self.root.iter_changes():
            if node.status in ("added", "removed"):
                return True
        return False

    @property
    def has_content_changes(self) -> bool:
        """True if text content changed."""
        for node in self.root.iter_changes():
            if node.text_change:
                return True
        return False

    def iter_changes(self) -> Iterator[NodeDiff]:
        """Iterate all changed nodes."""
        return self.root.iter_changes()

    def iter_all(self) -> Iterator[NodeDiff]:
        """Iterate all nodes including unchanged."""
        return self.root.iter_all()

    def get_changes_by_status(self) -> dict[str, list[NodeDiff]]:
        """Group changes by status type."""
        result: dict[str, list[NodeDiff]] = {
            "added": [],
            "removed": [],
            "modified": [],
        }
        for node in self.iter_changes():
            result[node.status].append(node)
        return result

    def get_added_paths(self) -> list[str]:
        """Get paths of all added nodes."""
        return [n.path for n in self.iter_changes() if n.status == "added"]

    def get_removed_paths(self) -> list[str]:
        """Get paths of all removed nodes."""
        return [n.path for n in self.iter_changes() if n.status == "removed"]

    def get_modified_paths(self) -> list[str]:
        """Get paths of all modified nodes."""
        return [n.path for n in self.iter_changes() if n.status == "modified"]

    def summary(self) -> str:
        """Get human-readable summary of changes."""
        if self.is_identical:
            return "Blocks are identical"

        by_status = self.get_changes_by_status()
        parts = []
        if by_status["modified"]:
            parts.append(f"{len(by_status['modified'])} modified")
        if by_status["added"]:
            parts.append(f"{len(by_status['added'])} added")
        if by_status["removed"]:
            parts.append(f"{len(by_status['removed'])} removed")

        return ", ".join(parts)
    
    def __bool__(self) -> bool:
        """True if there are any changes."""
        return self.change_count > 0
    
    def print(self) -> None:
        """Print diff in human-readable format."""
        print(self.summary())
        print(format_diff_tree(self))

    def __repr__(self) -> str:
        return f"BlockDiff({self.summary()})"


# =============================================================================
# Diff Computation
# =============================================================================

def _compute_node_diff(
    block_a: "Block | None",
    block_b: "Block | None",
    path: str = "",
) -> NodeDiff:
    """
    Recursively compute diff between two blocks at the same position.

    Args:
        block_a: Block from tree A (None if added in B)
        block_b: Block from tree B (None if removed from A)
        path: Current path in tree

    Returns:
        NodeDiff for this position
    """
    # Handle added node
    if block_a is None and block_b is not None:
        return NodeDiff(
            path=path,
            status="added",
            block_data=block_b.model_dump(),
        )

    # Handle removed node
    if block_a is not None and block_b is None:
        return NodeDiff(
            path=path,
            status="removed",
            block_data=block_a.model_dump(),
        )

    # Both None - shouldn't happen but handle gracefully
    if block_a is None and block_b is None:
        return NodeDiff(path=path, status="unchanged")

    # Both exist - compare fields
    assert block_a is not None and block_b is not None

    node = NodeDiff(path=path)
    has_changes = False

    # Compare text
    if block_a.text != block_b.text:
        node.text_change = (block_a.text, block_b.text)
        has_changes = True

    # Compare role
    if block_a._role != block_b._role:
        node.role_change = (block_a._role, block_b._role)
        has_changes = True

    # Compare tags
    if block_a.tags != block_b.tags:
        node.tags_change = (list(block_a.tags), list(block_b.tags))
        has_changes = True

    # Compare style
    if block_a.style != block_b.style:
        node.style_change = (list(block_a.style), list(block_b.style))
        has_changes = True

    # Compare attrs
    if block_a.attrs != block_b.attrs:
        node.attrs_change = (dict(block_a.attrs), dict(block_b.attrs))
        has_changes = True

    # Compare children recursively
    children_a = list(block_a.children)
    children_b = list(block_b.children)
    max_children = max(len(children_a), len(children_b))

    for i in range(max_children):
        child_path = f"{path}.{i}" if path else str(i)
        child_a = children_a[i] if i < len(children_a) else None
        child_b = children_b[i] if i < len(children_b) else None

        child_diff = _compute_node_diff(child_a, child_b, child_path)
        node.children.append(child_diff)

        if child_diff.status != "unchanged":
            has_changes = True

    node.status = "modified" if has_changes else "unchanged"
    return node


def diff_blocks(block_a: "Block", block_b: "Block") -> BlockDiff:
    """
    Compute diff between two block trees.

    Args:
        block_a: Original block tree
        block_b: New block tree

    Returns:
        BlockDiff with tree-structured comparison

    Example:
        diff = diff_blocks(old_block, new_block)

        if not diff.is_identical:
            print(diff.summary())
            for change in diff.iter_changes():
                print(f"  {change.path}: {change.status}")
    """
    from promptview.versioning.block_storage import compute_block_hash

    hash_a = compute_block_hash(block_a)
    hash_b = compute_block_hash(block_b)

    # Quick identity check
    if hash_a == hash_b:
        return BlockDiff(
            hash_a=hash_a,
            hash_b=hash_b,
            root=NodeDiff(path="", status="unchanged"),
        )

    # Compute full diff
    root_diff = _compute_node_diff(block_a, block_b, "")

    return BlockDiff(
        hash_a=hash_a,
        hash_b=hash_b,
        root=root_diff,
    )


# =============================================================================
# Text Diff Utilities
# =============================================================================

def get_text_diff(
    block_a: "Block",
    block_b: "Block",
    context_lines: int = 3,
) -> str:
    """
    Get unified text diff of rendered blocks.

    Args:
        block_a: Original block
        block_b: New block
        context_lines: Number of context lines around changes

    Returns:
        Unified diff string
    """
    text_a = block_a.render()
    text_b = block_b.render()

    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile="block_a",
        tofile="block_b",
        n=context_lines,
    )

    return "".join(diff)


def get_side_by_side_diff(
    block_a: "Block",
    block_b: "Block",
    width: int = 80,
) -> str:
    """
    Get side-by-side diff of rendered blocks.

    Args:
        block_a: Original block
        block_b: New block
        width: Total width of output

    Returns:
        Side-by-side diff string
    """
    text_a = block_a.render()
    text_b = block_b.render()

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    differ = difflib.HtmlDiff()
    # Use ndiff for simpler side-by-side
    diff_lines = list(difflib.ndiff(lines_a, lines_b))

    return "\n".join(diff_lines)


def get_inline_diff(
    block_a: "Block",
    block_b: "Block",
) -> list[tuple[str, str, str]]:
    """
    Get inline diff with change markers.

    Returns list of (marker, line_a, line_b) tuples:
    - marker: " " (unchanged), "-" (removed), "+" (added), "~" (modified)
    """
    text_a = block_a.render()
    text_b = block_b.render()

    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    result = []
    matcher = difflib.SequenceMatcher(None, lines_a, lines_b)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                result.append((" ", lines_a[i], lines_a[i]))
        elif tag == "delete":
            for i in range(i1, i2):
                result.append(("-", lines_a[i], ""))
        elif tag == "insert":
            for j in range(j1, j2):
                result.append(("+", "", lines_b[j]))
        elif tag == "replace":
            max_len = max(i2 - i1, j2 - j1)
            for k in range(max_len):
                line_a = lines_a[i1 + k] if i1 + k < i2 else ""
                line_b = lines_b[j1 + k] if j1 + k < j2 else ""
                result.append(("~", line_a, line_b))

    return result


# =============================================================================
# Diff Display Utilities
# =============================================================================

def format_diff_tree(diff: BlockDiff, indent: int = 2) -> str:
    """
    Format diff as indented tree structure.

    Args:
        diff: BlockDiff to format
        indent: Spaces per indentation level

    Returns:
        Formatted string
    """
    lines = []

    def format_node(node: NodeDiff, depth: int = 0):
        prefix = " " * (indent * depth)

        # Status indicator
        status_char = {
            "unchanged": " ",
            "modified": "~",
            "added": "+",
            "removed": "-",
        }[node.status]

        # Build line
        path_display = node.path or "(root)"
        line = f"{prefix}{status_char} {path_display}"

        # Add field changes for modified nodes
        if node.status == "modified" and node.has_field_changes:
            fields = [fc.field for fc in node.field_changes]
            line += f" [{', '.join(fields)}]"

        lines.append(line)

        # Recurse into children
        for child in node.children:
            format_node(child, depth + 1)

    format_node(diff.root)
    return "\n".join(lines)


def print_diff(diff: BlockDiff, show_text_diff: bool = True) -> None:
    """
    Print diff in human-readable format.

    Args:
        diff: BlockDiff to print
        show_text_diff: Whether to include text diff
    """
    print(f"=== Block Diff: {diff.summary()} ===")
    print()
    print("Tree structure:")
    print(format_diff_tree(diff))

    if show_text_diff and diff.has_content_changes:
        print()
        print("Field changes:")
        for node in diff.iter_changes():
            if node.has_field_changes:
                print(f"  {node.path or '(root)'}:")
                for fc in node.field_changes:
                    if fc.field == "text":
                        # Truncate long text
                        old = fc.old_value[:50] + "..." if len(fc.old_value) > 50 else fc.old_value
                        new = fc.new_value[:50] + "..." if len(fc.new_value) > 50 else fc.new_value
                        print(f"    {fc.field}: {old!r} → {new!r}")
                    else:
                        print(f"    {fc.field}: {fc.old_value!r} → {fc.new_value!r}")
