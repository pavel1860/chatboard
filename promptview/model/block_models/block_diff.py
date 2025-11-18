"""
Block Tree Diff Implementation

Efficiently compare two block trees using the block signature architecture.
Since signatures are content-addressed (content + styling), we can quickly
identify identical vs. changed blocks.
"""

from dataclasses import dataclass
from typing import Literal
from ...block import Block


DiffType = Literal["added", "removed", "modified", "moved", "unchanged"]


@dataclass
class BlockNodeDiff:
    """Represents a single node difference between two trees."""
    diff_type: DiffType
    path: str
    signature_id: str | None = None
    old_signature_id: str | None = None  # For modified nodes
    new_signature_id: str | None = None  # For modified nodes
    old_path: str | None = None  # For moved nodes
    new_path: str | None = None  # For moved nodes
    content: str | None = None
    role: str | None = None
    styles: list[str] | None = None
    tags: list[str] | None = None


@dataclass
class BlockTreeDiff:
    """
    Complete diff between two block trees.

    Efficiently compares trees using signature IDs:
    - Same signature_id + same path = unchanged
    - Same path + different signature_id = modified
    - Signature exists in both trees but different paths = moved
    - Signature only in tree2 = added
    - Signature only in tree1 = removed
    """
    tree1_id: int
    tree2_id: int
    added: list[BlockNodeDiff]
    removed: list[BlockNodeDiff]
    modified: list[BlockNodeDiff]
    moved: list[BlockNodeDiff]
    unchanged: list[BlockNodeDiff]

    @property
    def has_changes(self) -> bool:
        """Check if there are any differences between the trees."""
        return bool(self.added or self.removed or self.modified or self.moved)

    @property
    def change_summary(self) -> dict:
        """Get a summary of changes."""
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "moved": len(self.moved),
            "unchanged": len(self.unchanged),
            "total_changes": len(self.added) + len(self.removed) + len(self.modified) + len(self.moved)
        }

    def render_diff(self, context_lines: int = 3) -> str:
        """
        Render a human-readable diff similar to git diff.

        Args:
            context_lines: Number of unchanged lines to show around changes
        """
        lines = []
        lines.append(f"Diff between tree {self.tree1_id} and tree {self.tree2_id}")
        lines.append("=" * 60)

        if not self.has_changes:
            lines.append("No changes")
            return "\n".join(lines)

        # Show summary
        summary = self.change_summary
        lines.append(f"Changes: {summary['total_changes']} "
                    f"(+{summary['added']} -{summary['removed']} "
                    f"~{summary['modified']} ↔{summary['moved']})")
        lines.append("")

        # Show removed nodes
        if self.removed:
            lines.append("REMOVED:")
            for node in self.removed:
                content_preview = node.content[:50] if node.content else ''
                lines.append(f"  - [{node.path}] {content_preview}")
            lines.append("")

        # Show added nodes
        if self.added:
            lines.append("ADDED:")
            for node in self.added:
                content_preview = node.content[:50] if node.content else ''
                lines.append(f"  + [{node.path}] {content_preview}")
            lines.append("")

        # Show modified nodes
        if self.modified:
            lines.append("MODIFIED:")
            for node in self.modified:
                content_preview = node.content[:50] if node.content else ''
                lines.append(f"  ~ [{node.path}] {content_preview}")
                lines.append(f"    signature: {node.old_signature_id[:8]}... → {node.new_signature_id[:8]}...")
            lines.append("")

        # Show moved nodes
        if self.moved:
            lines.append("MOVED:")
            for node in self.moved:
                content_preview = node.content[:50] if node.content else ''
                lines.append(f"  ↔ {node.old_path} → {node.new_path}: {content_preview}")
            lines.append("")

        return "\n".join(lines)


async def diff_block_trees(tree1_id: int, tree2_id: int) -> BlockTreeDiff:
    """
    Compare two block trees and return a detailed diff.

    Uses efficient signature-based comparison:
    1. Load both trees with their nodes and signatures
    2. Compare by signature_id to detect identical blocks
    3. Compare by path to detect structural changes
    4. Categorize differences

    Args:
        tree1_id: ID of the first tree (baseline)
        tree2_id: ID of the second tree (comparison)

    Returns:
        BlockTreeDiff with categorized changes
    """
    from ...utils.db_connections import PGConnectionManager

    # Load nodes from both trees with signatures
    query = """
        SELECT
            bn.tree_id,
            bn.path,
            bn.signature_id,
            bs.role,
            bs.styles,
            bs.tags,
            b.content
        FROM block_nodes bn
        JOIN block_signatures bs ON bn.signature_id = bs.id
        JOIN blocks b ON bs.block_id = b.id
        WHERE bn.tree_id IN ($1, $2)
        ORDER BY bn.tree_id, bn.path
    """

    rows = await PGConnectionManager.fetch(query, tree1_id, tree2_id)

    # Separate nodes by tree
    tree1_nodes = {}  # path -> node data
    tree2_nodes = {}  # path -> node data
    tree1_sigs = {}   # signature_id -> path
    tree2_sigs = {}   # signature_id -> path

    for row in rows:
        node_data = {
            'path': row['path'],
            'signature_id': row['signature_id'],
            'content': row['content'],
            'role': row['role'],
            'styles': row['styles'],
            'tags': row['tags']
        }

        if row['tree_id'] == tree1_id:
            tree1_nodes[row['path']] = node_data
            tree1_sigs[row['signature_id']] = row['path']
        else:
            tree2_nodes[row['path']] = node_data
            tree2_sigs[row['signature_id']] = row['path']

    # Categorize differences
    added = []
    removed = []
    modified = []
    moved = []
    unchanged = []

    # Check all paths in tree1
    for path, node1 in tree1_nodes.items():
        if path in tree2_nodes:
            node2 = tree2_nodes[path]
            if node1['signature_id'] == node2['signature_id']:
                # Same signature at same path = unchanged
                unchanged.append(BlockNodeDiff(
                    diff_type="unchanged",
                    path=path,
                    signature_id=node1['signature_id'],
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))
            else:
                # Different signature at same path = modified
                modified.append(BlockNodeDiff(
                    diff_type="modified",
                    path=path,
                    old_signature_id=node1['signature_id'],
                    new_signature_id=node2['signature_id'],
                    content=node2['content'],
                    role=node2['role'],
                    styles=node2['styles'],
                    tags=node2['tags']
                ))
        else:
            # Path exists in tree1 but not tree2
            # Check if signature moved to different path
            sig_id = node1['signature_id']
            if sig_id in tree2_sigs and tree2_sigs[sig_id] != path:
                # Signature exists but at different path = moved
                moved.append(BlockNodeDiff(
                    diff_type="moved",
                    path=tree2_sigs[sig_id],
                    signature_id=sig_id,
                    old_path=path,
                    new_path=tree2_sigs[sig_id],
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))
            else:
                # Signature doesn't exist in tree2 = removed
                removed.append(BlockNodeDiff(
                    diff_type="removed",
                    path=path,
                    signature_id=sig_id,
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))

    # Check paths in tree2 that don't exist in tree1
    for path, node2 in tree2_nodes.items():
        if path not in tree1_nodes:
            sig_id = node2['signature_id']
            # Check if it's a moved node (already handled above)
            if sig_id not in tree1_sigs:
                # New signature = added
                added.append(BlockNodeDiff(
                    diff_type="added",
                    path=path,
                    signature_id=sig_id,
                    content=node2['content'],
                    role=node2['role'],
                    styles=node2['styles'],
                    tags=node2['tags']
                ))

    return BlockTreeDiff(
        tree1_id=tree1_id,
        tree2_id=tree2_id,
        added=added,
        removed=removed,
        modified=modified,
        moved=moved,
        unchanged=unchanged
    )


async def diff_block_objects(block1: Block, block2: Block) -> BlockTreeDiff:
    """
    Compare two Block objects directly without saving to database.

    Useful for comparing in-memory blocks before committing.

    Args:
        block1: First block (baseline)
        block2: Second block (comparison)

    Returns:
        BlockTreeDiff with categorized changes
    """
    from .block_log import dump_block, signature_hash, block_hash

    # Dump both blocks to get their structure
    nodes1 = dump_block(block1)
    nodes2 = dump_block(block2)

    # Build signature mappings
    tree1_nodes = {}
    tree2_nodes = {}
    tree1_sigs = {}
    tree2_sigs = {}

    for node in nodes1:
        blk_id = block_hash(node["content"], node["json_content"])
        sig_id = signature_hash(blk_id, node["styles"], node["role"], node["tags"], node["attrs"])

        node_data = {
            'path': node['path'],
            'signature_id': sig_id,
            'content': node['content'],
            'role': node['role'],
            'styles': node['styles'],
            'tags': node['tags']
        }
        tree1_nodes[node['path']] = node_data
        tree1_sigs[sig_id] = node['path']

    for node in nodes2:
        blk_id = block_hash(node["content"], node["json_content"])
        sig_id = signature_hash(blk_id, node["styles"], node["role"], node["tags"], node["attrs"])

        node_data = {
            'path': node['path'],
            'signature_id': sig_id,
            'content': node['content'],
            'role': node['role'],
            'styles': node['styles'],
            'tags': node['tags']
        }
        tree2_nodes[node['path']] = node_data
        tree2_sigs[sig_id] = node['path']

    # Same categorization logic as diff_block_trees
    added = []
    removed = []
    modified = []
    moved = []
    unchanged = []

    for path, node1 in tree1_nodes.items():
        if path in tree2_nodes:
            node2 = tree2_nodes[path]
            if node1['signature_id'] == node2['signature_id']:
                unchanged.append(BlockNodeDiff(
                    diff_type="unchanged",
                    path=path,
                    signature_id=node1['signature_id'],
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))
            else:
                modified.append(BlockNodeDiff(
                    diff_type="modified",
                    path=path,
                    old_signature_id=node1['signature_id'],
                    new_signature_id=node2['signature_id'],
                    content=node2['content'],
                    role=node2['role'],
                    styles=node2['styles'],
                    tags=node2['tags']
                ))
        else:
            sig_id = node1['signature_id']
            if sig_id in tree2_sigs and tree2_sigs[sig_id] != path:
                moved.append(BlockNodeDiff(
                    diff_type="moved",
                    path=tree2_sigs[sig_id],
                    signature_id=sig_id,
                    old_path=path,
                    new_path=tree2_sigs[sig_id],
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))
            else:
                removed.append(BlockNodeDiff(
                    diff_type="removed",
                    path=path,
                    signature_id=sig_id,
                    content=node1['content'],
                    role=node1['role'],
                    styles=node1['styles'],
                    tags=node1['tags']
                ))

    for path, node2 in tree2_nodes.items():
        if path not in tree1_nodes:
            sig_id = node2['signature_id']
            if sig_id not in tree1_sigs:
                added.append(BlockNodeDiff(
                    diff_type="added",
                    path=path,
                    signature_id=sig_id,
                    content=node2['content'],
                    role=node2['role'],
                    styles=node2['styles'],
                    tags=node2['tags']
                ))

    return BlockTreeDiff(
        tree1_id=0,  # No tree IDs for in-memory comparison
        tree2_id=0,
        added=added,
        removed=removed,
        modified=modified,
        moved=moved,
        unchanged=unchanged
    )
