from __future__ import annotations
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from .backends.postgres import PostgresBranchManager, PostgresTurnManager
from .models import Branch, Turn, TurnStatus
from .managers import BranchManager, TurnManager

class VersionGraph:
    """
    Unified view of branches and turns, regardless of backend.
    """

    def __init__(self, branch_mgr: BranchManager | None = None, turn_mgr: TurnManager | None = None):
        self.branch_mgr = branch_mgr or PostgresBranchManager()
        self.turn_mgr = turn_mgr or PostgresTurnManager()

    async def branch_lineage(self, branch_id: int) -> List[Branch]:
        """
        Get the full lineage of a branch from its ancestors to itself.
        """
        ancestors = await self.branch_mgr.get_ancestors(branch_id)
        ancestors_sorted = sorted(ancestors, key=lambda b: b.created_at)
        return ancestors_sorted

    async def branch_tree(self, branch_id: int) -> List[Branch]:
        """
        Get the full tree of a branch including all descendants.
        """
        return await self.branch_mgr.get_descendants(branch_id)

    async def branch_turns(
        self,
        branch_id: int,
        status: Optional[TurnStatus] = None,
        limit: Optional[int] = None
    ) -> List[Turn]:
        """
        Get turns for this branch and its ancestors, ordered newest first.
        """
        return await self.turn_mgr.history(branch_id, status=status, limit=limit)

    async def latest_turn(self, branch_id: int, status: TurnStatus = TurnStatus.COMMITTED) -> Turn:
        """
        Get the latest turn for a branch.
        """
        turns = await self.branch_turns(branch_id, status=status, limit=1)
        if not turns:
            raise ValueError(f"No turns found for branch {branch_id}")
        return turns[0]
    
    async def latest_turn_or_none(self, branch_id: int, status: TurnStatus = TurnStatus.COMMITTED) -> Turn | None:
        """
        Get the latest turn for a branch, or None if there are no turns.
        """
        turns = await self.branch_turns(branch_id, status=status, limit=1)
        return turns[0] if turns else None

    async def rewind_to(self, branch: Branch, to_turn: Turn):
        """
        Remove all data created after `to_turn` in `branch`.
        """
        await self.turn_mgr.rewind(branch, to_turn)

    async def fork_from(self, from_turn: Turn, name: Optional[str] = None) -> Branch:
        """
        Create a new branch starting from a specific turn.
        """
        return await self.branch_mgr.fork_branch(from_turn, name=name)

    async def diff_turns(
        self,
        branch_a_id: int,
        branch_b_id: int
    ) -> Tuple[List[Turn], List[Turn]]:
        """
        Compare committed turns between two branches.
        Returns (turns_only_in_a, turns_only_in_b).
        """
        turns_a = await self.branch_turns(branch_a_id, status=TurnStatus.COMMITTED)
        turns_b = await self.branch_turns(branch_b_id, status=TurnStatus.COMMITTED)

        ids_a = {t.id for t in turns_a}
        ids_b = {t.id for t in turns_b}

        only_a = [t for t in turns_a if t.id not in ids_b]
        only_b = [t for t in turns_b if t.id not in ids_a]

        return only_a, only_b

    async def merge_branch(
        self,
        source_branch_id: int,
        target_branch_id: int
    ) -> List[Turn]:
        """
        Merge committed turns from source branch into target branch.
        (For now, naive append — could be improved with conflict resolution.)
        """
        only_source, _ = await self.diff_turns(source_branch_id, target_branch_id)
        merged_turns = []
        for turn in only_source:
            new_turn = await self.turn_mgr.start_turn(
                branch=Branch(id=target_branch_id),  # minimal object
                message=turn.message,
                metadata=turn.metadata
            )
            await self.turn_mgr.update_status(new_turn, TurnStatus.COMMITTED)
            merged_turns.append(new_turn)
        return merged_turns

    
    @asynccontextmanager
    async def branch_context(self, branch: Branch | int):
        """Set a branch in namespace context for downstream saves."""
        ns = Branch.get_namespace()
        token = ns.set_ctx(branch if isinstance(branch, Branch) else await Branch.get(branch))
        try:
            yield branch
        finally:
            ns.reset_ctx(token)

    @asynccontextmanager
    async def start_turn(self, branch: Branch | int | None = None, status: TurnStatus = TurnStatus.STAGED, **kwargs):
        """Start a turn in a given or current branch."""
        if branch is None:
            branch = Branch.get_namespace().get_ctx()
            if branch is None:
                raise ValueError("No branch provided and no branch in context")

        turn = await self.turn_mgr.start_turn(branch, status=status, **kwargs)

        # set in context
        token = Turn.get_namespace().set_ctx(turn)
        try:
            yield turn
            await self.turn_mgr.update_status(turn, TurnStatus.COMMITTED)
        except Exception as e:
            await self.turn_mgr.update_status(turn, TurnStatus.REVERTED, str(e))
            raise
        finally:
            Turn.get_namespace().reset_ctx(token)