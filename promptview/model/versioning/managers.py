from __future__ import annotations
import datetime as dt
from typing import Optional, List
from .models import Branch, Turn, TurnStatus, VersionedModel

class BranchManager:
    """Backend-agnostic branch operations."""

    async def fork_branch(self, from_turn: Turn, name: Optional[str] = None) -> Branch:
        """Create a new branch starting from a specific turn."""
        new_branch = Branch(
            name=name,
            forked_from_index=from_turn.index,
            forked_from_turn_id=from_turn.id,
            forked_from_branch_id=from_turn.branch_id,
            current_index=from_turn.index + 1
        )
        return await new_branch.save()

    async def get_descendants(self, branch_id: int) -> List[Branch]:
        """Get all child branches recursively."""
        raise NotImplementedError  # backend-specific

    async def get_ancestors(self, branch_id: int) -> List[Branch]:
        """Get all parent branches recursively."""
        raise NotImplementedError


class TurnManager:
    """Backend-agnostic turn operations."""

    async def start_turn(self, branch: Branch, **kwargs) -> Turn:
        """Increment branch index and create a new turn."""
        branch.current_index += 1
        await branch.save()

        turn = Turn(
            branch_id=branch.id,
            index=branch.current_index,
            status=TurnStatus.STAGED,
            created_at=dt.datetime.now(),
            **kwargs
        )
        return await turn.save()

    async def update_status(self, turn: Turn, status: TurnStatus, message: Optional[str] = None):
        turn.status = status
        turn.ended_at = dt.datetime.now()
        if message:
            turn.message = message
        return await turn.save()

    async def history(self, branch_id: int, status: Optional[TurnStatus] = None, limit: Optional[int] = None) -> List[Turn]:
        """List turns in a branch (with backend-specific recursion for forks)."""
        raise NotImplementedError

    async def rewind(self, branch: Branch, to_turn: Turn):
        """Delete all versioned models created after a given turn in this branch."""
        await VersionedModel.query() \
            .filter(branch_id=branch.id, turn_id__gt=to_turn.id) \
            .delete()
