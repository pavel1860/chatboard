from contextlib import asynccontextmanager
from .models import Branch, Turn, TurnStatus
from .managers import TurnManager

class VersioningSession:
    def __init__(self, branch: Branch, turn_kwargs: dict | None = None):
        self.branch = branch
        self.turn_kwargs = turn_kwargs or {}
        self.turn: Turn | None = None

    async def __aenter__(self):
        # Create turn (backend manager will decide how to increment index)
        self.turn = await TurnManager().start_turn(self.branch, **self.turn_kwargs)
        Branch.set_current(self.branch)
        Turn.set_current(self.turn)
        return self.turn

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type:
            await TurnManager().update_status(self.turn, TurnStatus.REVERTED, message=str(exc))
        else:
            await TurnManager().update_status(self.turn, TurnStatus.COMMITTED)
        # Clear context
        Branch.set_current(None)
        Turn.set_current(None)

@asynccontextmanager
async def start_turn(branch: Branch, **kwargs):
    async with VersioningSession(branch, kwargs) as turn:
        yield turn
