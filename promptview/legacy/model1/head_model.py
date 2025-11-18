from datetime import datetime
from typing import TYPE_CHECKING, List, Type
from promptview.artifact_log.artifact_log3 import Branch, Turn
from promptview.artifact_log.artifact_log3 import ArtifactLog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from promptview.model.model import Model

class Head(BaseModel):
    id: int
    main_branch_id: int | None = Field(default=None)
    branch_id: int | None = Field(default=None)
    turn_id: int | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    # is_detached: bool
    _artifact_log: ArtifactLog | None = None
    
    
    @property
    def artifact_log(self):
        if self._artifact_log is None:
            raise ValueError("Artifact log is not initialized")
        return self._artifact_log
    
    
    @classmethod
    async def from_head_id(cls, head_id: int):
        artifact_log = ArtifactLog()
        _head = await artifact_log.init_head(head_id)         
        head = cls(
            id=_head["id"],
            main_branch_id=_head["main_branch_id"],
            branch_id=_head["branch_id"],
            turn_id=_head["turn_id"],
            created_at=_head["created_at"],
            updated_at=_head["updated_at"],
            # is_detached=_head["is_detached"],
        )
        head._artifact_log = artifact_log
        return head

    
    async def checkout(
            self, 
            branch: Branch | None = None, 
            turn: Turn | None = None, 
            head: "Head | None" = None,
            model: Type["Model"] | None = None,
            branch_id: int | None = None,
            turn_id: int | None = None,
        ):
        if branch is not None:
            branch_id = branch.id
        if turn is not None:
            turn_id = turn.id
        if branch_id is not None:
            await self.artifact_log.checkout_branch(
                branch_id=branch_id,
                turn_id=turn_id
            )
        if turn is not None:
            raise ValueError("You need to specify a branch")
        if head is not None:
            await self.artifact_log.checkout_head(head_id=head.id)
        if model is not None:
            if not hasattr(model, "turn_id"):
                raise ValueError("Model has no turn_id attribute")
            if not hasattr(model, "branch_id"):
                raise ValueError("Model has no branch_id attribute")
            await self.artifact_log.checkout_branch(                
                branch_id=int(model.branch_id),
                turn_id=int(model.turn_id)
            )
        self._copy_artifact_log_head()
        return self
    
    
    def _copy_artifact_log_head(self):
        if self._artifact_log is None:
            raise ValueError("Artifact log is not initialized")
        _head = self._artifact_log._head
        self.id=_head["id"]
        self.main_branch_id=_head["main_branch_id"]
        self.branch_id=_head["branch_id"]
        self.turn_id=_head["turn_id"]
        # self.created_at=_head["created_at"],
        # self.updated_at=_head["updated_at"],
    
    
    
    async def branch_from(
        self,
        head: "Head | None" = None,
        turn: Turn | None = None, 
        turn_id: int | None = None
    ):
        if turn is not None:
            turn_id = turn.id
        if head is not None:
            turn_id = head.turn_id
        branch = await self.artifact_log.create_branch(turn_id=turn_id)
        await self.checkout(branch=branch)
        return branch


class HeadModel:
    _head_id: int | None = Field(default=None)
    _head: Head | None = Field(default=None)
    
    def __init__(self, **kwargs):
        head = kwargs.get("head")
        if head is None:
            raise ValueError("Head is not provided")
        
        self._head = head
        self._head_id = head.id
        
    @property
    def head(self):
        if self._head is None:
            raise ValueError("Head is not initialized")
        return self._head
    
    def checkout(self, turn: int):
        pass
    
    async def after_save(self, **kwargs):
        if "head_id" not in kwargs:
            raise ValueError("Head id is not provided")
        head_id = kwargs.get("head_id")
        self._head = await Head.from_head_id(head_id=head_id)
        return self
    
    async def after_load(self, **kwargs):
        if "head_id" not in kwargs:
            raise ValueError("Head id is not provided")
        head_id = kwargs.get("head_id")
        self._head = await Head.from_head_id(head_id=head_id)
        return self
    
    async def delete(self):
        await self.head.artifact_log.delete_head(self.head.id)
