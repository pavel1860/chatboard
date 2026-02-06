from datetime import datetime
from typing import List

from pydantic import BaseModel

from chatboard.model import Model
from chatboard.model.fields import KeyField, ModelField

class AuthModel(Model):
    id: int = KeyField(primary_key=True)
    name: str | None = ModelField(None)
    email: str = ModelField(...)
    image: str | None = ModelField(None)
    emailVerified: datetime = ModelField(..., db_type="TIMESTAMPTZ")
    is_admin: bool = ModelField(default=False)
    created_at: datetime = ModelField(default_factory=datetime.now)
    
    async def list_partitions(self):
        from chatboard.model.versioning import ArtifactLog
        return await ArtifactLog.list_partitions(self.id)
    
    async def create_partition(self, name: str, users: List["AuthModel"]):
        from chatboard.model.versioning import ArtifactLog
        return await ArtifactLog.create_partition(name, [self.id] + [user.id for user in users])

class UserAuthPayload(BaseModel):
    name: str | None = None
    email: str
    emailVerified: datetime
    image: str | None = None 