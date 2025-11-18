
from promptview.model.model import Model, MODEL
from promptview.model.fields import KeyField, ModelField
from promptview.model.versioning import Turn, Branch
from promptview.model.namespace_manager import NamespaceManager
from promptview.model.version_control_models import Turn, Branch
import uuid
import datetime as dt
from typing import Type






class ArtifactModel(Model):
    """
    A model that is versioned and belongs to a repo.
    """
    _is_base: bool = True
    id: int = KeyField(primary_key=True)
    artifact_id: uuid.UUID = KeyField(default=None, type="uuid")
    version: int = ModelField(default=1)    
    branch_id: int = ModelField(foreign_key=True)
    turn_id: int = ModelField(foreign_key=True)    
    created_at: dt.datetime = ModelField(default_factory=dt.datetime.now)
    updated_at: dt.datetime | None = ModelField(default=None)
    deleted_at: dt.datetime | None = ModelField(default=None)
    
    
    
    async def save(self, turn: int | Turn | None = None, branch: int | Branch | None = None):
        """
        Save the artifact model instance to the database
        
        Args:
            branch: Optional branch ID to save to
            turn: Optional turn ID to save to
        """        

        ns = NamespaceManager.get_namespace(self.__class__.get_namespace_name())
        result = await ns.save(self)
        # Update instance with returned data (e.g., ID)
        for key, value in result.items():
            setattr(self, key, value)
        
        # Update relation instance IDs
        self._update_relation_instance()
        
        return self
    
    @classmethod
    async def get_artifact(cls: Type[MODEL], artifact_id: uuid.UUID, version: int | None = None):
        """
        Get an artifact model instance by artifact ID and version
        """
        ns = cls.get_namespace()
        data = await ns.get_artifact(artifact_id, version)
        if data is None:
            raise ValueError(f"Artifact '{artifact_id}' with version '{version}' not found")
        instance = cls(**data)
        instance._update_relation_instance()
        return instance
    
    
    async def delete(self, turn: int | Turn | None = None, branch: int | Branch | None = None):
        """
        Delete the artifact model instance from the database
        """
        ns = self.get_namespace()
        data = self._payload_dump()
        result = await ns.delete(data=data, id=self.primary_id, artifact_id=self.artifact_id, version=self.version + 1, branch=branch, turn=turn)
        return result
