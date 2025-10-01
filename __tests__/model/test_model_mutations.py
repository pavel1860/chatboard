import dotenv
dotenv.load_dotenv()
import os
import uuid
import pytest
import pytest_asyncio
from typing import Literal, List
from uuid import UUID


from promptview.auth.user_manager2 import AuthModel
from promptview.model.fields import ModelField, KeyField, RelationField
from promptview.model import Model, VersionedModel, ArtifactModel
from promptview.model.relation_model import RelationModel
from promptview.model.postgres2.pg_query_set import select
from promptview.model.context import Context
from promptview.model.versioning.models import Branch, Turn, TurnStatus
import datetime as dt


# from promptview.model.artifact_models import Artifact, VersioningStrategy
from promptview.model.namespace_manager2 import NamespaceManager
from promptview.model import Model, KeyField, ModelField, VersionedModel, Artifact


class Post(VersionedModel):
    """
    A simple block model with snapshot versioning.

    Blocks are created once and displayed in sequence.
    Updates modify the existing record (no version history needed).
    """
    id: int = KeyField(primary_key=True)
    text: str = ModelField()
    

@pytest_asyncio.fixture()
async def setup_db():
    """Ensure we start with a clean DB schema for the tests."""
    NamespaceManager.drop_all_tables()
    await NamespaceManager.initialize_all()
    yield
    NamespaceManager.drop_all_tables()
    
    

    
    
    
@pytest.mark.asyncio
async def test_artifact_model_creation(setup_db):
    """Test basic artifact model creation and properties."""
    branch = await Branch.get_main()
    with branch:
        async with branch.start_turn(auto_commit=False) as turn:
            post = await Post(text="Hello World").save()
    
    # Check that artifact_id is automatically generated
    assert post.id is not None
    assert isinstance(post.id, int)
    
    assert post.text == "Hello World"
    assert post.artifact_id is not None
    assert isinstance(post.artifact_id, int)
    assert post.artifact_id == 1
    
    
    
    
    
    
    
@pytest.mark.asyncio
async def test_artifact_model_update(setup_db):
    """Test basic artifact model update and properties."""
    branch = await Branch.get_main()
    with branch:
        async with branch.start_turn() as turn:
            post = await Post(text="Hello World").save()
            post.text = "Hello World2!!!"
            await post.save()
            assert post.text == "Hello World2!!!"
    
    assert post.text == "Hello World2!!!"
    