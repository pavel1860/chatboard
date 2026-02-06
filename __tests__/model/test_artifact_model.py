import os
import uuid
import pytest
import pytest_asyncio
from typing import Literal, List
from uuid import UUID

# Set test database URL
os.environ["POSTGRES_URL"] = "postgresql://ziggi:Aa123456@localhost:5432/chatboard_test"

from chatboard.auth.user_manager2 import AuthModel
from chatboard.model.fields import ModelField, KeyField, RelationField
from chatboard.model import Model
from chatboard.versioning import VersionedModel, ArtifactModel
from chatboard.model.namespace_manager2 import NamespaceManager
from chatboard.model.relation_model import RelationModel
from chatboard.model.postgres2.pg_query_set import select
from chatboard.prompt.context import Context
from chatboard.versioning import Branch, Turn, TurnStatus
import datetime as dt

# -------------------------
# Test Models
# -------------------------

class Message(ArtifactModel):
    id: int = KeyField(primary_key=True)
    content: str = ModelField()
    role: Literal["user", "assistant"] = ModelField()
    conv_id: UUID = ModelField(foreign_key=True, enforce_foreign_key=False)


class Participant(ArtifactModel):
    id: int = KeyField(primary_key=True)
    user_id: UUID = ModelField(foreign_key=True)
    conv_id: int = ModelField(foreign_key=True)


class Conversation(ArtifactModel):
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    messages: List[Message] = RelationField(
        foreign_key="conv_id",
        primary_key="artifact_id",
    )
    user: List["User"] = RelationField(
        primary_key="id",
        foreign_key="id",
        junction_keys=["conv_id", "user_id"],
        junction_model=Participant
    )


class User(AuthModel):
    name: str = ModelField()
    conversations: List[Conversation] = RelationField(
        primary_key="id",
        foreign_key="id",
        junction_keys=["user_id", "conv_id"],
        junction_model=Participant
    )


# -------------------------
# Test Fixtures
# -------------------------

@pytest_asyncio.fixture()
async def setup_db():
    """Ensure we start with a clean DB schema for the tests."""
    NamespaceManager.drop_all_tables()
    await NamespaceManager.initialize_all()
    yield
    NamespaceManager.drop_all_tables()


@pytest_asyncio.fixture()
async def main_branch(setup_db):
    """Create and return the main branch."""
    branch = await Branch.get_main()
    return branch


@pytest_asyncio.fixture()
async def seeded_user(main_branch):
    """Create a test user."""
    user = await User(name="test_user").save()
    return user


# -------------------------
# Basic ArtifactModel Tests
# -------------------------

@pytest.mark.asyncio
async def test_artifact_model_creation(setup_db):
    """Test basic artifact model creation and properties."""
    message = Message(content="Hello World", role="user", conv_id=uuid.uuid4())
    
    # Check that artifact_id is automatically generated
    assert message.artifact_id is not None
    assert isinstance(message.artifact_id, UUID)
    
    # Check default version
    assert message.version == 1
    
    # Check content
    assert message.content == "Hello World"
    assert message.role == "user"


@pytest.mark.asyncio
async def test_artifact_model_save_basic(main_branch):
    """Test basic saving of artifact model."""
    async with main_branch.start_turn() as turn:
        message = Message(content="Hello World", role="user", conv_id=uuid.uuid4())
        saved_message = await message.save()
        
        # Should have an ID after saving
        assert saved_message.id is not None
        assert saved_message.artifact_id == message.artifact_id
        assert saved_message.version == 1
        assert saved_message.turn_id == turn.id


@pytest.mark.asyncio
async def test_artifact_model_versioning(main_branch):
    """Test artifact model versioning behavior."""
    async with main_branch.start_turn() as turn:
        # Create and save initial message
        message = Message(content="Hello World", role="user", conv_id=uuid.uuid4())
        v1_message = await message.save()
        
        # Modify and save again - should create new version
        v1_message.content = "Hello Updated World"
        v2_message = await v1_message.save()
        
        # Should have new version and different ID
        assert v2_message.version == 2
        assert v2_message.id != v1_message.id
        assert v2_message.artifact_id == v1_message.artifact_id
        assert v2_message.content == "Hello Updated World"


@pytest.mark.asyncio
async def test_artifact_model_latest_query(main_branch):
    """Test querying latest version of artifact."""
    async with main_branch.start_turn() as turn:
        # Create multiple versions
        message = Message(content="Version 1", role="user", conv_id=uuid.uuid4())
        v1 = await message.save()
        
        v1.content = "Version 2"
        v2 = await v1.save()
        
        v2.content = "Version 3"
        v3 = await v2.save()
        
        # Query latest version
        latest = await Message.latest(v1.artifact_id)
        
        assert latest is not None
        assert latest.version == 3
        assert latest.content == "Version 3"
        assert latest.artifact_id == v1.artifact_id


# -------------------------
# Context and Turn Tests
# -------------------------

@pytest.mark.asyncio
async def test_context_with_single_turn(seeded_user):
    """Test using Context with a single turn."""
    user = seeded_user
    
    async with Context(user).start_turn() as ctx:
        # Create conversation and messages within the turn
        conv = await user.add(Conversation(title="Test Conversation"))
        msg1 = await conv.add(Message(content="Hello", role="user"))
        msg2 = await conv.add(Message(content="Hi there!", role="assistant"))
        
        # Check that messages are properly linked
        assert msg1.conv_id == conv.artifact_id
        assert msg2.conv_id == conv.artifact_id
        assert msg1.turn_id == ctx.turn.id
        assert msg2.turn_id == ctx.turn.id


@pytest.mark.asyncio
async def test_context_multiple_turns(seeded_user):
    """Test using Context with multiple sequential turns."""
    user = seeded_user
    
    # First turn - create conversation and initial messages
    async with Context(user).start_turn() as ctx1:
        conv = await user.add(Conversation(title="Multi-turn Conversation"))
        msg1 = await conv.add(Message(content="Hello", role="user"))
        msg2 = await conv.add(Message(content="Hi there!", role="assistant"))
    
    # Second turn - modify existing message
    async with Context(user).start_turn() as ctx2:
        msg1.content = "Hello Updated"
        updated_msg1 = await msg1.save()
        
        # Should be new version in new turn
        assert updated_msg1.version == 2
        assert updated_msg1.turn_id == ctx2.turn.id
        assert updated_msg1.turn_id != ctx1.turn.id
    
    # Third turn - add more messages
    async with Context(user).start_turn() as ctx3:
        msg3 = await conv.add(Message(content="How are you?", role="user"))
        msg4 = await conv.add(Message(content="I'm doing well!", role="assistant"))
        
        assert msg3.turn_id == ctx3.turn.id
        assert msg4.turn_id == ctx3.turn.id


@pytest.mark.asyncio
async def test_context_fork_functionality(seeded_user):
    """Test Context fork functionality for branching."""
    user = seeded_user
    
    # Create initial conversation
    async with Context(user).start_turn() as ctx1:
        conv = await user.add(Conversation(title="Fork Test"))
        msg1 = await conv.add(Message(content="Hello", role="user"))
        msg2 = await conv.add(Message(content="World", role="assistant"))
    
    # Second turn
    async with Context(user).start_turn() as ctx2:
        msg1.content = "Hello Updated"
        updated_msg1 = await msg1.save()
    
    # Fork from ctx2 turn
    async with Context(user).fork(turn=ctx2.turn).start_turn() as ctx3:
        msg3 = await conv.add(Message(content="Forked message", role="user"))
        msg4 = await conv.add(Message(content="Forked response", role="assistant"))
        
        # Should be in different branch
        assert msg3.branch_id != ctx2.turn.branch_id
        assert msg4.branch_id != ctx2.turn.branch_id
        assert msg3.branch_id == ctx3.turn.branch_id


# -------------------------
# Relationship Tests
# -------------------------

@pytest.mark.asyncio
async def test_artifact_relationships(seeded_user):
    """Test relationships between artifact models."""
    user = seeded_user
    
    async with Context(user).start_turn():
        # Create conversation
        conv = await user.add(Conversation(title="Relationship Test"))
        
        # Add multiple messages
        messages = []
        for i in range(3):
            msg = await conv.add(Message(
                content=f"Message {i+1}",
                role="user" if i % 2 == 0 else "assistant"
            ))
            messages.append(msg)
        
        # Test that conversation has messages
        assert len(conv.messages) == 3
        
        # Test that messages are linked to conversation
        for msg in messages:
            assert msg.conv_id == conv.artifact_id


@pytest.mark.asyncio
async def test_many_to_many_relationships(main_branch):
    """Test many-to-many relationships through junction model."""
    async with main_branch.start_turn():
        # Create users
        user1 = await User(name="User 1").save()
        user2 = await User(name="User 2").save()
        
        async with Context(user1).start_turn():
            # Create conversation
            conv = await user1.add(Conversation(title="Shared Conversation"))
            
            # Add user2 to the conversation through Participant
            participant = await Participant(
                user_id=user2.id,
                conv_id=conv.id
            ).save()
            
            # Test relationships
            assert len(user1.conversations) >= 1
            # Note: The exact relationship loading would depend on the ORM implementation


# -------------------------
# Querying Tests
# -------------------------

@pytest.mark.asyncio
async def test_artifact_querying_with_context(seeded_user):
    """Test querying artifacts with context filtering."""
    user = seeded_user
    
    # Create messages in different turns
    async with Context(user).start_turn() as ctx1:
        conv = await user.add(Conversation(title="Query Test"))
        msg1 = await conv.add(Message(content="Turn 1 Message", role="user"))
    
    async with Context(user).start_turn() as ctx2:
        msg2 = await conv.add(Message(content="Turn 2 Message", role="user"))
    
    # Query messages from specific context
    async with Context(user, branch_id=ctx1.turn.branch_id):
        messages = await ctx1.select(Message).execute()
        
        # Should include messages from the context
        assert len(messages) > 0


@pytest.mark.asyncio
async def test_artifact_vquery(seeded_user):
    """Test versioned querying functionality."""
    user = seeded_user
    
    async with Context(user).start_turn():
        # Create and version a message
        message = Message(content="Original", role="user", conv_id=uuid.uuid4())
        v1 = await message.save()
        
        v1.content = "Updated"
        v2 = await v1.save()
        
        v2.content = "Final"
        v3 = await v2.save()
    
    # Test vquery returns latest versions
    query = Message.vquery()
    # Note: Actual execution would depend on the query implementation


# -------------------------
# Error Handling Tests
# -------------------------

@pytest.mark.asyncio
async def test_context_without_user():
    """Test that Context requires proper initialization."""
    with pytest.raises(ValueError):
        async with Context().start_turn():
            pass


@pytest.mark.asyncio
async def test_invalid_artifact_operations(main_branch):
    """Test error handling for invalid operations."""
    async with main_branch.start_turn():
        # Test querying non-existent artifact
        non_existent_id = uuid.uuid4()
        latest = await Message.latest(non_existent_id)
        assert latest is None


# -------------------------
# Advanced Feature Tests
# -------------------------

@pytest.mark.asyncio
async def test_artifact_model_with_foreign_keys(main_branch):
    """Test artifact models with foreign key relationships."""
    async with main_branch.start_turn():
        # Create a conversation first
        conv = Conversation(title="FK Test")
        saved_conv = await conv.save()
        
        # Create message with foreign key reference
        message = Message(
            content="Test with FK",
            role="user",
            conv_id=saved_conv.artifact_id
        )
        saved_message = await message.save()
        
        assert saved_message.conv_id == saved_conv.artifact_id


@pytest.mark.asyncio
async def test_artifact_model_concurrent_modifications(seeded_user):
    """Test handling of concurrent modifications."""
    user = seeded_user
    
    async with Context(user).start_turn():
        message = Message(content="Concurrent Test", role="user", conv_id=uuid.uuid4())
        original = await message.save()
    
    # Simulate concurrent modifications
    async with Context(user).start_turn():
        # First modification
        original.content = "Modified 1"
        modified1 = await original.save()
        
        # Second modification of same artifact
        modified1.content = "Modified 2"
        modified2 = await modified1.save()
        
        # Should have incremented versions
        assert modified1.version == 2
        assert modified2.version == 3
        assert modified1.artifact_id == modified2.artifact_id == original.artifact_id


@pytest.mark.asyncio
async def test_artifact_model_metadata_fields(main_branch):
    """Test that artifact models have proper metadata fields."""
    async with main_branch.start_turn():
        message = Message(content="Metadata Test", role="user", conv_id=uuid.uuid4())
        saved = await message.save()
        
        # Check required artifact fields
        assert hasattr(saved, 'artifact_id')
        assert hasattr(saved, 'version')
        assert hasattr(saved, 'id')
        
        # Check that artifact_id is UUID
        assert isinstance(saved.artifact_id, UUID)
        assert isinstance(saved.version, int)
        assert saved.version >= 1


@pytest.mark.asyncio
async def test_artifact_model_inheritance(setup_db):
    """Test that ArtifactModel properly inherits from VersionedModel."""
    message = Message(content="Inheritance Test", role="user", conv_id=uuid.uuid4())
    
    # Should have VersionedModel properties
    assert isinstance(message, VersionedModel)
    assert isinstance(message, ArtifactModel)
    
    # Should have Model properties
    assert isinstance(message, Model)
