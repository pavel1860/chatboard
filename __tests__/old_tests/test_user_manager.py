import pytest
from unittest.mock import AsyncMock, MagicMock
from chatboard.conversation.message_log import UserManager
from chatboard.conversation.models import User, Session, Message, UserBackend
import datetime as dt

@pytest.fixture
def mock_backend():
    backend = AsyncMock(spec=UserBackend)
    return backend

@pytest.fixture
def user_manager(mock_backend):
    return UserManager(backend=mock_backend)

@pytest.fixture
def sample_user_data():
    return {
        "name": "Test User",
        "email": "test@example.com",
        "phone_number": "1234567890",
        "type": "app_a_user"
    }

@pytest.fixture
def sample_user(sample_user_data):
    now = dt.datetime.now(dt.timezone.utc)
    user = User(created_at=now, updated_at=now, **sample_user_data)
    user._id = 1  # Set the ID after creation
    return user

@pytest.mark.asyncio
async def test_create_user(user_manager, mock_backend, sample_user_data, sample_user):
    mock_backend.create_user.return_value = sample_user
    
    result = await user_manager.create_user(**sample_user_data)
    
    mock_backend.create_user.assert_called_once_with(**sample_user_data)
    assert result == sample_user
    assert result.id == 1
    assert result.type == sample_user_data["type"]

@pytest.mark.asyncio
async def test_get_user(user_manager, mock_backend, sample_user):
    mock_backend.get_user.return_value = sample_user
    
    result = await user_manager.get_user(user_id=1)
    
    mock_backend.get_user.assert_called_once_with(user_id=1)
    assert result == sample_user

@pytest.mark.asyncio
async def test_get_user_not_found(user_manager, mock_backend):
    mock_backend.get_user.return_value = None
    
    result = await user_manager.get_user(user_id=999)
    
    mock_backend.get_user.assert_called_once_with(user_id=999)
    assert result is None

@pytest.mark.asyncio
async def test_list_users(user_manager, mock_backend, sample_user):
    another_user = User(created_at=dt.datetime.now(dt.timezone.utc), updated_at=dt.datetime.now(dt.timezone.utc), type="app_a_user")
    another_user._id = 2
    users = [sample_user, another_user]
    mock_backend.list_users.return_value = users
    
    result = await user_manager.list_users(limit=10, offset=0)
    
    mock_backend.list_users.assert_called_once_with(limit=10, offset=0)
    assert result == users
    assert len(result) == 2

@pytest.mark.asyncio
async def test_add_session(user_manager, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    session = Session(user_id="1", created_at=now, updated_at=now)
    session._id = 1
    mock_backend.add_session.return_value = session
    
    result = await user_manager.add_session(user_id=1)
    
    mock_backend.add_session.assert_called_once_with(user_id=1)
    assert result == session
    assert result.id == 1
    assert result.user_id == "1"

@pytest.mark.asyncio
async def test_list_user_sessions(user_manager, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    sessions = []
    for i in range(2):
        session = Session(user_id="1", created_at=now, updated_at=now)
        session._id = i + 1
        sessions.append(session)
    mock_backend.list_user_sessions.return_value = sessions
    
    result = await user_manager.list_user_sessions(user_id=1, limit=10, offset=0)
    
    mock_backend.list_user_sessions.assert_called_once_with(user_id=1, limit=10, offset=0)
    assert result == sessions
    assert len(result) == 2

@pytest.mark.asyncio
async def test_get_user_messages(user_manager, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    messages = []
    for i, (role, name) in enumerate([("user", "Test User"), ("assistant", "Assistant")]):
        message = Message(role=role, name=name, content=f"Message {i+1}", created_at=now)
        message._id = i + 1
        messages.append(message)
    mock_backend.get_user_messages.return_value = messages
    
    result = await user_manager.get_user_messages(user_id=1, limit=10, offset=0)
    
    mock_backend.get_user_messages.assert_called_once_with(user_id=1, limit=10, offset=0)
    assert result == messages
    assert len(result) == 2 