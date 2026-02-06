import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from chatboard.conversation.models import User, Session, Message, UserBackend
from chatboard.conversation.message_log import UserManager
from chatboard.api.user_router import create_user_router
import datetime as dt

class TestUserModel(User):
    name: str
    email: str | None = None
    phone_number: str | None = None

@pytest.fixture
def mock_backend():
    backend = AsyncMock(spec=UserBackend)
    return backend

@pytest.fixture
def user_manager(mock_backend):
    return UserManager(backend=mock_backend)

@pytest.fixture
def app(mock_backend):
    router = create_user_router(TestUserModel, mock_backend)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def sample_user_data():
    return {
        "name": "Test User",
        "email": "test@example.com",
        "phone_number": "1234567890",
        "type": "test_user"
    }

@pytest.fixture
def sample_user(sample_user_data):
    now = dt.datetime.now(dt.timezone.utc)
    user = TestUserModel(created_at=now, updated_at=now, **sample_user_data)
    user._id = 1
    return user

def test_create_user(client, mock_backend, sample_user_data, sample_user):
    mock_backend.create_user.return_value = sample_user
    
    response = client.post("/users/", json=sample_user_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == sample_user_data["name"]
    assert data["email"] == sample_user_data["email"]
    assert data["phone_number"] == sample_user_data["phone_number"]
    mock_backend.create_user.assert_called_once()

def test_get_user(client, mock_backend, sample_user):
    mock_backend.get_user.return_value = sample_user
    
    response = client.get("/users/1")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["name"] == sample_user.name
    mock_backend.get_user.assert_called_once_with(user_id=1)

def test_get_user_not_found(client, mock_backend):
    mock_backend.get_user.return_value = None
    
    response = client.get("/users/999")
    
    assert response.status_code == 404
    assert response.json()["detail"] == "User with id 999 not found"

def test_list_users(client, mock_backend, sample_user):
    now = dt.datetime.now(dt.timezone.utc)
    another_user = TestUserModel(created_at=now, updated_at=now, name="Another User", type="test_user")
    another_user._id = 2
    users = [sample_user, another_user]
    mock_backend.list_users.return_value = users
    
    response = client.get("/users/")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == users[0].name
    mock_backend.list_users.assert_called_once_with(limit=10, offset=0)

def test_create_user_session(client, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    session = Session(user_id="1", created_at=now, updated_at=now)
    session._id = 1
    mock_backend.add_session.return_value = session
    
    response = client.post("/users/1/sessions")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["user_id"] == "1"
    mock_backend.add_session.assert_called_once_with(user_id=1)

def test_create_user_session_not_found(client, mock_backend):
    mock_backend.add_session.side_effect = ValueError("User with id 999 not found")
    
    response = client.post("/users/999/sessions")
    
    assert response.status_code == 404
    assert response.json()["detail"] == "User with id 999 not found"

def test_list_user_sessions(client, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    sessions = []
    for i in range(2):
        session = Session(user_id="1", created_at=now, updated_at=now)
        session._id = i + 1
        sessions.append(session)
    mock_backend.list_user_sessions.return_value = sessions
    
    response = client.get("/users/1/sessions")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == 1
    mock_backend.list_user_sessions.assert_called_once_with(user_id=1, limit=10, offset=0)

def test_get_user_messages(client, mock_backend):
    now = dt.datetime.now(dt.timezone.utc)
    messages = []
    for i, (role, name) in enumerate([("user", "Test User"), ("assistant", "Assistant")]):
        message = Message(role=role, name=name, content=f"Message {i+1}", created_at=now)
        message._id = i + 1
        messages.append(message)
    mock_backend.get_user_messages.return_value = messages
    
    response = client.get("/users/1/messages")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[1]["role"] == "assistant"
    mock_backend.get_user_messages.assert_called_once_with(user_id=1, limit=10, offset=0) 