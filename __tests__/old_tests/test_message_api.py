import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import datetime as dt
from unittest.mock import AsyncMock, patch, MagicMock
import json

from chatboard.conversation.message_api import router, MessageCreate, MessageUpdate
from chatboard.conversation.models import Message, Turn, Branch
from chatboard.conversation.message_log import MessageLog, MessageLogError

pytestmark = pytest.mark.asyncio

# Create test app
app = FastAPI()
app.include_router(router)

# Test client
client = TestClient(app)

# Mock data
@pytest.fixture
def sample_message_data():
    return {
        "role": "user",
        "name": "test_user",
        "content": "Hello, world!",
        "blocks": [{"type": "text", "content": "test"}],
        "extra": {"key": "value"},
        "run_id": "test-run-123",
        "platform_id": "test-platform",
        "ref_id": "ref-123"
    }

@pytest.fixture
def sample_message():
    return Message(
        id=1,
        created_at=dt.datetime.now(),
        role="user",
        name="test_user",
        content="Hello, world!",
        blocks=[{"type": "text", "content": "test"}],
        extra={"key": "value"},
        run_id="test-run-123",
        platform_id="test-platform",
        ref_id="ref-123",
        branch_order=0,
        branch_id=1,
        turn_id=1
    )

@pytest.fixture
def mock_message_log():
    message_log = AsyncMock(spec=MessageLog)
    message_log.head = MagicMock()
    message_log.head.branch = Branch(id=1, session_id=1)
    message_log.head.turn = Turn(id=1, branch_id=1)
    message_log.head.branch_id = 1
    message_log.head.turn_id = 1
    message_log.head.is_initialized = True
    return message_log

# Override dependency
@pytest.fixture(autouse=True)
def override_get_message_log(mock_message_log):
    async def mock_get_message_log():
        return mock_message_log
    with patch("chatboard.conversation.message_api.get_message_log", mock_get_message_log):
        yield mock_message_log

# Tests
@pytest.mark.asyncio
async def test_create_message_success(sample_message_data, mock_message_log, sample_message):
    # Setup mock
    mock_message_log.append.return_value = sample_message
    
    # Make request
    response = client.post("/api/messages/", json=sample_message_data)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == sample_message_data["role"]
    assert data["content"] == sample_message_data["content"]
    assert data["id"] == sample_message.id
    mock_message_log.append.assert_called_once()

@pytest.mark.asyncio
async def test_create_message_error(mock_message_log):
    # Setup mock to raise error
    mock_message_log.append.side_effect = MessageLogError("Test error")
    
    # Make request
    response = client.post("/api/messages/", json={
        "role": "user",
        "name": "test_user",
        "content": "test"
    })
    
    # Assertions
    assert response.status_code == 400
    assert response.json()["detail"] == "Test error"

@pytest.mark.asyncio
async def test_get_message_success(mock_message_log, sample_message):
    # Setup mock
    mock_message_log.get_messages.return_value = [sample_message]
    
    # Make request
    response = client.get(f"/api/messages/{sample_message.id}")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sample_message.id
    assert data["content"] == sample_message.content
    mock_message_log.get_messages.assert_called_once()

@pytest.mark.asyncio
async def test_get_message_not_found(mock_message_log):
    # Setup mock to return empty list
    mock_message_log.get_messages.return_value = []
    
    # Make request
    response = client.get("/api/messages/999")
    
    # Assertions
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

@pytest.mark.asyncio
async def test_list_messages_success(mock_message_log, sample_message):
    # Setup mock
    mock_message_log.get_messages.return_value = [sample_message]
    
    # Make request
    response = client.get("/api/messages/?limit=10&offset=0")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == sample_message.id
    mock_message_log.get_messages.assert_called_once_with(limit=10, offset=0, session_id=None)

@pytest.mark.asyncio
async def test_get_messages_in_turn_success(mock_message_log, sample_message):
    # Setup mock
    mock_message_log.get_messages.return_value = [sample_message]
    
    # Make request
    response = client.get(f"/api/messages/turn/{sample_message.turn_id}")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == sample_message.id
    mock_message_log.get_messages.assert_called_once()

@pytest.mark.asyncio
async def test_update_message_success(mock_message_log, sample_message):
    # Setup mock
    mock_message_log.get_messages.return_value = [sample_message]
    mock_message_log.update.return_value = None
    
    update_data = {
        "content": "Updated content",
        "extra": {"new": "value"}
    }
    
    # Make request
    response = client.patch(f"/api/messages/{sample_message.id}", json=update_data)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == update_data["content"]
    assert data["extra"] == update_data["extra"]
    mock_message_log.update.assert_called_once()

@pytest.mark.asyncio
async def test_update_message_not_found(mock_message_log):
    # Setup mock to return empty list
    mock_message_log.get_messages.return_value = []
    
    # Make request
    response = client.patch("/api/messages/999", json={"content": "test"})
    
    # Assertions
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

@pytest.mark.asyncio
async def test_delete_message_not_implemented():
    # Make request
    response = client.delete("/api/messages/1")
    
    # Assertions
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()

# Validation tests
@pytest.mark.asyncio
async def test_create_message_validation():
    # Missing required fields
    response = client.post("/api/messages/", json={})
    assert response.status_code == 422
    
    # Invalid role type
    response = client.post("/api/messages/", json={
        "role": 123,  # should be string
        "name": "test",
        "content": "test"
    })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_update_message_validation():
    # Invalid type for content
    response = client.patch("/api/messages/1", json={
        "content": {"invalid": "type"}  # should be string
    })
    assert response.status_code == 422
    
    # Invalid type for blocks
    response = client.patch("/api/messages/1", json={
        "blocks": "invalid"  # should be list
    })
    assert response.status_code == 422 