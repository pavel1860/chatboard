from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import datetime as dt

from promptview.conversation.message_log import MessageLog, MessageLogError
from promptview.conversation.models import Message, MessageBackend, Turn, Branch, Session

# Pydantic models for request/response
class MessageCreate(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    name: str = Field(..., description="Name of the message sender")
    content: str = Field(..., description="Content of the message")
    blocks: Optional[List[dict]] = Field(None, description="Optional structured blocks in the message")
    extra: dict = Field(default={}, description="Extra metadata for the message")
    run_id: Optional[str] = Field(None, description="Optional run ID for tracking")
    platform_id: Optional[str] = Field(None, description="Optional platform identifier")
    ref_id: Optional[str] = Field(None, description="Optional reference ID")

class MessageUpdate(BaseModel):
    role: Optional[str] = None
    name: Optional[str] = None
    content: Optional[str] = None
    blocks: Optional[List[dict]] = None
    extra: Optional[dict] = None
    platform_id: Optional[str] = None
    ref_id: Optional[str] = None

class MessageResponse(BaseModel):
    id: int
    created_at: dt.datetime
    role: str
    name: str
    content: str
    blocks: Optional[List[dict]]
    extra: dict
    run_id: Optional[str]
    platform_id: Optional[str]
    ref_id: Optional[str]
    branch_order: Optional[int]
    branch_id: int
    turn_id: int

    class Config:
        from_attributes = True

# Router setup
router = APIRouter(prefix="/messages", tags=["messages"])

# Dependency for MessageLog instance
async def get_message_log(session_id: int) -> MessageLog:
    # USER_ID = "test1"
    # message_log = await MessageLog.from_user_last_session(user_id=USER_ID)
    message_log = await MessageLog.from_session(session_id=session_id)
    # Create a test session and branch if not exists
    # session = await backend.add_session(Session(user_id=USER_ID))
    # branch = await backend.add_branch(Branch(session_id=session.id, branch_order=0, message_counter=0))
    # message_log.head._branch = branch
    
    return message_log

# API Endpoints
@router.post("/", response_model=MessageResponse)
async def create_message(
    message: MessageCreate,
    message_log: MessageLog = Depends(get_message_log)
) -> Message:
    try:
        new_message = Message(**message.model_dump())
        return await message_log.append(new_message)
    except MessageLogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: int,
    message_log: MessageLog = Depends(get_message_log)
) -> Message:
    try:
        messages = await message_log.get_messages(limit=1000)  # Get all messages to find the one we need
        message = next((m for m in messages if m.id == message_id), None)
        if message is None:
            raise MessageLogError(f"Message with id {message_id} not found")
        return message
    except MessageLogError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[MessageResponse])
async def list_messages(
    limit: int = 10,
    offset: int = 0,
    session_id: Optional[int] = None,
    message_log: MessageLog = Depends(get_message_log)
) -> List[Message]:
    try:
        return await message_log.get_messages(limit=limit, offset=offset, session_id=session_id)
    except MessageLogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/turn/{turn_id}", response_model=List[MessageResponse])
async def get_messages_in_turn(
    turn_id: int,
    message_log: MessageLog = Depends(get_message_log)
) -> List[Message]:
    try:
        messages = await message_log.get_messages(limit=1000)  # Get all messages
        return [m for m in messages if m.turn_id == turn_id]
    except MessageLogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{message_id}", response_model=MessageResponse)
async def update_message(
    message_id: int,
    message_update: MessageUpdate,
    message_log: MessageLog = Depends(get_message_log)
) -> Message:
    try:
        # First get the message
        messages = await message_log.get_messages(limit=1000)
        message = next((m for m in messages if m.id == message_id), None)
        if message is None:
            raise MessageLogError(f"Message with id {message_id} not found")
        
        # Update the message
        update_data = {k: v for k, v in message_update.model_dump().items() if v is not None}
        for key, value in update_data.items():
            setattr(message, key, value)
        
        await message_log.update(message)
        return message
    except MessageLogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{message_id}", status_code=204)
async def delete_message(
    message_id: int,
    message_log: MessageLog = Depends(get_message_log)
):
    # Note: The current MessageLog class doesn't support message deletion
    # This is a placeholder that returns a "Not Implemented" error
    raise HTTPException(status_code=501, detail="Message deletion is not implemented") 