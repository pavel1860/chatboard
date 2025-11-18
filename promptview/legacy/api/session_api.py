from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import datetime as dt

from promptview.conversation.message_log import SessionManager
from promptview.conversation.models import Session

# Pydantic models for response
class SessionResponse(BaseModel):
    id: int
    created_at: dt.datetime
    updated_at: dt.datetime
    user_id: int

    class Config:
        from_attributes = True

# Router setup
router = APIRouter(prefix="/sessions", tags=["sessions"])

# Dependency for SessionManager instance
async def get_session_manager() -> SessionManager:
    return SessionManager()

# @router.get("/", response_model=List[SessionResponse])
# async def list_all_sessions(
#     limit: int = 10,
#     offset: int = 0,
#     session_manager: SessionManager = Depends(get_session_manager)
# ) -> List[Session]:
#     try:
#         return await session_manager.list_sessions(user_id=None, limit=limit, offset=offset)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    user_id: int | None = None,
    limit: int = 10,
    offset: int = 0,
    session_manager: SessionManager = Depends(get_session_manager)
) -> List[Session]:
    try:
        sessions = await session_manager.list_sessions(user_id=user_id, limit=limit, offset=offset)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Session:
    try:
        session = await session_manager.get_session(session_id=session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session with id {session_id} not found")
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}", response_model=List[SessionResponse])
async def list_user_sessions(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    session_manager: SessionManager = Depends(get_session_manager)
) -> List[Session]:
    try:
        return await session_manager.list_sessions(user_id=user_id, limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/last", response_model=SessionResponse)
async def get_last_user_session(
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Session:
    try:
        session = await session_manager.get_last_user_session(user_id=user_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"No sessions found for user {user_id}")
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/user/{user_id}", response_model=SessionResponse)
async def create_session(
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
) -> Session:
    try:
        return await session_manager.create_session(user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 