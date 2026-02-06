
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .user_manager import  AuthManager, AuthModel, UserAuthPayload
import os



BACKEND_SECRET = os.getenv("BACKEND_SECRET")

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # if credentials.credentials != BACKEND_SECRET:
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Invalid API key"
    #     )
    if credentials.credentials is None:
        raise HTTPException(
            status_code=401,
            detail="No API key found"
        )
    return credentials.credentials



async def get_user_token(request: Request):
    session_token = request.cookies.get("next-auth.session-token")    
    # if not session_token:                
    #     raise HTTPException(
    #         status_code=401,
    #         detail="No session token found"
    #     )
    return session_token
    

    
    
def get_user_manager():
    from chatboard.legacy.app import Chatboard
    return Chatboard.get_app_ctx()._auth_manager()


async def get_auth_user(
        request: Request, 
        user_token: str = Depends(get_user_token), 
        _: str = Depends(verify_api_key), 
        user_manager: AuthManager = Depends(get_user_manager)
    ):
    if not user_token:
        chatboard_token = request.cookies.get("chatboard.anonymous-token")
        if not chatboard_token:
            raise HTTPException(
                status_code=401,
                detail="No user token found"
            )
        user =  await user_manager.get_by_session_token(chatboard_token, use_sessions=False)
        if not user:
            user = await user_manager.create_user(UserAuthPayload(anonymous_token=chatboard_token, email=chatboard_token))            
    else:
        user =  await user_manager.get_by_session_token(user_token)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

async def get_auth_admin_user(user: AuthModel = Depends(get_auth_user)):
    if not user.is_admin:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user
