from typing import Any, Dict, Generic, List, Optional, Type, final
from datetime import datetime
from uuid import UUID, uuid4
import os
from fastapi import HTTPException, Request
from typing_extensions import TypeVar


from pydantic import BaseModel

from promptview.auth.crypto import decode_nextauth_session_token
from .google_auth import GoogleAuth
from ..model import Model, ModelField, RelationField, Branch, KeyField
from uuid import UUID



# class AuthBranch(Model):
#     id: int = KeyField(primary_key=True)
#     branch_id: int = ModelField(foreign_key=True, foreign_cls=Branch)
#     user_id: UUID = ModelField(foreign_key=True, enforce_foreign_key=False)
    
    

class AuthModel(Model):
    _is_base: bool = True
    id: UUID = KeyField(primary_key=True)
    auth_user_id: str | None = ModelField(None, index="btree")
    is_guest: bool = ModelField(default=True)
    guest_token: UUID | None = ModelField(None)
    is_admin: bool = ModelField(default=False)
    created_at: datetime = ModelField(default_factory=datetime.now, order_by=True)
    # branches: List[Branch] = RelationField("Branch", foreign_key="user_id")
    # branches: List[Branch] = RelationField(
    #     primary_key="id",
    #     junction_keys=["user_id", "branch_id"],        
    #     foreign_key="id",
    #     junction_model=AuthBranch,        
    # )



    
    
    
class UserNotFound(Exception):
    pass

class UserAlreadyExists(Exception):
    pass


class UserNotAuthorized(Exception):
    pass
    



UserT = TypeVar("UserT", bound=AuthModel)

class AuthManager(Generic[UserT]):
    user_model: Type[UserT]

    def __init__(self, user_model: Type[UserT], providers: list[GoogleAuth]):            
        self.user_model = user_model
        self.providers = {
            "google": providers[0]
        }
        
    # --------- Hooks ----------
    async def before_create_guest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    async def after_create_guest(self, user: UserT) -> UserT:
        return user
    async def before_register_user(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    async def after_register_user(self, user: UserT) -> UserT:
        return user
    async def before_promote_guest(self, guest: UserT, data: Dict[str, Any]) -> Dict[str, Any]:
        return data
    async def after_promote_guest(self, user: UserT) -> UserT:
        return user
    async def before_fetch_user(self, identifier: Any) -> Any:
        return identifier
    async def after_fetch_user(self, user: Optional[UserT]) -> Optional[UserT]:
        return user

    # --------- Core Logic ----------
    async def create_guest(self, data: Dict[str, Any]) -> UserT:
        data = await self.before_create_guest(data)
        user = await self.user_model(
            is_guest=True,
            guest_token=str(uuid4()),  # fill in UUID creation
            created_at=datetime.utcnow(),
            **data
        ).save()
        # Save to DB here
        user = await self.after_create_guest(user)
        return user

    async def register_user(self, auth_user_id: str, data: Dict[str, Any]) -> UserT:
        data = await self.before_register_user(data)
        if await self.fetch_by_auth_user_id(auth_user_id):
            raise UserAlreadyExists(f"User {auth_user_id} already exists")
        user = await self.user_model(
            is_guest=False,            
            auth_user_id=auth_user_id,
            **data
        ).save()
        # Save to DB here
        user = await self.after_register_user(user)
        return user

    async def promote_guest(self, guest: UserT, auth_user_id: str, data: Dict[str, Any]) -> UserT:
        data = await self.before_promote_guest(guest, data)
        user = await self.fetch_by_auth_user_id(auth_user_id)
        if user:
            raise HTTPException(400, detail="User already exists")
        guest = await guest.update(
            **data, 
            is_guest=False, 
            guest_token=None,
            auth_user_id=auth_user_id
        )
        # Update in DB
        guest = await self.after_promote_guest(guest)
        return guest

    # --------- Fetch Logic with Dependency Pattern ----------
    async def fetch_by_auth_user_id(self, identifier: Any) -> Optional[UserT]:
        identifier = await self.before_fetch_user(identifier)
        user = await self.user_model.query().where(auth_user_id=identifier).last()
        user = await self.after_fetch_user(user)
        return user
    
    
    async def fetch_by_guest_token(self, identifier: Any) -> Optional[UserT]:
        identifier = await self.before_fetch_user(identifier)
        user = await self.user_model.query().where(guest_token=identifier).last()
        user = await self.after_fetch_user(user)
        return user
    
    async def get_user_from_request(self, request: Request) -> Optional[UserT]:
        guest_token = request.cookies.get("temp_user_token")
        session_token = request.cookies.get("next-auth.session-token")
        # if not session_token:
        #     session_token = request.headers.get("X-Backend-Jwt")
        
        #TODO hack for streaming to work. remove this hack
        # if not session_token:
        #     user_id = request.headers.get("X-Auth-User")
        #     user = await self.fetch_by_auth_user_id(user_id)
        #     if not user:
        #         raise UserNotFound(f"User {user_id} not found")
        #     return user
        auth_user_id = None
        if session_token:
            NEXTAUTH_SECRET = os.getenv("NEXTAUTH_SECRET")
            session = decode_nextauth_session_token(session_token, NEXTAUTH_SECRET)
            print("session", session)
            auth_user_id = session.get("user", {}).get("auth_user_id")
        # auth_user_id = request.headers.get("X-Auth-User")
        user = None
        if auth_user_id:
            user = await self.fetch_by_auth_user_id(auth_user_id)
            if not user:
                raise UserNotFound(f"User {auth_user_id} not found")
        elif guest_token:
            user = await self.fetch_by_guest_token(guest_token)
            if not user:
                raise UserNotFound(f"Guest User {guest_token} not found")
        
        return user
    
    async def token_exchange(self, token: str) -> dict:
        try:
            # Verify the Google token
            idinfo = self.providers["google"].verify_idinfo(token=token)
            # idinfo has: email, name, picture...
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Create or fetch user
        # user, created = user_interface.get_or_create_user(user_info=idinfo, db=db)
        user = await self.fetch_by_auth_user_id(idinfo['sub'])
        
        if not user:
            # user = await self.register_user(idinfo['sub'], {"email": idinfo['email'], "name": idinfo['name'], "picture": idinfo['picture']})
            user = await self.register_user(idinfo['sub'], idinfo)
            # raise HTTPException(status_code=401, detail="User not found")

        # Generate JWT
        jwt_token = self.providers["google"].create_access_token(data={"sub": user.auth_user_id})

        return {
            "access_token": jwt_token,
            "token_type": "bearer",
            "created": user.created_at,
            "user": user,
        }


    def get_user(self):
        async def _get_user_dep(request: Request):
            # Try to fetch user from request/session/cookie/etc.
            # Example:
            guest_token = request.cookies.get("temp_user_token")
            auth_user_id = request.headers.get("X-Auth-User")
            user = None
            if auth_user_id:
                user = await self.fetch_by_auth_user_id(auth_user_id)
            elif guest_token:
                user = await self.fetch_by_guest_token(guest_token)
            if not user:
                raise HTTPException(401, detail="User not found")
            return user
        return _get_user_dep