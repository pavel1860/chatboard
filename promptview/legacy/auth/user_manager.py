from typing import Generic, List, Optional, Type, final
from datetime import datetime
from uuid import UUID, uuid4
from fastapi import HTTPException, Request
from typing_extensions import TypeVar


from pydantic import BaseModel
from promptview.model import Model, ModelField
from promptview.model.fields import KeyField
from promptview.utils.db_connections import PGConnectionManager


# class UserModel(Model, HeadModel):
#     name: str | None = ModelField(None)
#     email: str = ModelField(...)
#     image: str | None = ModelField(None)
#     emailVerified: datetime = ModelField(..., db_type="TIMESTAMPTZ")
#     is_admin: bool = ModelField(default=False)
#     # user_auth_id: int = ModelField(...)
    
#     class Config: # do not fix this!
#         database_type="postgres"
#         is_head=True
#         is_abstract=True
#         namespace="users"

#     async def check_out_head(self, head_id: int):
#         head = await PGConnectionManager.fetch_one(
#             f"""
#             UPDATE heads AS uh
#             SET 
#                 main_branch_id = th.main_branch_id,
#                 branch_id      = th.branch_id,
#                 turn_id        = th.turn_id,
#                 updated_at     = NOW()
#             FROM heads AS th
#             WHERE uh.id = {self.head.id}
#             AND th.id = {head_id}
#             RETURNING uh.*;
#             """
#         )
#         if head is None:
#             raise UserManagerError("Invalid head id")
#         return dict(head)


class AuthModel(Model):
    _is_base: bool = True
    id: int = KeyField(primary_key=True)
    anonymous_token: UUID | None = ModelField(None)
    name: str | None = ModelField(None)
    email: str | None = ModelField(None)
    password: str | None = ModelField(None)
    image: str | None = ModelField(None)
    emailVerified: datetime | None = ModelField(None, db_type="TIMESTAMPTZ")
    is_admin: bool = ModelField(default=False)
    created_at: datetime = ModelField(default_factory=datetime.now)
    
    
    @property
    def is_anonymous(self) -> bool:
        return self.anonymous_token is not None and self.email is None
    
    
    

class UserAuthPayload(BaseModel):
    name: str | None = None
    anonymous_token: UUID | None = None
    email: str | None = None
    emailVerified: datetime | None = None
    image: str | None = None
    password: str | None = None
    is_admin: bool = False
    user_id: str | None = None
    
class UserAuthUpdatePayload(UserAuthPayload):
    id: int
    
class UserManagerError(Exception):
    pass


AUTH_MODEL = TypeVar("AUTH_MODEL", bound=AuthModel)

class AuthManager(Generic[AUTH_MODEL]):
    
    _auth_model: Optional[Type[AUTH_MODEL]] = None
    _add_temp_users: bool = False
    
    
    
    @classmethod
    def get_user_manager(cls):
        return cls()
    
    
    @classmethod
    def get_user_model(cls) -> Type[AUTH_MODEL]:
        if cls._auth_model is None:
            raise ValueError("User model not registered")
        return cls._auth_model
    
    @classmethod
    def register_user_model(cls, user_model: Type[AUTH_MODEL], add_temp_users: bool = False):
        cls._auth_model = user_model
        cls._add_temp_users = add_temp_users
    
    @staticmethod
    async def initialize_tables():

        await PGConnectionManager.execute(
            """
CREATE TABLE IF NOT EXISTS verification_token (
    identifier TEXT NOT NULL,
    expires TIMESTAMPTZ NOT NULL,
    token TEXT NOT NULL,
    
    PRIMARY KEY (identifier, token)
);

CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL,
    "userId" INTEGER NOT NULL,
    type VARCHAR(255) NOT NULL,
    provider VARCHAR(255) NOT NULL,
    "providerAccountId" VARCHAR(255) NOT NULL,
    refresh_token TEXT,
    access_token TEXT,
    expires_at BIGINT,
    id_token TEXT,
    scope TEXT,
    session_state TEXT,
    token_type TEXT,
    
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL,
    "userId" INTEGER NOT NULL,
    expires TIMESTAMPTZ NOT NULL,
    "sessionToken" VARCHAR(255) NOT NULL,
    
    PRIMARY KEY (id)
);                                                     
""")
        
        
 
    @classmethod    
    async def drop_tables(cls):
        await PGConnectionManager.execute(
            """
            DROP TABLE IF EXISTS verification_token;
            DROP TABLE IF EXISTS accounts;
            DROP TABLE IF EXISTS sessions;           
        """
        )
        
    
    async def after_create_user(self, user: AUTH_MODEL, request: Request):
        return user
    
    async def before_create_user(self, data: UserAuthPayload, request: Request) -> UserAuthPayload:
        return data
    
    
    @final
    async def create_user(self, data: UserAuthPayload, request: Request):
        data = await self.before_create_user(data, request)
        user_model = self.get_user_model()
        user = await user_model(**data.model_dump()).save()
        user = await self.after_create_user(user, request)
        return user
    
    @final
    async def create_user_from_anonymous(self, anonymous_token: UUID, data: UserAuthPayload, request: Request):
        user = await self.get_by_anonymous_token(request, anonymous_token)
        if user is None:
            return None
            raise HTTPException(status_code=401, detail="Unauthorized")
        return await user.update(**data.model_dump(exclude={"id", "anonymous_token"}))
    
    @final
    async def update_user(self, id: int, data: UserAuthPayload, request: Request):
        user_model = self.get_user_model()
        user = await user_model.get(id)
        await user.update(**data.model_dump(exclude={"id"}))
        return user

    
    
    async def get_by_id(self, id: int):
        user_model = self.get_user_model()
        user = await user_model.get(id)
        return user
    
    
    
    async def get_by_email(self, email: str):
        user_model = self.get_user_model()
        user = await user_model.query().filter(lambda x: x.email == email).first()
        return user
    
    
    async def on_session_token_user_not_found(self, request: Request, session_token: str):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    async def get_by_session_token(self, request: Request, session_token: str, use_sessions: bool = True):
        if use_sessions:
            res = await PGConnectionManager.fetch_one(
                f"""
                SELECT "userId" FROM sessions WHERE "sessionToken" = '{session_token}'
                """
            )
            if res is None:
                raise UserManagerError("Invalid session token")
            return await self.get_by_id(res["userId"])
        else:
            res = await PGConnectionManager.fetch_one(
                f"""
                SELECT "id" FROM users WHERE user_token = '{session_token}'
                """
            )
            if res is None:
                return None
            return await self.get_by_id(res["id"])
        
        
    async def get_by_anonymous_token(self, request: Request, anonymous_token: UUID):
        user_model = self.get_user_model()
        user = await user_model.query().filter(lambda x: x.anonymous_token == anonymous_token).first()
        return user
    
    
    async def on_missing_anonymous_token(self, request: Request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    async def on_missing_session_token(self, request: Request, anonymous_token: UUID | None = None):
        if anonymous_token is None:
            await self.on_missing_anonymous_token(request)
        else:
            user = await self.get_by_anonymous_token(request, anonymous_token)
            return user
    
    async def on_anonymous_user_not_found(self, request: Request, anonymous_token: UUID):
        return await self.create_user(UserAuthPayload(anonymous_token=anonymous_token), request)
    
    
    async def get_anonymous_user(self, request: Request, create_if_not_found: bool = False):
        anonymous_token = request.cookies.get("chatboard.anonymous-token")
        if anonymous_token is None:            
            await self.on_missing_anonymous_token(request)
        else:
            anonymous_token = UUID(anonymous_token)
            user = await self.get_by_anonymous_token(request, anonymous_token)
            if user is None:
                if not create_if_not_found:
                    return None
                user = await self.on_anonymous_user_not_found(request, anonymous_token)
            return user
    
    
    @classmethod
    async def get_session_user(cls, request: Request):
        user_manager = cls.get_user_manager()
        session_token = request.cookies.get("next-auth.session-token")
        ref_user_id = request.headers.get("x-ref-user-id")
        if not session_token:
            return None
        if user_id := request.headers.get("user_id"):
            user = await user_manager.get_by_id(int(user_id))
            if user is not None:
                if ref_user_id:
                    if not user.is_admin:
                        raise HTTPException(status_code=401, detail="Unauthorized")
                    user = await user_manager.get_by_id(int(ref_user_id))
            return user
        user = await user_manager.get_by_session_token(request, session_token)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user
    
    
    @classmethod
    async def get_user_or_create(cls, request: Request):
        user_manager = cls.get_user_manager()
        session_token = request.cookies.get("next-auth.session-token")
        if not session_token:
            return await user_manager.get_anonymous_user(request)
        else:
            user = await user_manager.get_by_session_token(request, session_token)
            return user
    
    @classmethod
    async def get_user_from_request2(cls, request: Request):
        user_model = cls.get_user_model()
        user_manager = cls.get_user_manager()
        session_token = request.cookies.get("next-auth.session-token")
        if not session_token:
            anonymous_token = request.cookies.get("chatboard.anonymous-token")
            if anonymous_token is None:
                raise HTTPException(status_code=401, detail="Unauthorized")            
            user = await user_model.query().where(user_token=anonymous_token).first()
            if user is None:
                user = await cls.create_new_user(anonymous_token)
            return user
        else:
            user_manager = cls.get_user_manager()
            user = await user_manager.get_user_by_session_token(session_token)
            return user
    
    
    
    async def delete_user(self):
        pass
    

    async def get_users(self):
        user_model = self.get_user_model()
        users = await user_model.query().filter(lambda x: x.is_admin == False)
        return users
    
    




