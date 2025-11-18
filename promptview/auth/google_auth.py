from datetime import datetime, timedelta
import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from authlib.integrations.starlette_client import OAuth

class GoogleAuth:
    def __init__(
        self,
        secret_key: str,
        algorithm: str,
        access_token_expire_minutes: int,
        client_id: str,
        client_secret: str,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.client_id = client_id
        self.client_secret = client_secret

        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=self.client_id,
            client_secret=self.client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )


    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=self.access_token_expire_minutes)
        )
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)


    # async def get_loggedin_user(
    #     request: Request
    # ) -> User:
    #     credentials_exception = HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )

    #     # Try to get token from cookie first
    #     access_token = request.cookies.get("access_token")

    #     # Alternatively, try from Authorization header "Bearer <token>"
    #     if not access_token:
    #         auth: str | None = request.headers.get("Authorization")
    #         if auth and auth.startswith("Bearer "):
    #             access_token = auth.removeprefix("Bearer ").strip()

    #     if not access_token:
    #         raise credentials_exception

    #     try:
    #         payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
    #         user_id: str = payload.get("id")
    #         if user_id is None:
    #             raise credentials_exception
    #     except Exception:
    #         raise credentials_exception

    #     # user = user_interface.get_by_id(user_id, db)
    #     user = await User.get(user_id)
    #     if user is None:
    #         raise credentials_exception

    #     return user

    # async def optional_get_loggedin_user(
    #         self,
    #         request: Request,
    #     ) -> User:
    #         user_id = None
    #         credentials_exception = HTTPException(
    #             status_code=status.HTTP_401_UNAUTHORIZED,
    #             detail="Could not validate credentials",
    #             headers={"WWW-Authenticate": "Bearer"},
    #         )

    #         # Try to get token from cookie first
    #         access_token = request.cookies.get("access_token")

    #         # Alternatively, try from Authorization header "Bearer <token>"
    #         if not access_token:
    #             auth: str | None = request.headers.get("Authorization")
    #             if auth and auth.startswith("Bearer "):
    #                 access_token = auth.removeprefix("Bearer ").strip()

    #         if not access_token:
    #             raise credentials_exception
    #         try:
    #             payload = jwt.decode(access_token, self.secret_key, algorithms=[self.algorithm])
    #             user_id: str = payload.get("id")
    #         except (JWTError, Exception):
    #             pass

    #         user = None
    #         if user_id:
    #             user = await User.get(user_id)
    #             if user is None:
    #                 raise credentials_exception
    #         else:
    #             raise credentials_exception
    #         return user

    # async def optional_get_loggedin_user(
    #     self,
    #     request: Request,
    # ) -> User:
    #     user_id = None
    #     credentials_exception = HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Could not validate credentials",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )

    #     # Try to get token from cookie first
    #     access_token = request.cookies.get("access_token")

    #     # Alternatively, try from Authorization header "Bearer <token>"
    #     if not access_token:
    #         auth: str | None = request.headers.get("Authorization")
    #         if auth and auth.startswith("Bearer "):
    #             access_token = auth.removeprefix("Bearer ").strip()

    #     if not access_token:
    #         raise credentials_exception
    #     try:
    #         payload = jwt.decode(access_token, self.secret_key, algorithms=[self.algorithm])
    #         user_id: str = payload.get("id")
    #     except (JWTError, Exception):
    #         pass

    #     user = None
    #     if user_id:
    #         user = await User.get(user_id)
    #         if user is None:
    #             raise credentials_exception
    #     else:
    #         raise credentials_exception
    #     return user


    def verify_idinfo(self, token: str) -> dict:
        return id_token.verify_oauth2_token(
            token, google_requests.Request(), self.client_id
        )
