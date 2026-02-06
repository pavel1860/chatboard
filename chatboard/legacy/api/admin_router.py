from fastapi import APIRouter
from typing import Type


# def build_admin_router(*, user_model_cls: Type[User], backend: UserBackend) -> APIRouter:
#     admin_router = APIRouter(prefix="/api")

#     # Create user router with the specific user model
#     user_router = create_user_router(user_model_cls, backend)

#     # # Include all routers
#     # admin_router.include_router(
#     #     message_router,
#     #     tags=["messages"],
#     # )

#     # admin_router.include_router(
#     #     session_router,
#     #     tags=["sessions"],
#     # )

#     # admin_router.include_router(
#     #     user_router,
#     #     tags=["users"],
#     # )

#     return admin_router


admin_router = APIRouter(prefix="/api")


