








from fastapi import Request
from pydantic import BaseModel
from typing import Generic, List, Optional, Dict, Any, Set, Type, TypeVar
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
from ..auth.user_manager2 import AuthModel

class ModelCtx(BaseModel):
    branch_id: Optional[int] = None
    turn_id: Optional[int] = None


AUTH_MODEL = TypeVar("AUTH_MODEL", bound=AuthModel)
CTX = TypeVar("CTX", bound=BaseModel)



class CtxRequest:
    request: Request
    
    
    def __init__(self, request: Request):
        self.request = request

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )

    @staticmethod
    def _validate(value: Any) -> "CtxRequest":
        print(f"value: {value}")
        req = value.get("request", None)
        if req is not None and isinstance(req, Request):
            return CtxRequest(request=req)
        else:
            raise TypeError(f"Invalid type for CtxRequest: {type(value)}")

    @staticmethod
    def _serialize(instance: "CtxRequest | None") -> Request | None:
        if instance is None:
            return None
        return instance.request



class Context(BaseModel):    
    _request: CtxRequest | None = None
    
    @property
    def request(self) -> Request:
        if self._request is None:
            raise ValueError("Request is not set")
        return self._request.request
    
    @classmethod
    def get_headers(cls, headers: Set[str], request: Request) -> dict[str, Any]:
        raw = {}
        for key in request.headers.keys():
            if key in headers:
                val = request.headers.get(key)
                if val and val != "null":
                    raw[key] = int(val) if val.isdigit() else val
        for key in headers:
            if key not in raw:
                raise ValueError(f"Header {key} is required")
        return raw
    
    # @classmethod
    # def get_user_manager(cls):
    
    #     return Chatboard.get_app_ctx()._auth_manager()
    

    @classmethod
    async def from_request(cls, request: Request):
        raise NotImplementedError("Context is not implemented")
    
    async def enter_query(self):
        raise NotImplementedError("Context is not implemented")
    
    async def exit_query(self, exc_type, exc_value, traceback):
        raise NotImplementedError("Context is not implemented")
    
    async def enter_mutation(self):
        raise NotImplementedError("Context is not implemented")
    
    async def exit_mutation(self, exc_type, exc_value, traceback):
        raise NotImplementedError("Context is not implemented")
    
    async def __aenter__(self):
        if self.request.method == "GET":
            await self.enter_query()
        else:
            await self.enter_mutation()
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.request.method == "GET":
            await self.exit_query(exc_type, exc_value, traceback)
        else:
            await self.exit_mutation(exc_type, exc_value, traceback)
    
    
    