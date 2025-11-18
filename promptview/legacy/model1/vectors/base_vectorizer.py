from typing import Literal, TypeVar, Generic
from pydantic import BaseModel, Field


DOC = TypeVar('DOC')
VEC = TypeVar('VEC')

VectorType = Literal["dense", "sparse"]

class BaseVectorizer(BaseModel, Generic[DOC, VEC]):    
    size: int
    type: VectorType
    
    class Config:
        arbitrary_types_allowed = True
    
    async def embed_documents(self, doc: list[DOC]) -> list[VEC]:
        raise NotImplementedError()
    
    async def embed_query(self, query: DOC) -> VEC:
        embeddings = await self.embed_documents([query])
        if embeddings is None:
            raise RuntimeError("Error while generating embeddings")
        return embeddings[0]




# class BaseSparseVectorizer(BaseVectorizer[DOC, VEC]):
#     pass  
# SPARSE_DOC = TypeVar('SPARSE_DOC')
# SPARSE_VEC = TypeVar('SPARSE_VEC')
    
# class BaseSparseVectorizer(BaseModel, Generic[SPARSE_DOC, SPARSE_VEC]):
    
#     class Config:
#         arbitrary_types_allowed = True
    