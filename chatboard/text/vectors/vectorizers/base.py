
from abc import abstractmethod
from typing import Any, List

from pydantic import BaseModel


class VectorMetrics:
    """Supported metrics for vector similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"





class VectorizerBase(BaseModel):
    """Base class for vectorizers"""
    name: str
    # size: int
    metric: VectorMetrics
    # is_sparse: bool = False

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def embed_documents(self, documents: List[Any]):
        """Embeds a list of documents into a vector space"""
        raise NotImplementedError
    
    @abstractmethod
    async def embed_query(self, query: Any):
        """Embeds a query into a vector space"""
        raise NotImplementedError




class VectorizerDenseBase(VectorizerBase):
    """Base class for vectorizers"""    
    size: int



class VectorizerSparseBase(VectorizerBase):
    pass