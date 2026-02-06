
import asyncio
from typing import Any, List

from ..vectorizers.base import (VectorizerBase,
                                                     VectorizerSparseBase,
                                                     VectorMetrics)
from pinecone_text.sparse import BM25Encoder
from pydantic import Field


class BM25Vectorizer(VectorizerSparseBase):
    name: str = "bm25"
    is_sparse: bool = True
    bm25: BM25Encoder = Field(default_factory=BM25Encoder.default)
    metric: VectorMetrics = VectorMetrics.EUCLIDEAN

    # def __init__(self):
        # super().__init__(bm25 = BM25Encoder.default())
    
    async def embed_documents(self, documents: List[str]):
        return await asyncio.to_thread(self.bm25.encode_documents, documents)
    
    async def embed_query(self, query: str):
        return await asyncio.to_thread(self.bm25.encode_queries, [query])
        

