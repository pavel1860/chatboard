from promptview.algebra.vectors.base_vectorizer import BaseVectorizer, VectorType
from typing import ClassVar, List
import numpy as np
# from scipy.sparse import csr_matrix

class TestVectorizer(BaseVectorizer[str, np.ndarray]):
    model: str = "test"
    type: VectorType = "dense"
    dimension: ClassVar[int] = 4
    
    async def embed_documents(self, docs: list[str]) -> List[np.ndarray]:
        return [np.random.rand(self.dimension) for _ in range(len(docs))]
    
    async def embed_query(self, query: str) -> np.ndarray:
        return np.random.rand(self.dimension)



# class TestSparseVectorizer(BaseVectorizer[str, csr_matrix]):
#     model: str = "test_sparse"
#     type: VectorType = "sparse"
#     dimension: ClassVar[int] = 4
    
#     async def embed_documents(self, docs: list[str]) -> List[csr_matrix]:
#         return [csr_matrix(np.random.rand(self.dimension)) for _ in range(len(docs))]
    
#     async def embed_query(self, query: str) -> csr_matrix:
#         return csr_matrix(np.random.rand(self.dimension))
    