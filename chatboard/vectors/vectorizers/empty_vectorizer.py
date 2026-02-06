from typing import List
from pydantic import Field
from ..embeddings.text_embeddings import DenseEmbeddings
from ..vectorizers.base import VectorMetrics, VectorizerBase
import numpy as np


class EmptyVectorizer(VectorizerBase):
    name: str = "dense"
    size: int = 300
    dense_embeddings: DenseEmbeddings = Field(default_factory=DenseEmbeddings)
    metric: VectorMetrics = VectorMetrics.COSINE
    
    async def embed_documents(self, documents: List[str]):
        return np.zeros((len(documents), self.size))
    
    async def embed_query(self, query: str):
        return np.zeros(self.size)