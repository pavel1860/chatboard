import asyncio
from typing import ClassVar
import openai
import numpy as np
from pydantic import BaseModel, Field
from pinecone_text.sparse import BM25Encoder
from .base_vectorizer import BaseVectorizer, VectorType



class BM25Vectorizer(BaseVectorizer[str, list[float]]):
    client: BM25Encoder = Field(default_factory=BM25Encoder.default)
    model: str = "bm25"
    type: VectorType = "sparse"
    dimension: ClassVar[int] = 30000
    
    
    async def embed_documents(self, docs: list[str]) -> list[np.ndarray]:
        """
        Embed the provided documents using OpenAI's embedding API.
        
        Args:
            docs (list[str]): A list of strings to embed.
        
        Returns:
            list[list[float]]: A list of embeddings, each embedding being a list of floats.
        """
        if not docs:
            raise ValueError("No documents provided for embedding.")
        
        # Use OpenAI API to get embeddings
        try:
            return await asyncio.to_thread(self.client.encode_documents, docs)
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            raise RuntimeError(f"Error while generating embeddings: {e}")
        
        
        
    async def embed_query(self, query: str):
        return await asyncio.to_thread(self.client.encode_queries, [query])