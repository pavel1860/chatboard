from typing import ClassVar
import openai
import numpy as np
from pydantic import BaseModel, Field
from .base_vectorizer import BaseVectorizer, VectorType




class OpenAIVectorizer(BaseVectorizer[str, np.ndarray]):
    client: openai.AsyncClient = Field(default_factory=openai.AsyncClient)
    model: str
    type: VectorType = "dense"
    
    async def embed_documents(self, docs: list[str]) -> list[np.ndarray]:
        """
        Embed the provided documents using OpenAI's embedding API.
        
        Args:
            docs (list[str]): A list of strings to embed.
        
        Returns:
            list[np.ndarray]: A list of embeddings, each embedding being a list of floats.
        """
        if not docs:
            raise ValueError("No documents provided for embedding.")
        
        # Use OpenAI API to get embeddings
        try:
            response = await self.client.embeddings.create(input=docs, model=self.model)
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            raise RuntimeError(f"Error while generating embeddings: {e}")
    
    


class OpenAISmallVectorizer(OpenAIVectorizer):
    dimension: ClassVar[int] = 1536
    model: str = Field(default="text-embedding-3-small")
    


class OpenAILargeVectorizer(OpenAIVectorizer):
    dimension: ClassVar[int] = 3072
    model: str = Field(default="text-embedding-3-large")
