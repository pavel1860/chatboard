import numpy as np
from pydantic import BaseModel, Field
from .base_vectorizer import BaseVectorizer, VectorType




class EmptyVectorizer(BaseVectorizer[str, list[float]]):
    model: str = "empty"
    type: VectorType = "dense"
    size: int = 300
    
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
            return [np.zeros(self.size) for _ in range(len(docs))]
        except Exception as e:
            raise RuntimeError(f"Error while generating embeddings: {e}")
    
