from typing import ClassVar
from pydantic import BaseModel, Field
import numpy as np
import asyncio

from .base_vectorizer import BaseVectorizer, VectorType

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence_transformers is not installed. Please install it with `pip install sentence-transformers`")

class SentenceTransformersVectorizer(BaseVectorizer[str, np.ndarray]):
    model: str
    type: VectorType = "dense"
    _model_inst: SentenceTransformer

    def __init__(self, **data):
        super().__init__(**data)        
        self._model_inst = SentenceTransformer(self.model)

    async def embed_documents(self, docs: list[str]) -> list[np.ndarray]:
        """
        Asynchronously embed the provided documents using MiniLM-L6-v2 model.

        Args:
            docs (List[str]): A list of strings to embed.

        Returns:
            List[np.ndarray]: A list of embeddings, each being a NumPy array.
        """
        # Use asyncio.to_thread to run the synchronous encoding in a separate thread
        embeddings = await asyncio.to_thread(self._model_inst.encode, docs)
        return [e for e in embeddings]        


class MiniLmL6Vectorizer(SentenceTransformersVectorizer):
    dimension: ClassVar[int] = 384
    model: str = Field(default="all-MiniLM-L6-v2")
    