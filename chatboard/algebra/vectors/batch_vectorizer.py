import asyncio
from typing import Any


from .base_vectorizer import BaseVectorizer




class BatchVectorizer:
    
    def __init__(self, vectorizers: dict[str, BaseVectorizer] | None = None):
        self.vectorizers = vectorizers or {}
        
    def add_vectorizer(self, name: str, vectorizer: BaseVectorizer):
        self.vectorizers[name] = vectorizer  
        
    def _pack_vectors(self, vectors: list[Any]) -> dict[str, Any]:
        return {name: vectors[i] for i, name in enumerate(self.vectorizers.keys())}

    async def embed_query(self, query: str) -> dict[str, Any]:
        vector_data = await asyncio.gather(*[vectorizer.embed_query(query) for vectorizer in self.vectorizers.values()])
        return self._pack_vectors(vector_data)
    
    async def embed_documents(self, docs: list[str]) -> list[dict[str, Any]]:
        vector_data = await asyncio.gather(*[vectorizer.embed_documents(docs) for vectorizer in self.vectorizers.values()])
        return [self._pack_vectors(vector_data[i]) for i in range(len(vector_data))]


