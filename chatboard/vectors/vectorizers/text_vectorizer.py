from typing import Any, List

from ..embeddings.text_embeddings import DenseEmbeddings
from ..tokenizer import Tokenizer
from ..vectorizers.base import (VectorizerBase,
                                                     VectorizerDenseBase,
                                                     VectorMetrics)
from pydantic import BaseModel, Field




def trim_and_stringify(doc):
    if isinstance(doc, str):
        return doc[:43000]
    return str(doc)[:43000]


def varify_type(doc):
    if isinstance(doc, str):
        return doc
    return str(doc)





class TextVectorizer(VectorizerDenseBase):
    name: str = "dense"
    size: int = 1536
    dense_embeddings: DenseEmbeddings = Field(default_factory=DenseEmbeddings)
    metric: VectorMetrics = VectorMetrics.COSINE
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)
    
    async def embed_documents(self, documents: List[str]):
        documents = self.tokenizer.trim_texts([varify_type(d) for d in documents])
        return await self.dense_embeddings.embed_documents(documents)
    
    async def embed_query(self, query: str):
        query = self.tokenizer.trim_text(varify_type(query))
        return await self.dense_embeddings.embed_query(query)