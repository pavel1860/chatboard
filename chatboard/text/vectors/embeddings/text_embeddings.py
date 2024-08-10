import os
from typing import Any, List

# from langchain.schema.embeddings import Embeddings
import langchain.schema.embeddings as lang_embeddings
from chatboard.clients.openai_client import (
    build_async_openai_client, build_async_openai_embeddings_client)
from langchain.pydantic_v1 import BaseModel
from openai import AsyncOpenAI
from pinecone_text.dense import OpenAIEncoder
from pinecone_text.sparse import BM25Encoder

openai_model = os.getenv("OPENAI_MODEL", 'text-embedding-ada-002')


class Embeddings(BaseModel):
    dense: Any
    sparse: Any
    


    def avg(self, alpha):
        if alpha < 0 or alpha > 1:
            raise Exception('alpha must be between 0 and 1')
        dense_embeddings = [v * alpha for v in self.dense]
        sparse_embeddings = {
            'indices': self.sparse['indices'],
            'values': [v * (1- alpha) for v in self.sparse['values']]
        } if self.sparse else None
        return Embeddings(
            dense=dense_embeddings,
            sparse=sparse_embeddings
        )
    


    


class HybridAdaMB25Embeddings(BaseModel, lang_embeddings.Embeddings):

    sparse_embeddings: Any
    dense_embeddings: Any

    def __init__(self):
        super().__init__()
        # self.sparse_embeddings = BM25Encoder.default()
        self.sparse_embeddings = None
        # self.dense_embeddings = OpenAIEmbeddings(model=openai_model)
        self.dense_embeddings = OpenAIEncoder()


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if type(texts) == str:
            texts = [texts]
        sparse_embeddings = self.sparse_embeddings.encode_documents(texts) if self.sparse_embeddings else None
        dense_embeddings = self.dense_embeddings.encode_documents(texts)
        return Embeddings(
            dense=dense_embeddings,
            sparse=sparse_embeddings
        )
        # return {
        #     'sparse_embeddings': sparse_embeddings, 
        #     'dense_embeddings': dense_embeddings
        # }

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        sparse_embeddings = self.sparse_embeddings.encode_queries(text)
        dense_embeddings = self.dense_embeddings.encode_queries(text)
        return Embeddings(
            dense=dense_embeddings,
            sparse=sparse_embeddings
        )
        # return {
        #     'sparse_embeddings': sparse_embeddings, 
        #     'dense_embeddings': dense_embeddings
        # }

    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Asynchronous Embed search docs."""
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, self.embed_documents, texts
    #     )

    # async def aembed_query(self, text: str) -> List[float]:
    #     """Asynchronous Embed query text."""
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, self.embed_query, text
    #     )





# class DenseEmbeddings(BaseModel, lang_embeddings.Embeddings):

#     sparse_embeddings: Any
#     dense_embeddings: Any

#     def __init__(self):
#         super().__init__()
#         self.dense_embeddings = OpenAIEncoder()

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         dense_embeddings = self.dense_embeddings.encode_documents(texts)
#         return [Embeddings(dense=dense_emb, sparse=None) for dense_emb in dense_embeddings]

#     def embed_query(self, text: str) -> List[float]:
#         """Embed query text."""
#         dense_embeddings = self.dense_embeddings.encode_queries(text)
#         return Embeddings(dense=dense_embeddings, sparse=None)


class DenseEmbeddings:

    def __init__(self):
        super().__init__()
        self.client = build_async_openai_embeddings_client()

    async def embed_documents(self, texts: List[str], model="text-embedding-3-small"):
        try:
            res = await self.client.embeddings.create(input=texts, model=model)
            return [embs.embedding for embs in res.data]
        except Exception as e:
            print("bad texts:", texts)
            raise e

    async def embed_query(self, text: str, model="text-embedding-3-small"):
        try:
            res = await self.client.embeddings.create(input=[text], model=model)
            return res.data[0].embedding
        except Exception as e:
            print(text)
            raise e