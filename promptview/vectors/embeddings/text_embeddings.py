import os
from typing import Any, List
from pydantic import BaseModel
from ...llms.utils.openai_clients import build_async_openai_embeddings_client

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