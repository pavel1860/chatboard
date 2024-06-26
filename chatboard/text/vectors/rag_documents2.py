from typing import Any, Dict, List, TypeVar, Union, Optional, Generic, Type
from uuid import uuid4
from chatboard.text.llms.completion_parsing import is_list_model
from pydantic import BaseModel
# from chatboard.text.app_manager import app_manager
from chatboard.text.vectors.stores.base import VectorStoreBase
from chatboard.text.vectors.stores.qdrant_vector_store import QdrantVectorStore
from chatboard.text.vectors.vectorizers.base import VectorMetrics, VectorizerBase
from qdrant_client.http.exceptions import UnexpectedResponse
import asyncio

from chatboard.text.vectors.vectorizers.text_vectorizer import TextVectorizer



K = TypeVar('K', bound=BaseModel)
V = TypeVar('V', bound=BaseModel)



# class RagDocMetadata(Generic[K, V], BaseModel):
#     id: Union[str, int]
#     key: Optional[Union[K, str]]
#     value: Union[V, str]
# class RagDocMetadata:
    
    
    



# class Point(BaseModel):
#     id: str
#     vector: Dict[str, Any]
#     metadata: RagDocMetadata[K, V]

class RagSearchResult(BaseModel):
    id: str
    score: float
    metadata: Any
    vector: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True



class RagDocuments:

    def __init__(
            self, 
            namespace: str, 
            vectorizers: List[VectorizerBase] = [], 
            vector_store: VectorStoreBase | None = None, 
            key_class: Type[K] | Type[str] = str, 
            metadata_class: Type[V] | Type[str] = str
        ) -> None:
        self.namespace = namespace
        if vector_store is None:
            self.vector_store = QdrantVectorStore(namespace)
        else:
            self.vector_store = vector_store
        if not vectorizers:
            self.vectorizers = [TextVectorizer()]
        else:
            self.vectorizers = vectorizers
        self.key_class = key_class
        self.metadata_class = metadata_class
        # app_manager.register_rag_space(namespace, metadata_class)


    async def _embed_documents(self, documents: List[Any]):
        embeds = await asyncio.gather(
            *[vectorizer.embed_documents(documents) for vectorizer in self.vectorizers]
        )
        vectors = []
        embeds_lookup = {vectorizer.name: embed for vectorizer, embed in zip(self.vectorizers, embeds)}
        for i in range(len(documents)):
            vec = {}
            for vectorizer in self.vectorizers:
                vec[vectorizer.name] = embeds_lookup[vectorizer.name][i]
            vectors.append(vec)
        return vectors
    

    def __deepcopy__(self, memo):
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        setattr(new_instance, "vectorizers", self.vectorizers)
        setattr(new_instance, "vector_store", self.vector_store)
        return new_instance
    
    
    def _pack_results(self, results):
        rag_results = []
        for res in results:
            # if self.key_class == str:
            #     key = res.metadata["key"]
            # else:
            #     key = self.key_class(**res.metadata["key"])
            metadata = self.metadata_class(**res.metadata)
            rag_results.append(RagSearchResult(
                id=res.id, 
                score=res.score, 
                metadata=metadata,
                vector=res.vector
            ))
        return rag_results
        # return [RagDocMetadata(id=res.id, key=res.key, value=res.value) for res in results]

    


    async def add_documents(self, keys: List[Any], metadata: List[Any], ids: List[str] | List[int] | None=None):
        if keys is not None and type(keys) != list:
            raise ValueError("keys must be a list")
        if type(metadata) != list:
            raise ValueError("values must be a list")
        if ids is not None and type(ids) != list:
            raise ValueError("ids must be a list")
        
        if self.key_class is not None:
            for i, key in enumerate(keys):
                if type(key) != self.key_class:
                    raise ValueError(f"key at index {i} is not of type {self.key_class}")
        for i, value in enumerate(metadata):
            # handling list models
            if is_list_model(self.metadata_class):
                if type(value) != list:
                    raise ValueError(f"value at index {i} is not a list")
            elif type(value) != self.metadata_class:
                raise ValueError(f"value at index {i} is not of type {self.metadata_class}")
        
        vectors = await self._embed_documents(keys)
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(keys))]
            
        # documents = [RagDocMetadata[self.key_class, self.value_class](id=i, key=key if include_key else None, value=value) for i, key, value in zip(ids, keys, values)]
        # documents = [self.metadata_class(id=i, **value.dict()) for i, value in zip(ids, metadata)]
        outputs = await self.vector_store.add_documents(vectors, metadata, ids=ids, namespace=self.namespace)
        return outputs
    

    async def update_documents(self, metadata: Dict, ids: List[int | str] | None=None, filters: Any=None):        
        res = await self.vector_store.update_documents(metadata, ids=ids, filters=filters)
        return res
        

    

    async def get_documents(
            self, 
            filters: Any=None, 
            ids: List[int | str] | None = None, 
            top_k=10, 
            with_metadata: bool=True, 
            with_vectors: bool=False, 
            order_by: Dict | None=None, 
            group_by: Dict | None=None,
            group_size=1,
            ):
        res = await self.vector_store.get_documents(
            filters=filters, 
            ids=ids, 
            top_k=top_k,
            with_payload=with_metadata, 
            with_vectors=with_vectors,
            order_by=order_by,
        )
        return self._pack_results(res)
    

    async def add_examples(self, examples: List[RagSearchResult]):
        keys = [example.metadata.key for example in examples]
        values = [example.metadata.value for example in examples]
        ids = [example.id for example in examples]
        return await self.add_documents(keys, values, ids)
    


    async def similarity(self, query: Any, top_k=3, filters=None, alpha=None, with_vectors=False):
        query_vector = await self._embed_documents([query])
        query_vector = query_vector[0]
        res = await self.vector_store.similarity(query_vector, top_k, filters, alpha, with_vectors=with_vectors)
        return self._pack_results(res)
    

    async def similar_example(self, exmpl: RagSearchResult, top_k=3, filters=None, alpha=None, with_vectors=False):
        query_vector = exmpl.vector
        res = await self.vector_store.similarity(query_vector, top_k, filters, alpha, with_vectors=with_vectors)
        return self._pack_results(res)
    

    async def get_many(self, top_k=10):
        res = await self.vector_store.get_many(top_k=top_k, with_payload=True)
        return self._pack_results(res)
    

    async def create_namespace(self, namespace: str | None = None):
        namespace = namespace or self.namespace
        indexs_to_create=[]
        for field, info in self.metadata_class.__fields__.items():
            if hasattr(info, 'field_info'): # check if pydantic v1
                extra = info.field_info.extra
                if "index" in extra:
                    print(field, extra["index"])
                    indexs_to_create.append({
                        "field": field,
                        "schema": extra["index"]
                    })
        return await self.vector_store.create_collection(self.vectorizers, namespace, indexs=indexs_to_create)
    
    async def verify_namespace(self):
        try:
            await self.vector_store.info()
        except UnexpectedResponse as e:
            await self.create_namespace()
    

    async def delete_namespace(self, namespace: str | None = None):
        namespace = namespace or self.namespace
        return await self.vector_store.delete_collection(namespace)
    

    async def delete_documents(self, ids: List[int | str] | None = None, filters: Any=None):
        if ids is not None:
            return await self.vector_store.delete_documents_ids(ids)
        elif filters is not None:
            return await self.vector_store.delete_documents(filters)