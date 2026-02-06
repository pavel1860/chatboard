import copy
from abc import abstractmethod
from typing import Any, Dict, List, Literal, TypedDict
from datetime import datetime

class OrderBy(TypedDict):
    key: str
    direction: Literal["asc", "desc"]
    start_from: int | float | datetime


class VectorStoreBase:



    @abstractmethod
    async def add_documents(self, embeddings, metadata: List[Any], ids=None, namespace=None, batch_size=100):
        """Adds documents to the vector store"""
        raise NotImplementedError
    
    @abstractmethod
    async def update_documents(self, metadata: Dict, ids: List[str | int] | None=None, filters=None,  namespace=None):
        """Updates documents in the vector store"""
        raise NotImplementedError
    
    @abstractmethod
    async def similarity(self, query, top_k=3, filters=None, alpha=None, with_vectors=False, fussion: Literal["RRF", "RSF"] | None=None):
        """Searches for similar documents in the vector store"""
        raise NotImplementedError


    @abstractmethod
    async def create_collection(self):
        """Creates a collection in the vector store"""
        raise NotImplementedError


    async def delete_collection(self, namespace: str | None = None):
        """Deletes a collection in the vector store"""
        raise NotImplementedError
    

    async def delete_documents_ids(self, ids: List[str | int]):
        """Deletes documents from the vector store by id"""
        raise NotImplementedError
    
    async def delete_documents(self, filters: Any):
        """Deletes documents from the vector store by filters"""
        raise NotImplementedError
    

    async def get_documents(self, filters: Any,  ids: List[str | int] | None=None, top_k: int=10,  offset: int=0, with_payload: bool=False, with_vectors: bool=False, order_by: OrderBy | str | None=None):
        """Retrieves documents from the vector store"""
        raise NotImplementedError
    

    def _init_client(self):
        """Initializes the vector store client"""
        raise NotImplementedError
    

    def info(self):
        """Returns information about the vector store"""
        raise NotImplementedError


    # def __deepcopy__(self, memo):
    #     new_instance = self.__class__.__new__(self.__class__)
    #     memo[id(self)] = new_instance

    #     # Deep copy the data attribute
    #     # new_instance.data = copy.deepcopy(self.data, memo)        

    #     # Reinitialize the Qdrant client
    #     # new_instance.client = QdrantClient()
    #     new_instance._init_client()

    #     return new_instance