import asyncio
import os
import random
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
from uuid import uuid4

from ..fussion.rsf_fussion import rsf_fussion
from ..stores.base import OrderBy, VectorStoreBase
from ..vectorizers.base import (VectorizerBase,
                                                     VectorizerDenseBase,
                                                     VectorizerSparseBase,
                                                     VectorMetrics)
from httpx import ConnectTimeout
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (DatetimeRange, Distance, FieldCondition,
                                  Filter, NamedSparseVector, NamedVector,
                                  PointStruct, Range, SearchRequest,
                                  SparseIndexParams, SparseVector,
                                  SparseVectorParams, VectorParams)
from ..utils import chunks

import os
from datetime import datetime


class VectorSearchResult(BaseModel):
    id: str | int
    score: float
    metadata: Any
    vector: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class VectorStoreException(Exception):
    pass


class VectorDbIndexType(str, Enum):
    keyword = "keyword"
    integer = "integer"
    float = "float"
    bool = "bool"
    geo = "geo"
    datetime = "datetime"
    text = "text"


def metrics_to_qdrant(metric: VectorMetrics):
    if metric == VectorMetrics.COSINE:
        return Distance.COSINE
    elif metric == VectorMetrics.EUCLIDEAN:
        return Distance.EUCLIDEAN
    elif metric == VectorMetrics.MANHATTAN:
        return Distance.MANHATTAN
    else:
        raise ValueError(f"Unsupported metric {metric}")

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=VectorizerBase)

# class QdrantVectorStore(Generic[T, U]):
class QdrantVectorStore(VectorStoreBase):

    def __init__(
        self,
        collection_name: str,
        url: str | None=None,
        api_key: str | None = None,
        ) -> None:
        self.url = url or os.environ.get("QDRANT_URL")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY", None)
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=True,
            # host=os.environ['QDRANT_HOST'], 
            # port=os.environ['QDRANT_PORT']
        )
        # self._init_client()
        self.collection_name = collection_name


    def _pack_points(self, points):
        recs = []
        for p in points:
            rec = VectorSearchResult(
                id=p.id,
                score=p.score if hasattr(p, "score") else -1,
                metadata=p.payload,
                vector=p.vector
            )
            recs.append(rec)
        return recs
       

    async def similarity(
            self, 
            query, 
            top_k=3, 
            filters=None, 
            alpha=None, 
            with_vectors=False, 
            fussion: Literal["RRF", "RSF"] | None = None,
            retry: int=3,
            base_delay=1, 
            max_delay=8
        ):
        query_filter = None
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            query_filter = models.Filter(
                must_not=must_not,
                must=must
            )
        for attempt in range(retry):
            try:
                if len(query.items()) == 1 and fussion == None:
                    res = await self._dense_similarity(query, top_k, query_filter, with_vectors)
                else:
                    if fussion is None:
                        raise ValueError("Fussion method must be provided.")
                    if fussion == "RRF":
                        res = await self._rff_similarity(query, top_k, query_filter, with_vectors)
                    elif fussion == "RSF":
                        res = await self._rsf_similarity(query, top_k, query_filter, with_vectors)
                
                return self._pack_points(res)
            except ResponseHandlingException as e:
                if type(e.args[0]) == ConnectTimeout:
                    print(ConnectTimeout)
                    if attempt == retry - 1:
                        raise e
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    await asyncio.sleep(delay + random.uniform(0, 1))
                    print("Retrying...")
                    continue
                raise e

        

    async def _dense_similarity(self, query, top_k=3, query_filter=None, with_vectors=False):
        vector_name, vector_value = list(query.items())[0]
        recs = await self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedVector(
                name=vector_name,
                vector=vector_value
            ),
            query_filter=query_filter,
            limit=top_k,            
            with_payload=True,
            with_vectors=with_vectors,
        )
        return recs
    
    async def _rff_similarity(self, query, top_k=3, query_filter=None, with_vectors=False):
        prefetch = []
        for name, vector in query.items():
            if type(vector) == list:
                prefetch.append(
                    models.Prefetch(
                        query=vector,
                        using=name,
                        # limit=top_k,
                        filter=query_filter
                    )
                )
            elif type(vector) == dict:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(indices=vector['indices'], values=vector['values']),
                        using=name,
                        # limit=top_k,
                        filter=query_filter
                    )
                )
        
        recs = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_vectors=with_vectors
        )
        return recs.points

    async def _rsf_similarity(self, query, top_k=3, query_filter=None, with_vectors=False, alpha: List[float] | float=0.5):
    
        search_requests = []
        for name, vector in query.items():
            if type(vector) == list:
                search_requests.append(
                    SearchRequest(
                        vector=models.NamedVector(
                            name=name,
                            vector=vector                    
                        ),
                        limit=top_k,
                        filter=query_filter,
                        with_payload=True,
                        with_vector=with_vectors,

                    )
                )
            elif type(vector) == dict:
                search_requests.append(
                    models.SearchRequest(
                        vector=models.NamedSparseVector(
                            vector=models.SparseVector(indices=vector['indices'], values=vector['values']),
                            name=name,
                        ),        
                        limit=top_k,                        
                        filter=query_filter,
                        with_vector=with_vectors,
                    )
                )
        output = await self.client.search_batch(collection_name=self.collection_name, requests=search_requests)
        return rsf_fussion(output, top_k, alpha)



    async def add_documents(self, vectors, metadata: List[Dict | BaseModel], ids=None, namespace=None, batch_size=100):
        namespace = namespace or self.collection_name
        # metadata = [m.dict(exclude={'__orig_class__'}) if isinstance(m, BaseModel) else m for m in metadata]
        metadata = [m if isinstance(m, dict) else m.model_dump(exclude={'__orig_class__'}) for m in metadata]
        if not ids:
            ids = [str(uuid4()) for i in range(len(vectors))]
        
        results = []
        for vector_chunk in chunks(zip(ids, vectors, metadata), batch_size=batch_size):
            points = [
                PointStruct(
                    id=id_,
                    payload=meta,
                    vector=vec
                )
                for id_, vec, meta in vector_chunk]
            await self.client.upsert(
                collection_name=namespace,
                points=points
            )
            results += [
                VectorSearchResult(
                    id=p.id,
                    score=-1,
                    metadata=p.payload,
                    vector=p.vector
                ) 
                for p in points]        
        return results

    async def update_documents(self, metadata: Dict, ids: List[str | int] | None=None, filters=None,  namespace=None):
        namespace = namespace or self.collection_name        
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            points = models.Filter(
                must_not=must_not,
                must=must
            )
        elif ids is not None:
            points = ids
        else:
            raise ValueError("ids or filters must be provided.")
        
        return await self.client.set_payload(
            collection_name=namespace,
            payload=metadata,
            points=points
        )



    async def create_collection(self, vectorizers, collection_name: str | None = None, indexs=None):
        http_client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=False,
        )
        collection_name=collection_name or self.collection_name
        vector_config = {}
        sparse_vector_config = {}
        for vectorizer in vectorizers:
            if isinstance(vectorizer, VectorizerDenseBase):
                vector_config[vectorizer.name] = VectorParams(size=vectorizer.size, distance=metrics_to_qdrant(vectorizer.metric))
            elif isinstance(vectorizer, VectorizerSparseBase):
                sparse_vector_config[vectorizer.name] = SparseVectorParams(
                    index=models.models.SparseIndexParams()
                )
        await http_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            sparse_vectors_config=sparse_vector_config           
        )
        if indexs:
            for index in indexs:
                try:
                    await http_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=index['field'],
                        field_schema=index['schema']
                    )
                except Exception as e:
                    raise e


    async def delete_collection(self, collection_name: str | None =None):
        collection_name = collection_name or self.collection_name
        await self.client.delete_collection(collection_name=collection_name)



    # async def delete_documents_ids(self, ids: List[Union[str, int]]):
    async def delete_documents_ids(self, ids: List[str | int]):
        res = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids,
            )
        )
        return res

    async def delete_documents(self, filters):
        if not filters:
            raise ValueError("filters must be provided. to delete all documents, use delete_collection method.")
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            filter_ = models.Filter(
                must_not=must_not,
                must=must
            )
        
        res = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=filter_
            )
        )
        return res


    async def get_documents(
            self, 
            filters: Any,  
            ids: List[str | int] | None=None, 
            top_k: int=10, 
            offset: int=0,
            with_payload=False, 
            with_vectors=False, 
            order_by: OrderBy | str | None=None,
        ):
        filter_ = None
        if ids is not None:
            # top_k: int | None = None
            filter_ = models.Filter(
                must=[
                    models.HasIdCondition(has_id=ids)
                ],
            )
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            filter_ = models.Filter(
                must_not=must_not,
                must=must
            )
        if order_by:
            if type(order_by) == str:
                pass                
            elif type(order_by) == dict:
                order_by = models.OrderBy(
                    key=order_by.get("key"),
                    direction=order_by.get("direction", "desc"), # type: ignore
                    start_from=order_by.get("start_from", None),  # start from this value
                )
                offset = None
            else:
                raise ValueError("order_by must be a string or a dict.")
        
            
        recs, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_,
            limit=top_k,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            order_by=order_by # type: ignore
        )
        return self._pack_points(recs)
    

    # async def group_documents(
    #         self,             
    #         group_by: Dict | None=None,
    #         group_size=None,
    #     ):
    #     self.client.search_groups(
    #         collection_name=self.collection_name,

    #     )


    async def get_many(self, ids: List[str | int] | None=None, top_k=10, with_payload=False, with_vectors=False):
        filter_ = None
        if ids is not None:
            top_k = None
            filter_ = models.Filter(
                must=[
                    models.HasIdCondition(has_id=ids)
                ],
            )
        recs, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return self._pack_points(recs)


    async def info(self):       
        print("* Getting collection info", self.collection_name) 
        return await self.client.get_collection(self.collection_name)
        
            


    def parse_filter(self, filters: Dict[str, Any]):

        must = []
        must_not = []

        # def is_range(value):    
        #     for k, v in value.items():
        #         if k not in ["$gt", "$gte", "$lt", "$lte"]:
        #             return False
        #     return True

        # def unpack_range(value):
        #     gt=value.get('$gt', None)
        #     gte=value.get('$gte', None)
        #     lt=value.get('$lt', None)
        #     lte=value.get('$lte', None)
        #     if type(gt) == datetime or type(gte) == datetime or type(lt) == datetime or type(lte) == datetime:
        #         gt = gt.isoformat(timespec='seconds') if gt else None
        #         gte = gte.isoformat(timespec='seconds') if gte else None
        #         lt = lt.isoformat(timespec='seconds') if lt else None
        #         lte = lte.isoformat(timespec='seconds') if lte else None
        #         return models.DatetimeRange(gt=gt,gte=gte,lt=lt,lte=lte)            
        #     return models.Range(gt=gt,gte=gte,lt=lt,lte=lte)
        def is_range(value):    
            for k, v in value.items():
                if k not in [">", ">=", "<", "<="]:
                    return False
            return True

        def unpack_range(value):
            gt=value.get('>', None)
            gte=value.get('>=', None)
            lt=value.get('<', None)
            lte=value.get('<=', None)
            if type(gt) == datetime or type(gte) == datetime or type(lt) == datetime or type(lte) == datetime:
                gt = gt.isoformat(timespec='seconds') if gt else None
                gte = gte.isoformat(timespec='seconds') if gte else None
                lt = lt.isoformat(timespec='seconds') if lt else None
                lte = lte.isoformat(timespec='seconds') if lte else None
                return models.DatetimeRange(gt=gt,gte=gte,lt=lt,lte=lte)            
            return models.Range(gt=gt,gte=gte,lt=lt,lte=lte)

        for field, value in filters.items():
            if type(value) == dict:
                if is_range(value):
                    must.append(models.FieldCondition(key=field, range=unpack_range(value)))
                else:
                    for k, v in value.items():
                        if k == "$ne":
                            must_not.append(models.FieldCondition(key=field, match=models.MatchValue(value=v)))
                        elif k == "$eq":
                            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=v)))
            else:
                if type(value) == list:
                    must.append(models.FieldCondition(key=field, match=models.MatchAny(any=value)))
                else:
                    must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
                    

        return must_not, must

    async def close(self):
        await self.client.close()