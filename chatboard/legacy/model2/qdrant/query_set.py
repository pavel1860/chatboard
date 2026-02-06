



from typing import TYPE_CHECKING, Generic, Type, TypeVar, Callable, List, Any
from chatboard.model.base_namespace import QuerySet
from qdrant_client.http.models import ScoredPoint, Record
from chatboard.model.postgres.query_set3 import QueryProxy
from chatboard.model.postgres.sql.expressions import Eq, param
from chatboard.model.postgres.sql.queries import Column, Table
from chatboard.model.qdrant.compiler import QdrantCompiler
from chatboard.model.qdrant.connection import QdrantConnectionManager
from functools import reduce
from operator import and_

if TYPE_CHECKING:
    from chatboard.model.model import Model

MODEL = TypeVar("MODEL", bound="Model")




class QdrantQuerySet(QuerySet[MODEL], Generic[MODEL]):
    def __init__(self, model_class: Type[MODEL]):
        super().__init__(model_class)
        self._filters: dict[str, Any] = {}
        self._limit: int | None = None
        self._query: str | None = None
        self.namespace = model_class.get_namespace()

    def filter(self, condition: Callable[[MODEL], Any] | None = None, **kwargs) -> "QdrantQuerySet[MODEL]":
        expressions = []

        if condition is not None:
            proxy = QueryProxy(self.model_class, Table(self.namespace.name))
            self._filters = condition(proxy)
        if kwargs:
            for field, value in kwargs.items():
                col = Column(field, Table(self.namespace.name))
                expressions.append(Eq(col, param(value)))
                
        if expressions:
            expr = reduce(and_, expressions) if len(expressions) > 1 else expressions[0]
            self._filters = expr

        return self

    def limit(self, limit: int) -> "QdrantQuerySet[MODEL]":
        self._limit = limit
        return self
    
    
    def similar(self, query: str) -> "QdrantQuerySet[MODEL]":
        self._query = query
        return self

    async def execute(self) -> List[MODEL]:
        ns = self.model_class.get_namespace()
        collection_name = ns.name
        vectors = None
        if self._query:
            vectors = await ns.batch_vectorizer.embed_query(self._query)
            
        # This example assumes a simple key-value filtering with no payload transformation yet
        # results = await QdrantConnectionManager.simple_query(
        #     collection_name=collection_name,
        #     filters=self._filters,
        #     limit=self._limit
        # )
        filters = None
        if self._filters:
            compiler = QdrantCompiler()
            filters = compiler.compile_expr(self._filters)
            
            
        primary_key = ns.primary_key.name
        
        if vectors is not None:
            def unpack_point(point: ScoredPoint) -> dict[str, Any]:
                meta = point.payload or {}
                return meta | {
                    "score": point.score,
                    "vector": point.vector,
                } | {primary_key: point.id}
            results = await QdrantConnectionManager.execute_query(
                collection_name=collection_name,
                query=vectors,
                limit=self._limit,
                filters=filters
            )
            return [ns.instantiate_model(unpack_point(hit)) for hit in results]
        else:
            
            def unpack_record(record: Record) -> dict[str, Any]:
                meta = record.payload or {}
                return meta | {
                    "vector": record.vector,
                } | {primary_key: record.id}
            results = await QdrantConnectionManager.scroll(
                collection_name=collection_name,
                filters=filters,
                limit=self._limit,
            )
            return [ns.instantiate_model(unpack_record(hit)) for hit in results]
        
        
        
            

        
        
