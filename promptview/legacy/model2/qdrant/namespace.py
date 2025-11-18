from typing import TYPE_CHECKING, Any, Optional, Type, List
from typing_extensions import TypeVar
from promptview.model.base_namespace import Distance, Namespace, NSFieldInfo, QuerySet, QuerySetSingleAdapter

from promptview.model.qdrant.field_info import QdrantFieldInfo
from promptview.model.qdrant.query_set import QdrantQuerySet  # Placeholder for later
from promptview.model.qdrant.connection import QdrantConnectionManager  # Placeholder for later

from qdrant_client.models import PointStruct
from qdrant_client.http import models
import uuid
if TYPE_CHECKING:
    from promptview.model.namespace_manager import NamespaceManager
    from promptview.model.model import Model

MODEL = TypeVar("MODEL", bound="Model")



def distance_to_qdrant(distance: Distance) -> models.Distance:
    if distance == Distance.COSINE:
        return models.Distance.COSINE
    elif distance == Distance.EUCLID:
        return models.Distance.EUCLID
    else:
        raise ValueError(f"Unsupported distance: {distance}")

class QdrantNamespace(Namespace[MODEL, QdrantFieldInfo]):
    """Qdrant implementation of Namespace"""
    
    def __init__(
        self,
        name: str,
        is_versioned: bool = False,
        is_repo: bool = False,
        is_context: bool = False,
        is_artifact: bool = False,
        repo_namespace: Optional[str] = None,
        namespace_manager: Optional["NamespaceManager"] = None
    ):
        super().__init__(
            name=name,
            db_type="qdrant",
            is_versioned=is_versioned,
            is_repo=is_repo,
            is_context=is_context,
            is_artifact=is_artifact,
            repo_namespace=repo_namespace,
            namespace_manager=namespace_manager
        )
        
    @property
    def client(self):
        return QdrantConnectionManager.get_client()
        
    def add_field(
        self,
        name: str,
        field_type: type[Any],
        default: Any | None = None,
        is_optional: bool = False,
        foreign_key: bool = False,
        is_key: bool = False,
        is_vector: bool = False,
        dimension: int | None = None,
        distance: Distance | None = None,
        is_primary_key: bool = False,
        is_default_temporal: bool = False,
        
    ) -> QdrantFieldInfo:
        """
        Add a field to the Qdrant namespace.
        """
        field = QdrantFieldInfo(
            name=name,
            field_type=field_type,
            default=default,
            is_optional=is_optional,
            foreign_key=foreign_key,
            is_key=is_key,
            is_vector=is_vector,
            dimension=dimension, 
            distance=distance,           
        )

        if is_primary_key:
            if self.find_primary_key() is not None:
                raise ValueError(f"Primary key field already exists, cannot add: {name}")
            self._primary_key = field

        if is_default_temporal:
            if self.default_temporal_field is not None:
                raise ValueError(f"Default temporal field already set: {self.default_temporal_field.name}")
            self.default_temporal_field = field

        if field.is_vector:
            self.vector_fields.add_vector_field(name, field)

        self._fields[name] = field
        return field



    async def insert(self, data: dict[str, Any]) -> dict[str, Any]:
        namespace = self
        client = QdrantConnectionManager.get_client()
        collection_name = namespace.name

        dump = namespace.validate_model_fields(data)

        dump, vectors = self.vector_fields.split_vector_data(dump)
        # if namespace.need_to_transform:
        #     vectors = await namespace.vector_fields.transform(data)
            # dump.update(vector_payload)

        # point_id = dump.get(namespace.primary_key.name)
        point_id = namespace.get_dump_primary_key(dump)
        if point_id is None:
            if namespace.primary_key.data_type == int:
                raise ValueError("integer primary key is required for Qdrant namespace")
            point_id = str(uuid.uuid4())
            # dump[namespace.primary_key.name] = point_id

        if not vectors:
            raise ValueError("Missing vector payload: cannot save Qdrant point without embedding")
        
        vector_payload = {vec_name: vectors[vec_name].tolist() for vec_name in vectors.keys()}

        point = PointStruct(
            id=point_id,
            vector=vector_payload,
            payload=dump
        )

        await client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        dump.update(vector_payload)
        dump.update({namespace.primary_key.name: point_id})
        return dump


    async def get(self, id: Any) -> dict[str, Any] | None:
        client = QdrantConnectionManager.get_client()
        collection_name = self.name

        result = await client.retrieve(
            collection_name=collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=True
        )

        if not result:
            return None

        return self.pack_model(result[0])
    
    
    def pack_model(self, record: models.Record) -> dict[str, Any]:
        dump = {}
        for field in self.iter_fields():
            if field.is_key:
                dump[field.name] = record.id
            elif field.is_vector and record.vector is not None:
                if field.name in record.vector:
                    dump[field.name] = record.vector[field.name]            
                else:
                    raise ValueError(f"Vector field {field.name} not found in record")
            elif field.name in record.payload:
                dump[field.name] = record.payload[field.name]
            else:
                raise ValueError(f"Field {field.name} not found in record")
        
        return dump


    async def delete(self, id: Any) -> MODEL | None:
        client = QdrantConnectionManager.get_client()
        collection_name = self.name

        existing = await self.get(id)
        if not existing:
            return None

        await client.delete(
            collection_name=collection_name,
            points_selector={"points": [id]}
        )
        return existing


    def query(self, **kwargs) -> QuerySet:
        """Return a QdrantQuerySet"""
        return QdrantQuerySet(self.model_class)

    
    async def create_namespace(self):
        collection_name = self.name
        """Create Qdrant collection (vector index)"""
        if not await self.client.collection_exists(collection_name):
            vectors_config = {}
            for field in self.vector_fields:
                if not field.dimension:
                    raise ValueError(f"Dimension is required for vector field: {field.name}")
                if not field.distance:
                    raise ValueError(f"Distance is required for vector field: {field.name}")
                vectors_config[field.name] = models.VectorParams(size=field.dimension, distance=distance_to_qdrant(field.distance))
            
            return await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

    async def drop_namespace(self):
        """Drop Qdrant collection"""
        return await self.client.delete_collection(collection_name=self.name)
    
    
    async def recreate_namespace(self):
        """Recreate Qdrant collection"""
        await self.drop_namespace()
        return await self.create_namespace()