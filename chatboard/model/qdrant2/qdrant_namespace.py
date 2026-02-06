from typing import TYPE_CHECKING, Any, List, Optional, Union, Type
import uuid




from ..base.base_namespace import BaseNamespace
from ..qdrant2.qdrant_field_info import QdrantFieldInfo
from .connection import QdrantConnectionManager

from qdrant_client.http.models import PointStruct, PointVectors, VectorParams, Distance, Filter
if TYPE_CHECKING:
    from ..model3 import Model

class QdrantNamespace(BaseNamespace["Model", QdrantFieldInfo]):
    def __init__(self, name: str, *fields: QdrantFieldInfo):
        super().__init__(name, db_type="qdrant")
        self._pk_field: Optional[QdrantFieldInfo] = None

        for f in fields:
            self.add_field(f)
            if f.is_primary_key:
                if self._pk_field:
                    raise ValueError("Multiple primary keys detected")
                self._pk_field = f

        if not self._pk_field:
            raise ValueError("Primary key field is required")

    def _vector_fields(self) -> List[QdrantFieldInfo]:
        return [f for f in self.iter_fields() if f.is_vector]

    @property
    def primary_key(self) -> QdrantFieldInfo:
        return self._pk_field  # type: ignore

    async def create_namespace(self, force: bool = False):
        client = QdrantConnectionManager.get_client()
        vfields = self._vector_fields()
        if not vfields:
            raise ValueError("At least one vector field required")
        if not force and await client.collection_exists(collection_name=self.name):
            return
        vectors_config = {
            f.name: VectorParams(size=f.dimension, distance=Distance[f.distance.name])
            for f in vfields
        }
        await client.create_collection(
            collection_name=self.name,
            vectors_config=vectors_config,
        )

    async def drop_namespace(self, ignore_if_not_exists: bool = True):
        client = QdrantConnectionManager.get_client()
        await client.delete_collection(
            collection_name=self.name
        )
        
    def make_field_info(self, **kwargs) -> QdrantFieldInfo:
        return QdrantFieldInfo(**kwargs)

    async def insert(self, data: dict[str, Any]) -> dict[str, Any]:
        namespace = self
        client = QdrantConnectionManager.get_client()
        collection_name = namespace.name

        dump = namespace.validate_model_fields(data)

        dump, vectors = self.split_vector_data(dump)
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
        dump.update({namespace.primary_key: point_id})
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

    async def update_vectors(self, updates: List[dict]):
        """
        Replace vector(s) for existing points, leaving payload intact.
        Each dict should include: { 'id': <id>, '<vector_field>': <vector list>, ... }
        """
        client = QdrantConnectionManager.get_client()
        vecs = []
        for upd in updates:
            pid = upd[self.primary_key]
            vdict = {f.name: upd[f.name] for f in self._vector_fields() if f.name in upd}
            if not vdict:
                raise ValueError("No vector field in update")
            vecs.append(PointVectors(id=pid, vector=vdict))

        await client.update_vectors(
            collection_name=self.name,
            points=vecs,
            wait=True
        )

    async def delete_vectors(self, ids: List[Union[int, str]], vector_names: List[str]):
        client = QdrantConnectionManager.get_client()
        await client.delete_vectors(
            collection_name=self.name,
            points=ids,
            vectors=vector_names,
            wait=True
        )

    async def delete(self, point_id: Union[int, str]):
        client = QdrantConnectionManager.get_client()
        await client.delete_points(
            collection_name=self.name,
            points=[point_id],
            wait=True
        )

    async def search(self, query_vector: List[float], limit: int = 10, filters: Optional[Filter] = None, with_vectors: bool = False):
        """
        Vector similarity search. Returns list of dict with payloads.
        """
        client = QdrantConnectionManager.get_client()
        res = await client.search(
            collection_name=self.name,
            query_vector=query_vector,
            query_filter=filters,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors
        )
        out = []
        for pt in res:
            row = {**pt.payload}
            if with_vectors:
                for v in pt.vectors or []:
                    row[v.name] = v.vector
            out.append(row)
        return out
