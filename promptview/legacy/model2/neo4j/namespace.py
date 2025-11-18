# neo4j_namespace.py
from typing import TYPE_CHECKING, Any
import uuid
from promptview.model.base_namespace import Distance, NSRelationInfo, Namespace
from promptview.model.neo4j.connection import Neo4jConnectionManager
from promptview.model.neo4j.field_info import Neo4jFieldInfo
from promptview.model.neo4j.query_set import Neo4jQuerySet


if TYPE_CHECKING:
    from promptview.model.namespace_manager import NamespaceManager
    from promptview.model.model import Model

class Neo4jNamespace(Namespace):
    """
    Neo4j implementation of Namespace.
    Handles mapping of model fields to Neo4j node properties, and model relations to Neo4j relationships.
    """
    def __init__(self, name, namespace_manager: "NamespaceManager"):
        super().__init__(name, db_type="neo4j", namespace_manager=namespace_manager)
        # self.driver = ... (set up Neo4j driver here, or use ResourceManager for connections)

    

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
    ) -> Neo4jFieldInfo:
        """
        Register a node property.
        """
        field = Neo4jFieldInfo(
            name=name,
            field_type=field_type,
            default=default,
            is_optional=is_optional,
            foreign_key=foreign_key,
            is_key=is_key,
        )
        if is_primary_key:
            if self._primary_key is not None:
                raise ValueError(f"Primary key field already exists, cannot add: {name}")
            self._primary_key = field
        if is_default_temporal:
            if self.default_temporal_field is not None:
                raise ValueError(f"Default temporal field already set: {self.default_temporal_field.name}")
            self.default_temporal_field = field
            
        self._fields[name] = field
        return field
        

    def add_relation(
        self, 
        name, 
        primary_key, 
        foreign_key, 
        foreign_cls, 
        junction_cls=None,
        junction_keys=None,
        on_delete="CASCADE", 
        on_update="CASCADE"):
        """
        Register a relationship type in Neo4j.
        """
        # Store in self._relations, build NSRelationInfo
        relation_info = NSRelationInfo(
            namespace=self,
            name=name,
            primary_key=primary_key,
            foreign_key=foreign_key,
            foreign_cls=foreign_cls,
            junction_cls=junction_cls,
            junction_keys=junction_keys,
            on_delete=on_delete,
            on_update=on_update,
        )
        self._relations[name] = relation_info
        return relation_info

    def add_many_relation(self, name, primary_key, foreign_key, foreign_cls, junction_cls, junction_keys, on_delete="CASCADE", on_update="CASCADE"):
        """
        Register a many-to-many relationship using intermediate nodes/relationships.
        """
        pass
    
    async def insert(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """
        Insert a node into the namespace.
        """
        label = self.name
        pk = self.primary_key.name
        node_props = {}
        for fname, finfo in self._fields.items():
            # value = getattr(model, fname, None)
            value = data.get(fname, None)
            if fname == pk and value is None:
                if self.primary_key.data_type == int:
                    raise ValueError("integer primary key is required for Neo4j namespace")
                value = str(uuid.uuid4())

            node_props[fname] = finfo.serialize(value)

        merge_clause = f"MERGE (n:`{label}` {{{pk}: ${pk}}})"
        set_props = [f"n.{fname} = ${fname}" for fname in self._fields if fname != pk]
        set_clause = ""
        if set_props:
            set_clause = "SET " + ", ".join(set_props)
        cypher_node = f"{merge_clause}\n{set_clause}\nRETURN n"

        record = await Neo4jConnectionManager.execute_write(cypher_node, node_props)
        # record = await result.single()
        if record:
            return node_props
        return None

    async def save(self, model):
        """
        Save or update a node (MERGE) and create relationships.
        """
        pass

    async def get(self, id):
        """
        Retrieve a node by its primary key property.
        Returns a hydrated model instance or None.
        """
        label = self.name
        pk = self.primary_key.name

        # Build Cypher and parameters
        cypher = f"MATCH (n:`{label}`) WHERE n.{pk} = $id RETURN n LIMIT 1"
        params = {"id": id}

        # Fetch the result using the connection manager
        record = await Neo4jConnectionManager.execute_read(cypher, params)
        if not record:
            return None

        node_props = record[0]["n"]

        # Build a dict for the model, including field deserialization
        return self.pack_model(node_props)
        

        # Optionally, relations could be hydrated separately (not implemented here)
        
    def pack_model(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Pack a Neo4j record into a model instance.
        """
        data = {}
        for fname, finfo in self._fields.items():
            val = record.get(fname)
            data[fname] = finfo.deserialize(val)
        return data

    async def delete(self, id):
        """
        Delete a node and optionally its relationships.
        """
        pass

    def query(self, **kwargs):
        """
        Start a Neo4jQuerySet for this model.
        """
        return Neo4jQuerySet(self.model_class, **kwargs)


    async def create_namespace(self):
        """
        (Optional) Create constraints or indexes in Neo4j for this model.
        """
        pass  # Neo4j is schema-less, but you can create constraints/indexes if desired.

    async def drop_namespace(self):
        """Drop Neo4j collection"""
        return await Neo4jConnectionManager.execute_write("MATCH (n) DETACH DELETE n")
    
    
    async def recreate_namespace(self):
        """Recreate Neo4j collection"""
        await self.drop_namespace()
        await self.create_namespace()