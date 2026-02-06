import asyncio
from chatboard.model.base_namespace import DatabaseType, NSManyToManyRelationInfo, NSRelationInfo, Namespace
from typing import TYPE_CHECKING, Iterator, Type, TypeVar, Dict, List, Any, Optional, ForwardRef

from chatboard.model.neo4j.namespace import Neo4jNamespace
from chatboard.model.postgres.builder import SQLBuilder
from chatboard.model.postgres.namespace import PostgresNamespace
from chatboard.model.postgres.operations import PostgresOperations
from chatboard.model.qdrant.namespace import QdrantNamespace
from chatboard.model.versioning import ArtifactLog


if TYPE_CHECKING:
    from chatboard.model.model import Model
    from chatboard.model.version_control_models import Branch, Turn


MODEL = TypeVar("MODEL", bound="Model")


class Extension:
    name: str
    is_installed: bool
    
    def __init__(self, name: str, is_installed: bool = False):
        self.name = name
        self.is_installed = is_installed

class NamespaceManager:
    """Manager for namespaces"""
    _namespaces: dict[str, Namespace] = {}
    _relations: dict[str, dict[str, dict[str, Any]]] = {}
    _reversed_relations: dict[str, dict[str, NSRelationInfo]] = {}
    _is_initialized: bool = False
    _main_branch: "Branch | None" = None
    extensions: dict[str, dict[str, Extension]] = {}
    
    @classmethod
    def initialize(cls):
        """Initialize the namespace manager"""
        cls._namespaces = {}
        cls._relations = {}
        cls.extensions = {
            "postgres": {},
            "qdrant": {},
        }
        cls._reversed_relations = {}
        
    @classmethod
    def build_namespace(cls, model_name: str, db_type: DatabaseType, is_versioned: bool = True, is_context: bool = False, is_artifact: bool = False, is_repo: bool = False, repo_namespace: Optional[str] = None) -> Namespace:
        """
        Build a namespace for a model.
        
        Args:
            model_name: The name of the model
            db_type: The type of database to use
            is_versioned: Whether the namespace should be versioned
            is_context: Whether the namespace should be a context
            is_repo: Whether the namespace should be a repo
            repo_namespace: The namespace of the repo this model belongs to
            
        Returns:
            The namespace for the model
        """
        if not cls._namespaces:
            cls.initialize()
        if db_type == "qdrant":
            # raise NotImplementedError("Qdrant is not implemented")
            namespace = QdrantNamespace(
                name=model_name,
                namespace_manager=cls,
            )
        elif db_type == "postgres":
            namespace = PostgresNamespace(
                name=model_name, 
                is_versioned=is_versioned, 
                is_repo=is_repo, 
                is_context=is_context,
                is_artifact=is_artifact,
                repo_namespace=repo_namespace,                 
                namespace_manager=cls
            )
        elif db_type == "neo4j":
            namespace = Neo4jNamespace(
                name=model_name,
                namespace_manager=cls,
            )
        else:
           raise ValueError(f"Invalid database type: {db_type}")
        cls._namespaces[model_name] = namespace
        return namespace
    
    @classmethod
    def register_extension(cls, db_type: DatabaseType, extension_name: str):
        if db_type not in cls.extensions:
            raise ValueError(f"Invalid database type: {db_type}")
        if extension_name not in cls.extensions[db_type]:
            cls.extensions[db_type][extension_name] = Extension(extension_name)
        
        
    @classmethod
    def install_extensions(cls):
        """
        Install the necessary PostgreSQL extensions.
        """ 
        for extension in cls.extensions["postgres"].values():
            if not extension.is_installed:
                SQLBuilder.create_extension(extension.name)
                extension.is_installed = True
       
        
    @classmethod
    def get_namespace(cls, model_name: str) -> Namespace:
        """
        Get a namespace by model name.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The namespace for the model
            
        Raises:
            ValueError: If the namespace manager is not initialized or the namespace is not found
        """
        if not cls._namespaces:
            raise ValueError("NamespaceManager not initialized")
        if not model_name in cls._namespaces:
            raise ValueError(f"Namespace for model {model_name} not found")
        # if not cls._is_initialized:
            # asyncio.create_task(cls.create_all_namespaces())
        return cls._namespaces[model_name]
    
    @classmethod
    def get_namespace_or_none(cls, model_name: str) -> Namespace | None:
        """
        Get a namespace by model name.
        """
        if not cls._namespaces:
            return None
        if not model_name in cls._namespaces:
            return None
        return cls._namespaces[model_name]
    
    @classmethod
    def get_namespace_by_model_cls(cls, model_cls: Type[MODEL] | str, throw_error: bool = True) -> Namespace:
        """
        Get a namespace by model class.
        """
        if not isinstance(model_cls, str):
            model_cls = model_cls.__name__        
        for namespace in cls._namespaces.values():
            if namespace.model_class.__name__ == model_cls:
                return namespace        
        if throw_error:
            raise ValueError(f"Namespace for model {model_cls} not found")
        return None
                
    @classmethod
    def get_turn_namespace(cls) -> Namespace:
        """
        Get the namespace for the turn model.
        """
        return cls._namespaces["turns"]
    
    @classmethod
    def replace_forward_refs(cls):
        """
        Replace forward refs with the actual model class.
        """
        for namespace in cls._namespaces.values():
            for relation in namespace.iter_relations():
                if isinstance(relation.foreign_cls, ForwardRef):
                    ns = cls.get_namespace_by_model_cls(relation.foreign_cls.__forward_arg__)
                    if ns is None:
                        raise ValueError(f"Namespace for model {relation.foreign_cls.__forward_arg__} not found")
                    relation.foreign_cls = ns.model_class
    
    @classmethod
    async def get_or_create_main_branch(cls):
        from .version_control_models import Branch, TurnStatus
        branch = await Branch.query().filter(id=1).first()
        if branch is None:
            branch = await Branch(name="main").save()
            turn = await branch.add_turn(status=TurnStatus.BRANCH_CREATED)
        return branch
    
    
    @classmethod
    def initialize_namespace_metadata(cls):
        cls.replace_forward_refs()        
        for namespace in cls.iter_namespaces("postgres"):            
            for relation in namespace.iter_relations():
                cls.add_reversed_relation(relation)
    
    @classmethod
    async def create_all_namespaces(cls):
        """
        Create all registered namespaces in the database.
        
        This method should be called after all models have been registered.
        """
        from .version_control_models import Branch
        if not cls._namespaces:
            raise ValueError("No namespaces registered")
        cls.install_extensions()
        
        cls.initialize_namespace_metadata()
        
        for namespace in cls.iter_namespaces("postgres"):
            SQLBuilder.create_enum_types(namespace)
        # if versioning:
            # await ArtifactLog.initialize_versioning()            
        
        
            
        # for namespace in cls._namespaces.values():
        for namespace in cls.iter_namespaces("postgres"):
            namespace.create_namespace()
            
        for namespace in cls.iter_namespaces("postgres"):
            namespace.create_foreign_keys()
            # if namespace.is_versioned:
                # SQLBuilder.create_foreign_key(
                #     table_name=namespace.table_name,
                #     column_name="turn_id",
                #     column_type="INTEGER",
                #     referenced_table="turns",
                #     referenced_column="id",
                #     on_delete="CASCADE",
                #     on_update="CASCADE",
                # )
                # SQLBuilder.create_foreign_key(
                #     table_name=namespace.table_name,
                #     column_name="branch_id",
                #     column_type="INTEGER",
                #     referenced_table="branches",
                #     referenced_column="id",
                #     on_delete="CASCADE",
                #     on_update="CASCADE",
                # )
            for field in namespace.iter_fields():
                if field.index:
                    SQLBuilder.create_index(
                        namespace=namespace,
                        column_name=field.name,
                        index_name=f"{namespace.table_name}_{field.name}_idx",
                    )
                
        # cls.initialize_namespace_metadata()
        
        await cls.get_or_create_main_branch()
        cls._is_initialized = True
        
    @classmethod
    def get_all_namespaces(cls) -> List[Namespace]:
        """
        Get all registered namespaces.
        
        Returns:
            A list of all registered namespaces
        """
        return list(cls._namespaces.values())
    
    @classmethod
    def iter_namespaces(cls, db_type: DatabaseType | None = None) -> Iterator[Namespace]:
        """
        Iterate over all registered namespaces.
        """
        for namespace in cls._namespaces.values():
            if db_type is None or namespace.db_type == db_type:
                yield namespace
                
    @classmethod
    def add_reversed_relation(cls, relation_info: NSRelationInfo | NSManyToManyRelationInfo):
        """
        Add a reversed relation to the namespace manager.
        """
        if isinstance(relation_info, NSManyToManyRelationInfo):
            table = relation_info.junction_table
            if table not in cls._reversed_relations:
                cls._reversed_relations[table] = {}                
            keys = relation_info.junction_keys
            cls._reversed_relations[table][keys[0]] = relation_info
            cls._reversed_relations[table][keys[1]] = relation_info            
        else:
            if relation_info.foreign_table not in cls._reversed_relations:
                cls._reversed_relations[relation_info.foreign_table] = {}
            cls._reversed_relations[relation_info.foreign_table][relation_info.foreign_key] = relation_info
        # target_table = relation_info.foreign_table if isinstance(relation_info, NSRelationInfo) else relation_info.junction_table
        
        # if target_table not in cls._reversed_relations:
        #     cls._reversed_relations[target_table] = {}
        # cls._reversed_relations[target_table][relation_info.foreign_key] = relation_info
        # if isinstance(relation_info, NSManyToManyRelationInfo):
            
            
    @classmethod
    def get_reversed_relation(cls, table_name: str, key: str) -> NSRelationInfo | None:
        """
        Get a reversed relation by table name and key.
        """
        return cls._reversed_relations.get(table_name, {}).get(key)
    
    
    @classmethod
    def register_relation(
        cls,
        source_namespace: str,
        relation_name: str,
        target_namespace: Optional[str],
        target_forward_ref: Optional[ForwardRef] = None,
        key: str = "id",
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
    ):
        """
        Register a relation between namespaces.
        
        Args:
            source_namespace: The namespace of the source model
            relation_name: The name of the relation
            target_namespace: The namespace of the target model
            target_forward_ref: The forward reference to the target model
            key: The name of the foreign key in the target model
            on_delete: The action to take when the referenced row is deleted
            on_update: The action to take when the referenced row is updated
        """
        if source_namespace not in cls._relations:
            cls._relations[source_namespace] = {}
        
        cls._relations[source_namespace][relation_name] = {
            "target_namespace": target_namespace,
            "target_forward_ref": target_forward_ref,
            "key": key,
            "on_delete": on_delete,
            "on_update": on_update,
        }
    
    @classmethod
    def get_relations(cls, namespace: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all relations for a namespace.
        
        Args:
            namespace: The namespace to get relations for
            
        Returns:
            A dictionary of relation names to relation information
        """
        return cls._relations.get(namespace, {})


    @classmethod
    async def create_turn(cls, branch_id: int, state: Optional[Any] = None, **kwargs) -> "Turn":
        """
        Create a turn for a partition.
        """
        return await ArtifactLog.create_turn(
            branch_id=branch_id,
            state=state,
            **kwargs
        )
        
    @classmethod
    async def get_last_turn(cls, branch_id: int, filters: dict[str, Any] = {}) -> "Turn | None":
        """
        Get the last turn for a partition and branch.
        """
        return await ArtifactLog.get_last_turn(branch_id, filters)
    
    @classmethod
    async def get_turn(cls, turn_id: int) -> "Turn":
        """
        Get a turn by id.
        """
        return await ArtifactLog.get_turn(turn_id)
    
    @classmethod
    async def create_branch(cls, name: Optional[str] = None, forked_from_turn_id: Optional[int] = None) -> "Branch":
        """
        Create a branch.
        """
        return await ArtifactLog.create_branch(name, forked_from_turn_id)
    
    @classmethod
    async def get_branch(cls, branch_id: int, raise_error: bool = False) -> "Branch | None":
        """
        Get a branch by id.
        """
        if raise_error:
            return await ArtifactLog.get_branch(branch_id)
        else:
            return await ArtifactLog.get_branch_or_none(branch_id)
    
    @classmethod
    async def commit_turn(cls, turn_id: int, message: Optional[str] = None) -> "Turn":
        """
        Commit a turn.
        """
        return await ArtifactLog.commit_turn(turn_id, message)
    
    
    @classmethod
    def drop_all_namespaces(cls):
        """
        Drop all namespaces.
        """
        SQLBuilder.drop_all_tables()
        for namespace in cls.iter_namespaces("qdrant"):
            namespace.drop_namespace()
        for namespace in cls.iter_namespaces("neo4j"):
            namespace.drop_namespace()
    
    
    @classmethod
    async def recreate_all_namespaces(cls):
        """
        Recreate all namespaces.
        """
        cls.drop_all_namespaces()
        await cls.create_all_namespaces()
