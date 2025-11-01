import sys
from typing import TYPE_CHECKING, Any, Dict, ForwardRef, Type, Optional, get_args, get_origin, List, Union

from promptview.model.base.types import VersioningStrategy
from promptview.utils.db_connections import PGConnectionManager
from .postgres2.pg_namespace import PgNamespace
from .qdrant2.qdrant_namespace import QdrantNamespace
from .util import resolve_annotation
import time

if TYPE_CHECKING:
    from .model3 import Model
    from .versioning.models import Branch

_extensions_registry = set()

class NamespaceManager:
    _initialized = False
    _registry: Dict[tuple[str, str], Any] = {}
    _model_to_namespace: Dict[Type, Any] = {}
    _main_branch: "Branch | None" = None

    @classmethod
    def build_namespace(cls, model_name: str, db_type: str = "postgres", versioning_strategy: VersioningStrategy = VersioningStrategy.NONE, **kwargs):
        key = (model_name, db_type)
        if key in cls._registry:
            return cls._registry[key]
        if db_type == "postgres":
            ns = PgNamespace(model_name, versioning_strategy=versioning_strategy)
        elif db_type == "qdrant":
            ns = QdrantNamespace(model_name)
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")
        cls._registry[key] = ns
        return ns

    @classmethod
    def get_namespace(cls, namespace_name: str, db_type: str = "postgres"):
        key = (namespace_name, db_type)
        ns = cls._registry.get(key)
        if ns is None:
            raise ValueError(f"Namespace '{namespace_name}' (db: {db_type}) not found.")
        return ns

    @classmethod
    def get_namespace_or_none(cls, namespace_name: str, db_type: str = "postgres"):
        return cls._registry.get((namespace_name, db_type))

    @classmethod
    def register_model_namespace(cls, model_cls: Type, namespace: Any):
        cls._model_to_namespace[model_cls] = namespace

    @classmethod
    def get_namespace_for_model(cls, model_cls: Type):
        ns = cls._model_to_namespace.get(model_cls)
        if ns is None:
            raise ValueError(f"Namespace for model {model_cls} not found.")
        return ns
    
    @classmethod
    def get_main_branch(cls):
        if cls._main_branch is None:
            raise ValueError("Main branch not initialized")
        return cls._main_branch
    
    @classmethod
    def get_main_branch_or_none(cls):
        return cls._main_branch
    
    
    @classmethod
    async def init_main_branch(cls):
        from .versioning.models import Branch
        cls._main_branch = await Branch().get_main()
        print("Initialized main branch")


    @classmethod
    async def initialize_all(cls, init_main_branch: bool = True):
        if cls._initialized:
            print("NamespaceManager already initialized")
            return        
        start_time = time.time()
            
        await PgNamespace.install_extensions()
        # from .versioning.models import Branch
        # from ..evaluation.models import TestTurn
        # Branch.model_rebuild()
        # BlockNode.model_rebuild()
        cls.finalize()
        # 1️⃣ Create all tables first
        for ns in cls._registry.values():
            if hasattr(ns, "create_namespace"):
                await ns.create_namespace()
        print("Created all namespaces")
        # 2️⃣ Then add all foreign keys
        for ns in cls._registry.values():
            if hasattr(ns, "add_foreign_keys"):
                await ns.add_foreign_keys()
        print("Added all foreign keys")
        if init_main_branch:
            await cls.init_main_branch()
        end_time = time.time()
        print(f"initialize_all took {end_time - start_time} seconds")
        cls._initialized = True
                
    @classmethod
    async def initialize_clean(cls):
        """ 
        Initialize the namespace manager with a clean database.
        """        
        cls._initialized = False
        await PGConnectionManager.close()
        cls.drop_all_tables()
        await cls.initialize_all()
        

    
    # @classmethod
    # def finalize(cls):
    #     """Run any pending parsers after all models are loaded, and resolve relations."""
    #     for ns in cls._registry.values():
    #         if hasattr(ns, "_pending_field_parser"):
    #             ns._pending_field_parser.parse()
    #             del ns._pending_field_parser
    #         if hasattr(ns, "_pending_relation_parser"):
    #             ns._pending_relation_parser.parse()
    #             del ns._pending_relation_parser

    #     # NEW: Resolve foreign class forward refs
    #     globalns = {}
    #     for model_cls in cls._model_to_namespace.keys():
    #         globalns[model_cls.__name__] = model_cls

    #     for ns in cls._registry.values():
    #         for rel in ns._relations.values():
    #             try:
    #                 rel.resolve_foreign_cls(globalns)
    #             except Exception as e:
    #                 raise RuntimeError(f"Failed to resolve relation {rel.name} in {ns.name}: {e}")
    @classmethod
    def finalize(cls):
        # 1. Build globalns
        globalns = {model_cls.__name__: model_cls for model_cls in cls._model_to_namespace.keys()}

        # 2. Resolve forward refs on all models before parsing
        for model_cls in list(globalns.values()):
            for field_name, field in model_cls.model_fields.items():
                try:
                    field.annotation = resolve_annotation(field.annotation, globalns)
                except ValueError as e:
                    e.add_note(f"Failed to resolve annotation for field {field_name} in {model_cls.__name__}")
                    raise e

        # 3. Now run all parsers with resolved annotations
        for ns in cls._registry.values():
            if hasattr(ns, "_pending_field_parser"):
                ns._pending_field_parser.parse()
                # del ns._pending_field_parser
                
        for ns in cls._registry.values():       
            if hasattr(ns, "_pending_relation_parser"):
                ns._pending_relation_parser.parse()
                del ns._pending_relation_parser
                
        for ns in cls._registry.values():
            if hasattr(ns, "_pending_field_parser"):
                ns._pending_field_parser.validate_foreign_keys()
                del ns._pending_field_parser
                
        for ns in cls._registry.values():
            for name, rel in ns._relations.items():
                rev_rel = rel.foreign_namespace.get_relation_for_namespace(ns, foreign_key=rel.foreign_key)
                if rev_rel is not None and ns != rel.foreign_namespace:
                    if rev_rel.primary_key != rel.foreign_key:
                        raise ValueError(f"Primary key '{rev_rel.primary_key}' of '{rev_rel.name}' ({rev_rel.primary_cls.__name__}) does not match foreign key '{rel.foreign_key}' of '{rel.name}' ({rel.primary_cls.__name__})")
                    if rev_rel.foreign_key != rel.primary_key:
                        raise ValueError(f"Foreign key '{rev_rel.foreign_key}' ({rev_rel.foreign_cls.__name__}) of '{rev_rel.name}' does not match primary key '{rel.primary_key}' of '{rel.name}' ({rel.foreign_cls.__name__})")


    @classmethod
    async def drop_all_namespaces(cls, dry_run: bool = False) -> list[str]:
        sql_statements = []
        for (_, _), ns in list(cls._registry.items()):
            if hasattr(ns, "drop_namespace") and callable(ns.drop_namespace):
                if dry_run:
                    sql = await ns.drop_namespace(dry_run=True)
                    if sql:
                        sql_statements.append(sql)
                else:
                    await ns.drop_namespace(dry_run=False)
        return sql_statements



    @classmethod
    def drop_all_tables(cls, exclude_tables: list[str] | None = None):
        from .postgres2.pg_manager import PgManager
        PgManager.drop_all_tables(exclude_tables)
        PgManager.drop_enum_types(exclude_tables)
    
    @classmethod
    def get_turn_namespace(cls):
        return cls._registry.get(("turns", "postgres"))
    
    @classmethod
    def get_branch_namespace(cls):
        return cls._registry.get(("branches", "postgres"))
    
    @classmethod
    def should_save_to_db(cls):
        turn = cls.get_turn_namespace() is not None and cls.get_branch_namespace() is not None
        branch = cls.get_branch_namespace() is not None
        if not turn or not branch:
            return False
        return True
    
    
    
    
    
    