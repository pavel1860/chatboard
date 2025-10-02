from typing import Any, Type
from pydantic._internal._model_construction import ModelMetaclass

from promptview.model.base.types import VersioningStrategy
from .namespace_manager2 import NamespaceManager

class ModelMeta(ModelMetaclass, type):
    """
    Metaclass for all ORM models.
    - Defers field and relation parsing until NamespaceManager.finalize().
    - Sets up backend namespaces and model attributes.
    """
    def __new__(cls, name, bases, dct):
        # Skip base/abstract models
        if dct.get("_is_base", False) or name in ("Model", "RelationModel"):
            return super().__new__(cls, name, bases, dct)

        db_type = dct.get("_db_type", "postgres")
        model_name = name
        namespace_name = dct.get("_namespace_name") or cls._default_namespace_name(model_name, db_type)
        versioning_strategy = dct.get("_versioning_strategy", VersioningStrategy.NONE)
        # Create the actual Pydantic model class
        cls_obj = super().__new__(cls, name, bases, dct)

        # Check if namespace exists
        # existing_ns = NamespaceManager.get_namespace_or_none(namespace_name, db_type)
        # if existing_ns and getattr(existing_ns, "_model_cls", None):
        #     existing_ns.set_model_class(cls_obj)
        #     NamespaceManager.register_model_namespace(cls_obj, existing_ns)
        #     cls_obj._namespace_name = namespace_name
        #     cls_obj._is_versioned = dct.get("_is_versioned", False)
        #     cls_obj._field_extras = {}
        #     cls_obj._relations = {}
        #     return cls_obj

        # Otherwise → build namespace
        ns = NamespaceManager.build_namespace(
            model_name=namespace_name,
            db_type=db_type,
            is_versioned=dct.get("_is_versioned", False),
            is_context=dct.get("_is_context", False),
            is_repo=dct.get("_is_repo", False),
            is_artifact=dct.get("_is_artifact", False),
            repo_namespace=dct.get("_repo", None)
        )

        # Attach parsers for deferred execution
        from .field_parser import FieldParser
        from .relation_parser import RelationParser

        field_parser = FieldParser(
            model_cls=cls_obj,
            model_name=model_name,
            db_type=db_type,
            namespace=ns,
            reserved_fields=set()
        )
        relation_parser = RelationParser(cls_obj, ns)
        
        ns.set_pending_parsers(field_parser, relation_parser)

        # Register the namespace but do not parse yet
        ns.set_model_class(cls_obj)
        NamespaceManager.register_model_namespace(cls_obj, ns)

        cls_obj._namespace_name = namespace_name
        cls_obj._is_versioned = dct.get("_is_versioned", False)
        cls_obj._field_extras = {}
        cls_obj._relations = {}

        return cls_obj

    @staticmethod
    def _default_namespace_name(model_cls_name: str, db_type: str) -> str:
        from ..utils.string_utils import camel_to_snake
        name = camel_to_snake(model_cls_name)
        if db_type == "postgres":
            name += "s"
        return name
