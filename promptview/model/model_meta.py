from typing import TYPE_CHECKING, Any, Type
from pydantic._internal._model_construction import ModelMetaclass

from promptview.model.base.types import VersioningStrategy
from .namespace_manager2 import NamespaceManager

if TYPE_CHECKING:
    from .base.base_field_info import BaseFieldInfo
    from .sql2.relations import RelField



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
        # Process transformer decorators BEFORE creating the class
        # (because they need to be registered before field parsing)
        transformer_funcs = {}
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and hasattr(attr_value, "_transformer_field_name"):
                field_name = getattr(attr_value, "_transformer_field_name")
                vectorizer_cls = getattr(attr_value, "_vectorizer_cls")
                transformer_funcs[field_name] = (attr_value, vectorizer_cls)

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

        # Register transformers with namespace
        for field_name, (transform_func, vectorizer_cls) in transformer_funcs.items():
            ns.register_transformer(field_name, transform_func, vectorizer_cls)

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

    # def __getattr__(cls, name) -> Any:
    #     """
    #     This method is called for EVERY attribute access on the class!
    #     Be careful - you must use super().__getattribute__ to avoid infinite recursion.
    #     """
    #     from .sql2.relations import RelField, NsRelation
    #     from .namespace_manager2 import NamespaceManager
    #     # Get the _columns dict (must use super to avoid recursion)
    #     if name.startswith("_"):
    #         return super().__getattribute__(name)
        
        
    #     columns = super().__getattribute__('model_fields')    
        
    #     # If it's a column, return a ColumnExpression
    #     if name in columns:
    #         # print(name)
    #         try:
    #             ns = cls.get_namespace()
    #             return RelField(NsRelation(ns), name, columns[name])
    #         except:
    #             return super().__getattribute__(name)
    #         # return super().__getattribute__(name)
        
    #     # Otherwise, return the normal attribute
    #     return super().__getattribute__(name)
    
    def __getattr__(cls, name) -> Any:
        """
        This method is called for EVERY attribute access on the class!
        Be careful - you must use super().__getattribute__ to avoid infinite recursion.
        """
        from .sql2.relations import RelField, NsRelation
        from .namespace_manager2 import NamespaceManager
        
        # Don't intercept any attributes starting with __ (Pydantic internals and special methods)
        # These should be handled by the parent metaclass (ModelMetaclass) through normal
        # Python attribute resolution. By raising AttributeError here, we prevent that.
        # Instead, we should just not handle them at all - but since __getattr__ is only
        # called when the attribute doesn't exist, we need to raise AttributeError.
        # The key is to raise it in a way that doesn't interfere with Pydantic's handling.
        if name.startswith("__"):
            # For __ attributes, don't intercept - let Python's normal resolution work
            # This means we should raise AttributeError, but in a way that allows
            # the parent metaclass to potentially handle it via __getattribute__
            # Actually, the parent's __getattribute__ won't be called from __getattr__
            # So we need to manually check the parent metaclass
            # Try to get it from ModelMetaclass directly
            try:
                return ModelMetaclass.__getattribute__(cls, name)
            except AttributeError:
                # If that fails, raise normally - this will let Pydantic handle it
                raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")
        
        # For regular attributes, check if it's a model field
        try:
            columns = super().__getattribute__('model_fields')
        except AttributeError:
            # If model_fields doesn't exist yet, fall back to normal lookup
            return super().__getattribute__(name)
        
        # If it's a column, return a ColumnExpression
        if name in columns:
            try:
                ns = cls.get_namespace()
                return RelField(NsRelation(ns), name, columns[name])
            except:
                return super().__getattribute__(name)
        
        # Otherwise, return the normal attribute
        return super().__getattribute__(name)