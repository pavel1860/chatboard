import numpy as np
# from scipy.sparse import csr_matrix
from pydantic import AfterValidator, BeforeValidator, BaseModel, PlainSerializer
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler

from typing import TYPE_CHECKING, Annotated, Any, Callable, List, Type
import ast

if TYPE_CHECKING:
    from chatboard.algebra.vectors.base_vectorizer import BaseVectorizer


class Vector(np.ndarray):
    
    
    def __new__(cls, input_array, info=None):
        # Create the ndarray instance
        obj = np.asarray(input_array).view(cls)
        # Add new attribute
        return obj
    

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )

    @staticmethod
    def _validate(value: Any) -> "Vector":
        print(f"value: {value}")
        if isinstance(value, str):
            import ast
            value = np.array(ast.literal_eval(value))
        elif isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise TypeError(f"Invalid type for NDArray: {type(value)}")
        return Vector(value)

    @staticmethod
    def _serialize(instance: "Vector | None") -> list | None:
        if instance is None:
            return None
        return instance.tolist()        
        
        
# class SparseVector(csr_matrix): pass







def transformer(field_name: str | None = None, vectorizer_cls: "Type[BaseVectorizer] | None" = None):
    def decorator(func: Callable):
        func.is_transformer = True
        func._transformer_field_name = field_name
        func._vectorizer_cls = vectorizer_cls
        return func
    return decorator