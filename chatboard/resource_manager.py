





from typing import Type

from .algebra.vectors.base_vectorizer import BaseVectorizer

VectorizerName = str

class ResourceManager:
    
    _instantiated_vectorizers: dict[VectorizerName, BaseVectorizer] = {}
    # _field_vectorizers: dict[str, BaseVectorizer] = {}
    
    @classmethod
    def register_vectorizer(cls, vectorizer_cls: Type[BaseVectorizer]):        
        vectorizer = cls._instantiated_vectorizers.get(vectorizer_cls.__name__)
        if vectorizer is None:
            vectorizer = vectorizer_cls()
            cls._instantiated_vectorizers[vectorizer_cls.__name__] = vectorizer
        return vectorizer
        # cls._field_vectorizers[name] = vectorizer
        
    @classmethod  
    def get_vectorizer_by_name(cls, name: str) -> BaseVectorizer:
        return cls._field_vectorizers[name]
    
    @classmethod
    def get_vectorizer_by_cls(cls, vectorizer_cls: Type[BaseVectorizer]) -> BaseVectorizer:
        return cls._instantiated_vectorizers[vectorizer_cls.__name__]
    
    