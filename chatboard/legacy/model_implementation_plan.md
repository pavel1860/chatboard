# Model Implementation Plan

This document outlines the specific implementation steps to create the decorator-based ORM system using the existing files in the `chatboard/model2` directory.

## Files to Modify/Create

1. `chatboard/model2/fields.py` - Implement ModelField and Model base class
2. `chatboard/model2/model.py` - Implement the model factory and decorators
3. `chatboard/model2/namespace_manager.py` - Extend the existing namespace manager
4. `chatboard/model2/postgres/namespace.py` - Extend the PostgresNamespace implementation

## Implementation Details

### 1. Update `chatboard/model2/fields.py`

```python
from pydantic._internal._model_construction import ModelMetaclass
from pydantic import PrivateAttr, create_model, ConfigDict, BaseModel, Field
from typing import Any, Callable, Dict, ForwardRef, Generic, List, Optional, Protocol, Self, Type, TypeVar, get_args, get_origin

def ModelField(
    default: Any = None,
    *,
    index: Optional[str] = None,
    **kwargs
) -> Any:
    """Define a model field with ORM-specific metadata"""
    # Create extra metadata for the field
    extra = kwargs.pop("json_schema_extra", {}) or {}
    extra["index"] = index
    
    # Create the field with the extra metadata
    return Field(default, json_schema_extra=extra, **kwargs)

class ModelMeta(ModelMetaclass, type):
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)
        # We don't need to do anything special here anymore
        # The decorator will handle the ORM functionality
        return cls_obj

class Model(BaseModel, metaclass=ModelMeta):
    """Base class for all models"""
    # This class is now much simpler - it's just a Pydantic model
    # The ORM functionality will be added by the decorator
    pass

class VectorModel(Model):
    """Model with vector embedding support"""
    pass

class VersionedModel(Model):
    """Model with versioning support"""
    pass
```

### 2. Create `chatboard/model2/model.py`

```python
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from pydantic import BaseModel

from chatboard.model2.namespace_manager import NamespaceManager
from chatboard.model2.base_namespace import DatabaseType

T = TypeVar('T', bound=BaseModel)

class ModelFactory:
    """Factory for creating model decorators"""
    
    def postgres(self, namespace: Optional[str] = None, **options):
        """Create a PostgreSQL model decorator"""
        return self._create_decorator("postgres", namespace, **options)
    
    def qdrant(self, namespace: Optional[str] = None, **options):
        """Create a Qdrant model decorator"""
        return self._create_decorator("qdrant", namespace, **options)
    
    def _create_decorator(self, db_type: DatabaseType, namespace: Optional[str] = None, **options):
        """Create a model decorator for the specified database type"""
        def decorator(cls: Type[T]) -> Type[T]:
            # Get model name and namespace
            model_name = cls.__name__
            namespace_name = namespace or f"{model_name.lower()}s"
            
            # Build namespace
            ns = NamespaceManager.build_namespace(model_name, db_type)
            
            # Process fields
            for field_name, field_info in cls.model_fields.items():
                # Extract field metadata
                extra = field_info.json_schema_extra or {}
                
                # Add field to namespace
                ns.add_field(field_name, field_info.annotation, extra)
            
            # Add namespace reference to the class
            setattr(cls, '_namespace_name', namespace_name)
            
            # Add ORM methods to the class
            @classmethod
            async def initialize(cls):
                """Initialize the model (create table)"""
                ns = NamespaceManager.get_namespace(cls._namespace_name)
                await ns.create_namespace()
            
            async def save(self):
                """Save the model instance to the database"""
                ns = NamespaceManager.get_namespace(self.__class__._namespace_name)
                data = self.model_dump()
                result = await ns.save(data)
                # Update instance with returned data (e.g., ID)
                for key, value in result.items():
                    setattr(self, key, value)
                return self
            
            @classmethod
            async def get(cls, id: Any):
                """Get a model instance by ID"""
                ns = NamespaceManager.get_namespace(cls._namespace_name)
                data = await ns.get(id)
                return cls(**data) if data else None
            
            @classmethod
            def query(cls):
                """Create a query for this model"""
                ns = NamespaceManager.get_namespace(cls._namespace_name)
                return ns.query()
            
            # Add the methods to the class
            cls.initialize = initialize
            cls.save = save
            cls.get = get
            cls.query = query
            
            return cls
        
        return decorator

# Create a singleton instance
model = ModelFactory()
```

### 3. Update `chatboard/model2/namespace_manager.py`

The existing file already has a good structure, but we need to add a few methods:

```python
from chatboard.model2.base_namespace import DatabaseType, Namespace
from typing import TYPE_CHECKING, Type, TypeVar, Dict, Any

from chatboard.model2.postgres.namespace import PostgresNamespace

if TYPE_CHECKING:
    from chatboard.model2.fields import Model

MODEL = TypeVar("MODEL", bound="Model")   

class NamespaceManager:
    # _instance: "NamespaceManager | None" = None
    _namespaces: dict[str, Namespace] = {}
    
    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance
    
    @classmethod
    def initialize(cls):
        cls._namespaces = {}
        
    @classmethod
    def build_namespace(cls, model_name: str, db_type: DatabaseType) -> Namespace:
        """
        Build a namespace for a model.
        """
        if not cls._namespaces:
            cls.initialize()
        if db_type == "qdrant":
            raise NotImplementedError("Qdrant is not implemented")
        elif db_type == "postgres":
            namespace = PostgresNamespace(model_name)            
        else:
           raise ValueError(f"Invalid database type: {db_type}")        
        cls._namespaces[model_name] = namespace
        return namespace
        
    @classmethod
    def get_namespace(cls, model_name: str) -> Namespace:
        if not cls._namespaces:
            raise ValueError("NamespaceManager not initialized")
        if not model_name in cls._namespaces:
            raise ValueError(f"Namespace for model {model_name} not found")
        return cls._namespaces[model_name]
    
    @classmethod
    async def save(cls, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save data to a namespace"""
        namespace = cls.get_namespace(model_name)
        return await namespace.save(data)
    
    @classmethod
    async def get(cls, model_name: str, id: Any) -> Dict[str, Any]:
        """Get data from a namespace by ID"""
        namespace = cls.get_namespace(model_name)
        return await namespace.get(id)
    
    @classmethod
    async def query(cls, model_name: str, **kwargs):
        """Create a query for a namespace"""
        namespace = cls.get_namespace(model_name)
        return namespace.query(**kwargs)
```

### 4. Update `chatboard/model2/postgres/namespace.py`

The existing file needs to be extended with the actual database operations:

```python
from typing import Any, Dict, Literal, Type
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from chatboard.model2.postgres.builder import SQLBuilder
from chatboard.utils.model_utils import get_list_type, is_list_type
from chatboard.model2.base_namespace import Namespace, NSFieldInfo
from chatboard.utils.db_connections import PGConnectionManager
import datetime as dt

PgIndexType = Literal["btree", "hash", "gin", "gist", "spgist", "brin"]

class PgFieldInfo(NSFieldInfo[PgIndexType]):
    pass

class PostgresNamespace(Namespace):
    def __init__(self, name: str):
        super().__init__(name)
        
    @property
    def table_name(self) -> str:
        return self.name
        
    def add_field(
        self, 
        name: str, 
        field_type: type[Any],
        extra: dict[str, Any] | None = None,
    ):
        """
        Add a field to the namespace.
        """
        db_field_type = SQLBuilder.map_field_to_sql_type(field_type, extra)

        self._fields[name] = PgFieldInfo(
            name=name,
            field_type=field_type,
            db_field_type=db_field_type,
            index=extra.get("index") if extra else None,
            extra=extra,
        )
        
    def add_relation(
        self, 
        name: str, 
        field_info: FieldInfo,
    ):
        """Add a relation to the namespace"""
        # Implementation for relations
        pass

    async def create_namespace(self):
        """Create the namespace in the database"""
        return await SQLBuilder.create_table(self)
        
    async def drop_namespace(self):
        """Drop the namespace from the database"""
        return await SQLBuilder.drop_table(self)
    
    async def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save data to the namespace"""
        # Implementation for saving data
        # This would use SQLBuilder to generate and execute INSERT/UPDATE SQL
        # For now, just return the data with an ID
        return {"id": 1, **data}
    
    async def get(self, id: Any) -> Dict[str, Any]:
        """Get data from the namespace by ID"""
        # Implementation for getting data
        # This would use SQLBuilder to generate and execute SELECT SQL
        # For now, just return dummy data
        return {"id": id, "name": "Example", "age": 30}
    
    def query(self, **kwargs):
        """Create a query for this namespace"""
        # Implementation for creating a query
        # This would return a QuerySet for this namespace
        from chatboard.model2.postgres.query import PostgresQuerySet
        return PostgresQuerySet(self, **kwargs)
```

### 5. Create `chatboard/model2/postgres/query.py`

```python
from typing import Any, Dict, List, Optional

class PostgresQuerySet:
    """PostgreSQL implementation of QuerySet"""
    
    def __init__(self, namespace, **kwargs):
        self.namespace = namespace
        self.filters = kwargs
        self.limit_value = None
        self.offset_value = None
        self.order_by_value = None
    
    def filter(self, **kwargs):
        """Filter the query"""
        self.filters.update(kwargs)
        return self
    
    def limit(self, limit: int):
        """Limit the query results"""
        self.limit_value = limit
        return self
    
    def offset(self, offset: int):
        """Set the offset for the query"""
        self.offset_value = offset
        return self
    
    def order_by(self, field: str, ascending: bool = True):
        """Set the order for the query"""
        self.order_by_value = (field, ascending)
        return self
    
    async def execute(self) -> List[Dict[str, Any]]:
        """Execute the query"""
        # Implementation would execute the query
        # This would use SQLBuilder to generate and execute SELECT SQL
        # For now, just return dummy data
        return [{"id": 1, "name": "Example", "age": 30}]
    
    async def first(self) -> Optional[Dict[str, Any]]:
        """Get the first result"""
        self.limit_value = 1
        results = await self.execute()
        return results[0] if results else None
    
    async def count(self) -> int:
        """Count the results"""
        # Implementation would count the results
        # This would use SQLBuilder to generate and execute COUNT SQL
        # For now, just return a dummy count
        return 1
    
    def __await__(self):
        """Make the query awaitable"""
        return self.execute().__await__()
```

## Usage Example

Once implemented, the usage would look like:

```python
from pydantic import BaseModel
from chatboard.model2.fields import Model, ModelField
from chatboard.model2.model import model

@model.postgres(namespace="users")
class User(Model):
    id: int = None
    name: str = ModelField(index="btree")
    age: int = ModelField()

# Initialize the model (create table)
await User.initialize()

# Create and save a user
user = User(name="John", age=30)
await user.save()

# Get a user by ID
user = await User.get(1)

# Query users
users = await User.query().filter(age__gt=25).limit(10).execute()
```

## Implementation Steps

1. Start by implementing the `fields.py` file with the `ModelField` function and simplified `Model` class
2. Create the `model.py` file with the `ModelFactory` class and decorators
3. Update the `namespace_manager.py` file with the additional methods
4. Extend the `postgres/namespace.py` file with the database operations
5. Create the `postgres/query.py` file for the query system
6. Test the implementation with a simple example

## Future Enhancements

1. Add support for vector operations
2. Implement versioning
3. Add support for relations
4. Create migration utilities from the old system
5. Add support for more database types (e.g., Qdrant)