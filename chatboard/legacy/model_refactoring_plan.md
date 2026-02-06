# Model Refactoring Plan

## Current Issues

The current model architecture in `chatboard/model/` has several issues:

1. **Complex Metaclass Implementation**: The `ModelMeta` class handles too many responsibilities
2. **Tight Coupling**: The `Model` class is tightly coupled with database operations
3. **Poor Separation of Concerns**: Database operations, model definition, and query operations are mixed
4. **Extensibility Issues**: Adding new database types requires changes in multiple places
5. **Complex Query System**: The query system has many special cases and complex logic

## Proposed Solution

Move from a metaclass-based approach to a decorator-based pattern:

```python
from pydantic import BaseModel
from chatboard import model


@model.postgres()
class User(BaseModel):
    name: str = ModelField(index="btree")
    age: int = ModelField()
```

This approach will:
- Separate model definition from ORM functionality
- Provide a cleaner, more intuitive API
- Improve extensibility and maintainability
- Reduce complexity and "magic"

## Next Steps

1. Design the decorator interface
2. Implement the core model factory
3. Create the namespace system
4. Build the query system
5. Add support for advanced features (versioning, vectors, relations)

## Detailed Design

### 1. Decorator Interface

The decorator interface will be the main entry point for defining models:

```python
# Module structure
from chatboard.model import model, ModelField

# Database-specific decorators
@model.postgres(
    namespace="users",  # Optional: override table name
    versioned=False,    # Optional: enable versioning
    schema="public"     # Optional: database-specific options
)
class User(BaseModel):
    id: int = None
    name: str = ModelField(index="btree")
    age: int = ModelField()
    
    class Config:
        # Pydantic config options still work
        validate_assignment = True
```

The decorator will:
1. Register the model with the namespace system
2. Process field definitions and extract metadata
3. Return an enhanced class that inherits from both the original class and a base ORM class

### 2. Field Definition System

Fields will be defined using a `ModelField` function that extends Pydantic's `Field`:

```python
# Basic field with index
name: str = ModelField(index="btree")

# the ModelField should return Field from pydantic and put all the added fields in json_schema_extra.
```

### 3. Revised Implementation with Namespace Manager

Let's implement a better version that uses a NamespaceManager:

```python
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

# Simple implementation of ModelField
def ModelField(
    default: Any = None,
    *,
    index: Optional[str] = None,
    **kwargs
) -> Any:
    """Define a model field with ORM-specific metadata"""
    # Create extra metadata for the field
    extra = {
        "index": index,
    }
    
    # Create the field with the extra metadata
    return Field(default, json_schema_extra=extra, **kwargs)


# Namespace system
class Namespace:
    """Base namespace interface"""
    name: str
    fields: Dict[str, Dict[str, Any]]
    
    def __init__(self, name: str):
        self.name = name
        self.fields = {}
    
    def add_field(self, name: str, field_type: Type, extra: Dict[str, Any] = None):
        """Add a field to the namespace"""
        self.fields[name] = {
            'type': field_type,
            'extra': extra or {},
        }
    
    async def create(self):
        """Create the namespace in the database"""
        raise NotImplementedError
    
    async def drop(self):
        """Drop the namespace from the database"""
        raise NotImplementedError
    
    async def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save data to the namespace"""
        raise NotImplementedError
    
    async def get(self, id: Any) -> Dict[str, Any]:
        """Get data from the namespace by ID"""
        raise NotImplementedError
    
    def query(self):
        """Create a query for this namespace"""
        raise NotImplementedError


class PostgresNamespace(Namespace):
    """PostgreSQL implementation of Namespace"""
    
    async def create(self):
        """Create the table in PostgreSQL"""
        # Implementation would create the table based on fields
        print(f"Creating PostgreSQL table {self.name}")
        
        # Build SQL CREATE TABLE statement
        columns = []
        for name, info in self.fields.items():
            field_type = info['type']
            extra = info['extra']
            
            # Map Python type to SQL type
            sql_type = self._map_type_to_sql(field_type)
            
            # Add index if specified
            index = extra.get('index')
            index_str = f" {index}" if index else ""
            
            columns.append(f"{name} {sql_type}{index_str}")
        
        # Create table SQL
        sql = f"CREATE TABLE IF NOT EXISTS {self.name} (\n"
        sql += ",\n".join(columns)
        sql += "\n);"
        
        print(f"SQL: {sql}")
        # In a real implementation, this would execute the SQL
        
    def _map_type_to_sql(self, field_type: Type) -> str:
        """Map Python type to SQL type"""
        # Simple mapping for demonstration
        if field_type == int:
            return "INTEGER"
        elif field_type == float:
            return "FLOAT"
        elif field_type == str:
            return "TEXT"
        elif field_type == bool:
            return "BOOLEAN"
        else:
            return "JSONB"  # Default to JSONB for complex types
    
    async def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save data to PostgreSQL"""
        # Implementation would save data to PostgreSQL
        print(f"Saving to PostgreSQL table {self.name}: {data}")
        # In a real implementation, this would execute an INSERT or UPDATE
        return {"id": 1, **data}  # Return with ID
    
    async def get(self, id: Any) -> Dict[str, Any]:
        """Get data from PostgreSQL"""
        # Implementation would fetch from PostgreSQL
        print(f"Getting from PostgreSQL table {self.name} with ID {id}")
        # In a real implementation, this would execute a SELECT
        return {"id": id, "name": "Example", "age": 30}  # Example data
    
    def query(self):
        """Create a query for this namespace"""
        # Implementation would return a QuerySet
        return PostgresQuerySet(self)


class NamespaceManager:
    """Manager for namespaces"""
    _namespaces: Dict[str, Namespace] = {}
    
    @classmethod
    def get_namespace(cls, name: str) -> Namespace:
        """Get a namespace by name"""
        if name not in cls._namespaces:
            raise ValueError(f"Namespace {name} not found")
        return cls._namespaces[name]
    
    @classmethod
    def create_namespace(cls, name: str, db_type: str) -> Namespace:
        """Create a new namespace"""
        if name in cls._namespaces:
            return cls._namespaces[name]
        
        if db_type == "postgres":
            namespace = PostgresNamespace(name)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        cls._namespaces[name] = namespace
        return namespace


# Query system
class QuerySet:
    """Base query set"""
    
    def filter(self, **kwargs):
        """Filter the query"""
        raise NotImplementedError
    
    def limit(self, limit: int):
        """Limit the query results"""
        raise NotImplementedError
    
    async def execute(self):
        """Execute the query"""
        raise NotImplementedError


class PostgresQuerySet(QuerySet):
    """PostgreSQL implementation of QuerySet"""
    
    def __init__(self, namespace: PostgresNamespace):
        self.namespace = namespace
        self.filters = {}
        self.limit_value = None
    
    def filter(self, **kwargs):
        """Filter the query"""
        self.filters.update(kwargs)
        return self
    
    def limit(self, limit: int):
        """Limit the query results"""
        self.limit_value = limit
        return self
    
    async def execute(self):
        """Execute the query"""
        # Implementation would execute the query
        print(f"Executing query on {self.namespace.name} with filters {self.filters} and limit {self.limit_value}")
        # In a real implementation, this would execute a SELECT with WHERE and LIMIT
        return [{"id": 1, "name": "Example", "age": 30}]  # Example data


# Model factory
class ModelFactory:
    """Factory for creating model decorators"""
    
    def postgres(self, namespace: Optional[str] = None, **options):
        """Create a PostgreSQL model decorator"""
        def decorator(cls: Type[T]) -> Type[T]:
            # Get or create namespace
            model_name = cls.__name__
            namespace_name = namespace or f"{model_name.lower()}s"
            ns = NamespaceManager.create_namespace(namespace_name, "postgres")
            
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
                await ns.create()
            
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

### 4. Usage Example

With this implementation, the usage would look like:

```python
from pydantic import BaseModel
from chatboard.model import model, ModelField

@model.postgres(namespace="users")
class User(BaseModel):
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

### 5. Key Improvements in This Design

1. **Clear Separation of Concerns**:
   - Model definition (Pydantic models)
   - Namespace management (NamespaceManager)
   - Database operations (Namespace implementations)
   - Query building (QuerySet)

2. **No Direct Database Access in Models**:
   - All database operations go through the namespace
   - Models don't need to know about database details

3. **Extensibility**:
   - Easy to add new database types by implementing new Namespace classes
   - Clear extension points for new features

4. **Simplified Field Definition**:
   - ModelField is a thin wrapper around Pydantic's Field
   - All metadata is stored in json_schema_extra

5. **Explicit Initialization**:
   - Tables are created explicitly with initialize()
   - No "magic" happening behind the scenes

### 6. Next Steps

1. Implement the full namespace system with actual database operations
2. Add support for more field types (vectors, relations)
3. Implement a more robust query system
4. Add versioning support
5. Create migration utilities from the old system
