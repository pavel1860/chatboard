# Model Versioning Implementation Plan

## Overview

We need to add versioning to the new model system, similar to how it works in `artifact_log3.py`, but with some changes:

1. Change "Head" to "Repo" (no context manager)
2. Update the API for saving to be `Message(...).save(branch=branch_id)`
3. Update the query API to be `User.query(branch=branch_id).filter(name="John Doe").execute()`
4. Implement native branching operations in `postgres/operations.py`

## Database Schema

We'll need to create the following tables:

1. `branches` - Stores information about branches
   - `id` - Primary key
   - `name` - Branch name
   - `created_at` - Creation timestamp
   - `updated_at` - Last update timestamp
   - `forked_from_turn_index` - Index of the turn this branch was forked from
   - `forked_from_branch_id` - ID of the branch this branch was forked from

2. `turns` - Stores information about turns (commits)
   - `id` - Primary key
   - `created_at` - Creation timestamp
   - `ended_at` - End timestamp
   - `index` - Turn index within the branch
   - `status` - Turn status (staged, committed, reverted)
   - `message` - Commit message
   - `branch_id` - Branch ID
   - `metadata` - Additional metadata (JSON)

3. `repos` - Stores information about repositories (replaces heads)
   - `id` - Primary key
   - `created_at` - Creation timestamp
   - `updated_at` - Last update timestamp
   - `main_branch_id` - ID of the main branch
   - `branch_id` - ID of the current branch
   - `turn_id` - ID of the current turn

## Implementation Steps

### 1. Create Enum and Model Classes

Create the necessary enum and model classes for versioning:

```python
class TurnStatus(enum.Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    REVERTED = "reverted"

class Turn(BaseModel):
    id: int
    created_at: datetime
    ended_at: datetime | None = None    
    index: int
    status: TurnStatus    
    message: str | None = None
    branch_id: int
    metadata: dict | None = None

class Branch(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    forked_from_turn_index: int | None = None
    forked_from_branch_id: int | None = None
    last_turn: Turn | None = None

class Repo(BaseModel):
    id: int
    main_branch_id: int
    branch_id: int
    turn_id: int
    created_at: datetime
    updated_at: datetime
```

### 2. Modify PostgresOperations

Update the `PostgresOperations` class to handle branching operations:

1. Add methods to create and manage branches and turns
2. Modify the `save` method to support saving to a specific branch
3. Modify the `query` method to support querying from a specific branch

```python
class PostgresOperations:
    # Existing methods...
    
    @classmethod
    async def initialize_versioning(cls):
        """Initialize versioning tables"""
        # Create branches, turns, and repos tables
        
    @classmethod
    async def create_branch(cls, name: str = None, forked_from_turn_id: int = None):
        """Create a new branch"""
        # Implementation...
        
    @classmethod
    async def create_turn(cls, branch_id: int, index: int, status: TurnStatus):
        """Create a new turn"""
        # Implementation...
        
    @classmethod
    async def create_repo(cls, init_repo: bool = True):
        """Create a new repo"""
        # Implementation...
        
    @classmethod
    async def save(cls, namespace: "PostgresNamespace", data: Dict[str, Any], branch_id: int = None) -> Dict[str, Any]:
        """Save data to the database with versioning support"""
        # Implementation...
        
    @classmethod
    async def query(cls, namespace: "PostgresNamespace", branch_id: int = None, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query records with versioning support"""
        # Implementation...
```

### 3. Update Model Class

Modify the `Model` class to support versioning:

```python
class Model(BaseModel, metaclass=ModelMeta):
    # Existing attributes...
    
    async def save(self, branch: int = None):
        """Save the model instance to the database with optional branch"""
        ns = NamespaceManager.get_namespace(self.__class__.get_namespace_name())
        data = self.model_dump()
        result = await ns.save(data, branch)
        # Update instance with returned data
        for key, value in result.items():
            setattr(self, key, value)
        return self
    
    @classmethod
    def query(cls, branch: int = None):
        """Create a query for this model with optional branch"""
        ns = NamespaceManager.get_namespace(cls.get_namespace_name())
        return ns.query(branch)
```

### 4. Update Namespace Classes

Update the `PostgresNamespace` class to support versioning:

```python
class PostgresNamespace(Namespace):
    # Existing methods...
    
    async def save(self, data: Dict[str, Any], branch: int = None) -> Dict[str, Any]:
        """Save data to the namespace with optional branch"""
        return await PostgresOperations.save(self, data, branch)
    
    def query(self, branch: int = None) -> QuerySet:
        """Create a query for this namespace with optional branch"""
        return PostgresQuerySet(self, branch)
```

### 5. Update QuerySet Class

Update the `QuerySet` class to support versioning:

```python
class PostgresQuerySet:
    def __init__(self, namespace: PostgresNamespace, branch: int = None):
        self.namespace = namespace
        self.branch = branch
        self.filters = {}
        self.limit_value = None
        
    # Existing methods...
    
    async def execute(self) -> List[Dict[str, Any]]:
        """Execute the query"""
        return await PostgresOperations.query(
            self.namespace, 
            branch_id=self.branch,
            filters=self.filters, 
            limit=self.limit_value
        )
```

## Implementation Details

### Versioning Logic

1. When a model is saved with a branch ID, it's associated with the current turn of that branch
2. When querying with a branch ID, we retrieve all records associated with turns in that branch's history
3. The most recent version of a record (based on turn index) is returned when querying

### Repo Management

1. A repo points to a specific branch and turn
2. When saving or querying without a branch ID, the current branch and turn of the repo are used
3. Repos can be created, updated, and deleted

### Branch Operations

1. Branches can be created from existing turns
2. Branches can be checked out (making them the current branch of a repo)
3. Branches can be merged (not implemented in this plan)

## Example Usage

```python
# Create a model
user = User(name="John Doe", age=30)

# Save to the default branch
await user.save()

# Save to a specific branch
await user.save(branch=2)

# Query from the default branch
users = await User.query().filter(name="John Doe").execute()

# Query from a specific branch
users = await User.query(branch=2).filter(name="John Doe").execute()