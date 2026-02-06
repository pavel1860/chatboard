# chatboard Model Documentation

The `Model` class is the core component of chatboard's data layer, providing a Pydantic-based ORM system with advanced vector database capabilities. It supports multiple database backends, including Qdrant (vector database) and PostgreSQL, with features for vector search, filtering, relations, and more.

## Table of Contents
- [Basic Model Definition](#basic-model-definition)
- [Model Configuration](#model-configuration)
- [CRUD Operations](#crud-operations)
- [Vector Search](#vector-search)
- [Filtering and Querying](#filtering-and-querying)
- [Relations](#relations)
- [Advanced Features](#advanced-features)

## Basic Model Definition

Models are defined by subclassing the `Model` class and defining fields with appropriate types and field definitions:

```python
from chatboard.model import Model, ModelField, IndexType
from chatboard.model.vectors.openai_vectorizer import OpenAISmallVectorizer
import datetime as dt

class BasicTestModel(Model):
    created_at: dt.datetime = ModelField(auto_now_add=True)
    updated_at: dt.datetime = ModelField(auto_now=True)
    topic: str = ModelField(index=IndexType.Text)
    content: str = ModelField()
    uuid: str = ModelField(index=IndexType.Uuid)
    
    class VectorSpace:
        dense: OpenAISmallVectorizer
```

Each field can be configured with various options:
- `ModelField` allows you to define field properties like auto timestamps and indices
- `IndexType` allows you to specify how fields should be indexed (Text, Uuid, Integer, etc.)

## Model Configuration

Models can be configured using the `Config` class:

```python
class Post(Model):
    title: str = ModelField()
    content: str = ModelField()
    user_id: int = ModelField(default=None)
    
    class Config:
        database_type = "postgres"  # Use PostgreSQL instead of Qdrant
        # versioned = False         # Enable/disable versioning
        # is_head = False           # Mark as a "head" model
        # namespace = "custom_name" # Custom namespace (defaults to snakecased plural)
```

## CRUD Operations

### Creating and Saving Models

```python
# Create a new model instance
model = BasicTestModel(
    topic="animals", 
    content="I like turtles",
    uuid="1234"
)

# Save to database - this sets the ID and handles vector embeddings
await model.save()
```

### Retrieving Models

```python
# Get a model by ID
model = await BasicTestModel.get("model_id")

# Get multiple models by IDs
models = await BasicTestModel.get_many(["id1", "id2"])

# Get all models (with limit)
all_models = await BasicTestModel.limit(10)

# Get the first or last model (based on default temporal field)
first = await BasicTestModel.first()
last = await BasicTestModel.last()
```

### Deleting Models

```python
# Delete a single model by ID
await BasicTestModel.delete("model_id")

# Delete multiple models by IDs
await BasicTestModel.batch_delete(["id1", "id2"])

# Delete models matching a filter
await BasicTestModel.batch_delete(filters=lambda x: x.topic == "animals")
```

## Vector Search

Models can be searched using vector similarity, powered by the configured vectorizer:

```python
# Find models similar to a query string
similar_models = await BasicTestModel.similar("I like cats").limit(5)

# Control which vector spaces to use (if you have multiple)
similar_models = await BasicTestModel.similar("I like cats", vec=["dense"]).limit(5)

# Apply additional filters to vector search
similar_models = await BasicTestModel.similar("I like cats").filter(lambda x: x.topic == "animals")
```

## Filtering and Querying

Models can be filtered using a lambda-based filter syntax:

```python
# Simple equality filter
animals = await BasicTestModel.filter(lambda x: x.topic == "animals")

# OR condition
animals_or_physics = await BasicTestModel.filter(
    lambda x: (x.topic == "animals") | (x.topic == "physics")
)

# Date/time comparisons
recent = await BasicTestModel.filter(
    lambda x: x.created_at > dt.datetime(2021, 1, 1)
)

# Complex conditions
specific_range = await BasicTestModel.filter(
    lambda x: (x.created_at >= dt.datetime(2021, 1, 1)) & 
              (x.created_at < dt.datetime(2024, 1, 1))
)
```

### Ordering Results

```python
# Order by a field ascending
ordered_asc = await BasicTestModel.limit(20).order_by("order", ascending=True)

# Order by a field descending
ordered_desc = await BasicTestModel.limit(20).order_by("order", ascending=False)

# Combine with filters
filtered_ordered = await BasicTestModel.filter(
    lambda x: x.topic == "movies"
).order_by("order", ascending=False)
```

## Relations

You can define relations between models:

```python
class Post(Model):
    title: str = ModelField()
    content: str = ModelField()
    user_id: int = ModelField(default=None)
    
    class Config:
        database_type = "postgres"

class User(Model):
    name: str = ModelField()
    age: int = ModelField()
    posts: Relation[Post] = ModelRelation(key="user_id")
    
    class Config:
        database_type = "postgres"
```

Working with relations:

```python
# Add a related model
user = await User(name="John Doe", age=30).save()
await user.posts.add(Post(title="My Post", content="This is my post"))

# Query related models
user = await User.get("user_id")
user_posts = await user.posts.limit(10)

# Filter related models
recent_posts = await user.posts.filter(lambda x: x.created_at > some_date)
```

## Advanced Features

### Batch Operations

```python
# Batch create/update
points = [
    BasicTestModel(topic="animals", content="turtles are slow", uuid="1234"),
    BasicTestModel(topic="physics", content="quantum mechanics is weird", uuid="5678")
]
await BasicTestModel.batch_upsert(points)

# Batch delete
await BasicTestModel.batch_delete(["id1", "id2", "id3"])
```

### Fusion Queries

For models with multiple vector spaces, you can perform fusion queries:

```python
# RRF fusion across multiple vector queries
results = await Model.fusion(
    "cat query", 
    "dog query",
    type="rrf"
).limit(10)
```

### Model Copy

```python
# Create a copy of a model instance
model_copy = original_model.copy(with_ids=False)
```

### Namespace Management

```python
# Delete a model's entire namespace
await BasicTestModel.delete_namespace()

# Initialize all model namespaces
await connection_manager.init_all_namespaces()

# Drop all namespaces
await connection_manager.drop_all_namespaces()
```
