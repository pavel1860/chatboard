import asyncio
from typing import Optional
from chatboard.model.model import Model, Relation
from chatboard.model.fields import ModelField, KeyField, RelationField
from chatboard.model.namespace_manager import NamespaceManager


class User(Model):
    """User model with posts relation"""
    _namespace_name = "users"
    _db_type = "postgres"
    _is_versioned = True
    
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    age: int = ModelField()
    
    # Define a relation to Post
    posts: Relation["Post"] = RelationField(primary_key="author_id")


class Post(Model):
    """Post model with author_id field"""
    _namespace_name = "posts"
    _db_type = "postgres"
    _is_versioned = True
    
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    content: str = ModelField()
    author_id: int = ModelField()  # Foreign key to User.id


async def main():
    # Initialize the database
    print("Creating tables...")
    await NamespaceManager.create_all_namespaces(versioning=True)
    
    # Create a user
    print("Creating user...")
    user = User(name="John Doe", age=30)
    await user.save()
    print(f"User created with ID: {user.id}")
    
    # Create posts and add them to the user's posts
    print("Creating posts...")
    post1 = Post(title="First Post", content="This is my first post")
    await user.posts.add(post1)
    print(f"Post created with ID: {post1.id}")
    
    post2 = Post(title="Second Post", content="This is my second post")
    await user.posts.add(post2)
    print(f"Post created with ID: {post2.id}")
    
    # Query the user's posts
    print("Querying user's posts...")
    user_posts = await user.posts.all().execute()
    print(f"User has {len(user_posts)} posts:")
    for post in user_posts:
        print(f"- {post.title}: {post.content}")
    
    # Query posts by author_id
    print("Querying posts by author_id...")
    posts = await Post.query().filter(author_id=user.id).execute()
    print(f"Found {len(posts)} posts by author_id:")
    for post in posts:
        print(f"- {post.title}: {post.content}")


if __name__ == "__main__":
    asyncio.run(main())