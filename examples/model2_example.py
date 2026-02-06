import sys
sys.path.append("..")
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatboard.model.model import Model
from chatboard.model.fields import ModelField, KeyField
from chatboard.model.postgres.operations import PostgresOperations
import asyncio


class User(Model):
    # Set namespace name, database type, and versioning
    _namespace_name = "users"
    _db_type = "postgres"
    _is_versioned = True
    
    # Fields
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    age: int = ModelField()


class Post(Model):
    # Set namespace name, database type, and versioning
    _namespace_name = "posts"
    _db_type = "postgres"
    _is_versioned = True
    
    # Fields
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    content: str = ModelField()
    author_id: int = ModelField()


async def main():
    # Initialize versioning tables
    print("Initializing versioning...")
    await PostgresOperations.initialize_versioning()
    
    # Drop existing tables if they exist
    print("Dropping existing tables...")
    from chatboard.model.postgres.builder import SQLBuilder
    await SQLBuilder.drop_many_tables(["users", "posts"])
    
    # Initialize the models (create tables)
    print("Creating tables...")
    await User.initialize()
    await Post.initialize()
    
    # Create a branch
    print("Creating branch...")
    main_branch = await PostgresOperations.create_branch(name="main")
    print(f"Main branch created with ID: {main_branch.id}")
    
    # Create a user in the main branch
    print("Creating user in main branch...")
    user = User(name="John Doe", age=30)
    await user.save(branch=main_branch.id)
    print(f"User created with ID: {user.id}")
    
    # Create a post in the main branch
    print("Creating post in main branch...")
    post = Post(title="Hello World", content="This is my first post", author_id=user.id)
    await post.save(branch=main_branch.id)
    print(f"Post created with ID: {post.id}")
    
    # Commit the turn
    print("Committing turn...")
    await PostgresOperations.commit_turn(main_branch.id, "Initial commit")
    
    # Commit the turn to ensure we have a turn to fork from
    print("Committing turn...")
    turn_id = await PostgresOperations.commit_turn(main_branch.id, "Initial commit")
    print(f"Turn committed with ID: {turn_id}")
    
    # Create a feature branch
    print("Creating feature branch...")
    feature_branch = await PostgresOperations.create_branch(name="feature", forked_from_turn_id=turn_id)
    print(f"Feature branch created with ID: {feature_branch.id}")
    
    # Update the user in the feature branch
    print("Updating user in feature branch...")
    user.age = 31
    await user.save(branch=feature_branch.id)
    print(f"User updated in feature branch")
    
    # Query users from main branch
    print("Querying users from main branch...")
    main_users = await User.query(branch=main_branch.id).filter(name="John Doe").execute()
    print(f"User in main branch: {main_users[0]}")
    
    # Query users from feature branch
    print("Querying users from feature branch...")
    feature_users = await User.query(branch=feature_branch.id).filter(name="John Doe").execute()
    print(f"User in feature branch: {feature_users[0]}")


if __name__ == "__main__":
    asyncio.run(main())