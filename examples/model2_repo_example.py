import sys
sys.path.append("..")
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from typing import Optional
from chatboard.model.model import Model, ArtifactModel, RepoModel, Relation
from chatboard.model.fields import ModelField, KeyField, RelationField
from chatboard.model.namespace_manager import NamespaceManager


class Post(ArtifactModel):
    """Post model with repo awareness"""
    _db_type = "postgres"
    _repo = "blogs"  # This specifies that Post belongs to the Blog repo
    
    id: int = KeyField(primary_key=True)
    title: str = ModelField()
    content: str = ModelField()
    author_id: int = ModelField()
    blog_id: int = ModelField(default=None)  # Foreign key to Blog
    comments: Relation["Comment"] = RelationField(primary_key="post_id")


class Comment(ArtifactModel):
    """Comment model with repo awareness"""
    _db_type = "postgres"
    _repo = "blogs"  # This specifies that Comment belongs to the Blog repo
    
    id: int = KeyField(primary_key=True)
    message: str = ModelField()
    post_id: int = ModelField()


class Blog(RepoModel):
    """Blog model that acts as a repo for posts and comments"""
    _db_type = "postgres"
    
    id: int = KeyField(primary_key=True)
    name: str = ModelField()
    description: str = ModelField()
    posts: Relation[Post] = RelationField(primary_key="blog_id")


async def main():
    # Initialize the database
    print("Creating tables...")
    await NamespaceManager.create_all_namespaces(versioning=True)
    
    # Create a blog repo
    print("Creating blog repo...")
    blog = Blog(name="My Tech Blog", description="A blog about technology")
    await blog.save()
    print(f"Blog created with ID: {blog.id}, Branch ID: {blog.main_branch_id}")
    
    # Get the current turn
    turn = await blog.get_current_turn()
    print(f"Current turn: {turn.id if turn else 'None'}")
    
    # Create a post
    print("Creating post...")
    post = Post(title="First Post", content="This is my first post", author_id=1)
    await post.save()
    print(f"Post created with ID: {post.id}, Branch ID: {post.branch_id}, Turn ID: {post.turn_id}")
    
    # Add the post to the blog's posts relation
    await blog.posts.add(post)
    print(f"Post added to blog with ID: {post.id}, Blog ID: {post.blog_id}")
    
    # Create a comment
    print("Creating comment...")
    comment = Comment(message="Great post!", post_id=post.id)
    await comment.save()
    print(f"Comment created with ID: {comment.id}, Branch ID: {comment.branch_id}, Turn ID: {comment.turn_id}")
    
    # Add a comment to the post's comments relation
    await post.comments.add(comment)
    print(f"Comment added to post with ID: {comment.id}, Post ID: {comment.post_id}")
    
    # Commit the current turn
    new_turn_id = await blog.commit_turn("Initial commit")
    print(f"Committed turn and created new turn with ID: {new_turn_id}")
    
    # Create another post in the new turn
    print("Creating second post...")
    post2 = Post(title="Second Post", content="This is my second post", author_id=1)
    await post2.save()
    print(f"Post created with ID: {post2.id}, Branch ID: {post2.branch_id}, Turn ID: {post2.turn_id}")
    
    # Add the post to the blog's posts relation
    await blog.posts.add(post2)
    print(f"Post added to blog with ID: {post2.id}, Blog ID: {post2.blog_id}")
    
    # Query posts by branch
    print("Querying posts by branch...")
    posts = await Post.query(branch=blog.main_branch_id).execute()
    print(f"Found {len(posts)} posts in branch {blog.main_branch_id}:")
    for p in posts:
        print(f"- {p.title}: {p.content}")


if __name__ == "__main__":
    asyncio.run(main())