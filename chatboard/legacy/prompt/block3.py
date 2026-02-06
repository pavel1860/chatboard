import contextvars
import textwrap
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload
from pydantic import BaseModel

# Define types
TitleType = Literal["md", "xml", "html"]
BulletType = Literal["number", "alpha", "roman", "roman_upper", "*", "-"]

# Context variable for tracking the current block in context managers
block_ctx = contextvars.ContextVar("block_ctx", default=None)


def sanitize_value(value: str) -> str:
    """Clean up string values by removing extra whitespace and indentation."""
    return textwrap.dedent(value).strip()


class Block:
    """
    Core Block class for building structured prompts.
    
    This class provides the foundation for creating, manipulating, and rendering
    prompt blocks with support for nesting, tagging, and different formatting options.
    """
    
    def __init__(
        self, 
        content: Optional[str] = None, 
        role: str = "user", 
        id: Optional[str] = None,
        tag: Optional[str] = None,
        name: Optional[str] = None,
        parent: Optional['Block'] = None,
        title_type: Optional[TitleType] = None,
        bullet_type: Optional[BulletType] = None
    ):
        """
        Initialize a Block with content and metadata.
        
        Args:
            content: The text content of the block
            role: The role of the block (user, assistant, system)
            id: Unique identifier for the block
            tag: Tag for grouping and retrieving blocks
            name: Human-readable name for the block
            parent: Parent block if this is a nested block
            title_type: Type of title formatting (md, xml, html)
            bullet_type: Type of bullet for list items
        """
        self._id = id or str(uuid.uuid4())
        self._role = role
        self._tag = tag
        self._name = name
        self._items = []
        self._parent = parent
        self._token = None
        self._depth = 1 if parent is None else parent._depth + 1
        self._title_type = title_type
        self._bullet_type = bullet_type
        self._content = None
        
        # Add content if provided
        if content is not None:
            self._content = sanitize_value(content)
            
        # Auto-append to parent if in a context
        if parent is None:
            try:
                parent_ctx = block_ctx.get()
                if parent_ctx is not None:
                    parent_ctx.append(self)
                    self._parent = parent_ctx
                    self._depth = parent_ctx._depth + 1
            except LookupError:
                pass
    
    # Properties
    
    @property
    def id(self) -> str:
        """Get the block's unique identifier."""
        return self._id
    
    @property
    def role(self) -> str:
        """Get the block's role."""
        return self._role
    
    @role.setter
    def role(self, value: str):
        """Set the block's role."""
        self._role = value
    
    @property
    def tag(self) -> Optional[str]:
        """Get the block's tag."""
        return self._tag
    
    @tag.setter
    def tag(self, value: str):
        """Set the block's tag."""
        self._tag = value
    
    @property
    def name(self) -> Optional[str]:
        """Get the block's name."""
        return self._name
    
    @property
    def parent(self) -> Optional['Block']:
        """Get the block's parent."""
        return self._parent
    
    @property
    def depth(self) -> int:
        """Get the block's depth in the hierarchy."""
        return self._depth
    
    @property
    def items(self) -> List[Any]:
        """Get the block's items."""
        return self._items
    
    @property
    def title_type(self) -> Optional[TitleType]:
        """Get the block's title type."""
        return self._title_type
    
    @property
    def bullet_type(self) -> Optional[BulletType]:
        """Get the block's bullet type."""
        return self._bullet_type
    
    @property
    def content(self) -> Optional[str]:
        """Get the block's content."""
        return self._content
    
    # Content management methods
    
    def append(self, item: Union[str, 'Block']) -> 'Block':
        """
        Add an item to this block.
        
        Args:
            item: String or Block to append
            
        Returns:
            self for method chaining
        """
        if isinstance(item, str):
            item = sanitize_value(item)
        self._items.append(item)
        return self
    
    def extend(self, items: List[Union[str, 'Block']]) -> 'Block':
        """
        Add multiple items to this block.
        
        Args:
            items: List of strings or Blocks to append
            
        Returns:
            self for method chaining
        """
        for item in items:
            self.append(item)
        return self
    
    def remove(self, item_or_id: Union[str, 'Block']) -> 'Block':
        """
        Remove an item from this block.
        
        Args:
            item_or_id: Item or ID of item to remove
            
        Returns:
            self for method chaining
        """
        if isinstance(item_or_id, str):
            # Remove by ID
            self._items = [item for item in self._items 
                          if not (isinstance(item, Block) and item.id == item_or_id)]
        else:
            # Remove the item directly
            if item_or_id in self._items:
                self._items.remove(item_or_id)
        return self
    
    def clear(self) -> 'Block':
        """
        Remove all items from this block.
        
        Returns:
            self for method chaining
        """
        self._items = []
        return self
    
    # Rendering methods
    
    def render_items(self, indent: int = 0) -> str:
        """
        Render all items in this block.
        
        Args:
            indent: Number of spaces to indent
            
        Returns:
            Rendered string of all items
        """
        result = []
        for item in self._items:
            if isinstance(item, Block):
                result.append(item.render())
            else:
                result.append(str(item))
        
        content = "\n".join(result)
        if indent > 0:
            content = textwrap.indent(content, " " * indent)
        return content
    
    def _render_as_title(self) -> str:
        """
        Render this block as a title with its content.
        
        Returns:
            Rendered string with title and content
        """
        content = self.render_items(indent=2)
        title_text = self._content or ""
        
        if self._title_type == "md":
            # Markdown format
            hashes = "#" * min(self._depth + 1, 6)  # Max 6 levels of headers in markdown
            return f"{hashes} {title_text}\n{content}"
        
        elif self._title_type == "xml":
            # XML format
            return f"<{title_text}>\n{content}\n</{title_text}>"
        
        elif self._title_type == "html":
            # HTML format
            tag = f"h{min(self._depth + 1, 6)}"  # h1 through h6
            return f"<{tag}>{title_text}</{tag}>\n{content}"
        
        # Default fallback
        return f"{title_text}\n{content}"
    
    def _get_bullet(self, index: int) -> str:
        """
        Get the bullet for a specific index.
        
        Args:
            index: Item index
            
        Returns:
            Bullet string
        """
        if self._bullet_type == "number":
            return f"{index + 1}. "
        elif self._bullet_type == "alpha":
            return f"{chr(97 + index)}. "
        elif self._bullet_type == "roman":
            # Simple roman numeral conversion for 1-10
            romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
            return f"{romans[index] if index < len(romans) else index + 1}. "
        elif self._bullet_type == "roman_upper":
            romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            return f"{romans[index] if index < len(romans) else index + 1}. "
        else:
            return f"{self._bullet_type} "
    
    def _render_as_list(self) -> str:
        """
        Render this block as a list.
        
        Returns:
            Rendered list string
        """
        result = []
        
        for i, item in enumerate(self._items):
            bullet = self._get_bullet(i)
            if isinstance(item, Block):
                # For nested blocks, render them and prefix the first line
                lines = item.render().split("\n")
                if lines:
                    lines[0] = f"{bullet}{lines[0]}"
                    # Indent subsequent lines to align with text after bullet
                    indent = " " * len(bullet)
                    for j in range(1, len(lines)):
                        lines[j] = f"{indent}{lines[j]}"
                    result.append("\n".join(lines))
            else:
                # For simple strings
                result.append(f"{bullet}{item}")
        
        return "\n".join(result)
    
    def render(self) -> str:
        """
        Render this block and all its items based on its type.
        
        Returns:
            Rendered string of this block and all its items
        """
        if self._title_type:
            return self._render_as_title()
        elif self._bullet_type:
            return self._render_as_list()
        
        # If there's content, include it first
        result = []
        if self._content:
            result.append(self._content)
        
        # Then add rendered items
        items_rendered = self.render_items()
        if items_rendered:
            result.append(items_rendered)
            
        return "\n".join(result)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the block to a dictionary representation.
        
        Returns:
            Dictionary representation of the block
        """
        items = []
        for item in self._items:
            if isinstance(item, Block):
                items.append(item.to_dict())
            else:
                items.append(str(item))
                
        return {
            "id": self._id,
            "role": self._role,
            "tag": self._tag,
            "name": self._name,
            "content": self._content,
            "items": items,
            "depth": self._depth,
            "title_type": self._title_type,
            "bullet_type": self._bullet_type
        }
    
    # Tagging and retrieval methods
    
    def get(self, id_or_tag: str) -> Optional['Block']:
        """
        Get a block by ID or tag.
        
        Args:
            id_or_tag: ID or tag to search for
            
        Returns:
            Matching block or None if not found
        """
        # Check if this block matches
        if self._id == id_or_tag or self._tag == id_or_tag:
            return self
        
        # Check children
        for item in self._items:
            if isinstance(item, Block):
                result = item.get(id_or_tag)
                if result is not None:
                    return result
        
        return None
    
    def find_all(self, tag: Optional[str] = None, role: Optional[str] = None) -> List['Block']:
        """
        Find all blocks matching criteria.
        
        Args:
            tag: Tag to match
            role: Role to match
            
        Returns:
            List of matching blocks
        """
        results = []
        
        # Only check direct children, not nested ones
        for item in self._items:
            if isinstance(item, Block):
                if (tag is None or item._tag == tag) and (role is None or item._role == role):
                    results.append(item)
        
        return results
    
    def rearrange(self, order: List[str]) -> 'Block':
        """
        Rearrange blocks based on a list of IDs or tags.
        
        Args:
            order: List of IDs or tags in desired order
            
        Returns:
            New block with rearranged items
        """
        result = Block(role=self._role)
        
        # Add blocks in the specified order
        for id_or_tag in order:
            block = self.get(id_or_tag)
            if block is not None:
                result.append(block)
        
        return result
    
    # Context manager support
    
    def __enter__(self) -> 'Block':
        """Enter context manager, making this block the current context."""
        self._token = block_ctx.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, restoring previous context."""
        if self._token is not None:
            block_ctx.reset(self._token)
    
    # Operator overloading
    
    def __add__(self, other: Union[str, 'Block']) -> 'Block':
        """Support for + operator (concatenation)."""
        result = Block(content=self._content, role=self._role)
        result.extend(self._items)
        result.append(other)
        return result
    
    def __iadd__(self, other: Union[str, 'Block']) -> 'Block':
        """Support for += operator (in-place concatenation)."""
        self.append(other)
        return self
    
    def __str__(self) -> str:
        """Convert to string (basic rendering)."""
        if self._content is not None:
            return self._content
        if not self._items:
            return ""
        if len(self._items) == 1 and isinstance(self._items[0], str):
            return self._items[0]
        # Return a simple representation to avoid recursion
        if self._title_type:
            return f"TitleBlock({self._content})"
        elif self._bullet_type:
            return f"ListBlock({len(self._items)} items)"
        return f"Block({len(self._items)} items)"
    
    # def __repr__(self) -> str:
    #     """Detailed string representation."""
    #     return f"Block(id={self._id}, role={self._role}, items={len(self._items)})"
    
    # Factory methods for specialized blocks
    
    def title(
        self, 
        content: str, 
        type: TitleType = "md", 
        id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> 'TitleBlock':
        """
        Create a title block.
        
        Args:
            content: Title text
            type: Title format (md, xml, html)
            id: Block ID
            tag: Block tag
            
        Returns:
            New TitleBlock
        """
        block = TitleBlock(content, type=type, role=self._role, id=id, tag=tag, parent=self)
        self.append(block)
        return block
    
    def list(
        self, 
        items: Optional[List[str]] = None, 
        bullet: BulletType = "number", 
        id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> 'ListBlock':
        """
        Create a list block.
        
        Args:
            items: List items
            bullet: Bullet style
            id: Block ID
            tag: Block tag
            
        Returns:
            New ListBlock
        """
        block = ListBlock(bullet=bullet, role=self._role, id=id, tag=tag, parent=self)
        if items:
            block.extend(items)
        self.append(block)
        return block
    
    # Static factory methods
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Block':
        """
        Create a block from a dictionary representation.
        
        Args:
            data: Dictionary representation of a block
            
        Returns:
            New Block
        """
        block = Block(
            content=data.get("content"),
            role=data.get("role", "user"),
            id=data.get("id"),
            tag=data.get("tag"),
            name=data.get("name"),
            title_type=data.get("title_type"),
            bullet_type=data.get("bullet_type")
        )
        
        for item in data.get("items", []):
            if isinstance(item, dict):
                block.append(Block.from_dict(item))
            else:
                block.append(item)
        
        return block


class TitleBlock(Block):
    """
    Block with a title and specialized rendering.
    
    This is a thin wrapper around Block with title_type set.
    """
    
    def __init__(
        self, 
        title: str, 
        type: TitleType = "md", 
        role: str = "user", 
        id: Optional[str] = None,
        tag: Optional[str] = None,
        name: Optional[str] = None,
        parent: Optional[Block] = None
    ):
        """
        Initialize a TitleBlock.
        
        Args:
            title: Title text
            type: Title format (md, xml, html)
            role: Block role
            id: Block ID
            tag: Block tag
            name: Block name
            parent: Parent block
        """
        super().__init__(
            content=title, 
            role=role, 
            id=id, 
            tag=tag, 
            name=name, 
            parent=parent, 
            title_type=type
        )


class ListBlock(Block):
    """
    Block for creating lists with different bullet styles.
    
    This is a thin wrapper around Block with bullet_type set.
    """
    
    def __init__(
        self, 
        bullet: BulletType = "number", 
        role: str = "user", 
        id: Optional[str] = None,
        tag: Optional[str] = None,
        name: Optional[str] = None,
        parent: Optional[Block] = None
    ):
        """
        Initialize a ListBlock.
        
        Args:
            bullet: Bullet style
            role: Block role
            id: Block ID
            tag: Block tag
            name: Block name
            parent: Parent block
        """
        super().__init__(
            role=role, 
            id=id, 
            tag=tag, 
            name=name, 
            parent=parent, 
            bullet_type=bullet
        )


# Convenience function to create blocks
def block(
    content: Optional[str] = None, 
    role: str = "user", 
    id: Optional[str] = None,
    tag: Optional[str] = None,
    name: Optional[str] = None,
    title_type: Optional[TitleType] = None,
    bullet_type: Optional[BulletType] = None
) -> Block:
    """
    Create a new Block.
    
    This function can be used as a factory function or as a decorator.
    
    Args:
        content: Block content
        role: Block role
        id: Block ID
        tag: Block tag
        name: Block name
        title_type: Type of title formatting
        bullet_type: Type of bullet for list items
        
    Returns:
        New Block
    """
    return Block(
        content=content, 
        role=role, 
        id=id, 
        tag=tag, 
        name=name, 
        title_type=title_type, 
        bullet_type=bullet_type
    )
