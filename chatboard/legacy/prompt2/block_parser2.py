import json
from typing import Any, Dict, List, Type, Optional, Generator
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field
import re


def clean_xml_text(text: str) -> str:
    """Clean and normalize XML text content."""
    return re.sub(r'\s+', ' ', text).strip()


def parse_json_like(text: str) -> dict:
    """Parse JSON-like strings that may have unquoted keys or trailing commas."""
    # Add quotes to unquoted keys
    text = re.sub(r'(\s*?{\s*?|\s*?,\s*?)([a-zA-Z0-9_]+?)\s*?:', r'\1"\2":', text)
    # Remove trailing commas
    text = re.sub(r',(\s*?[}\]])', r'\1', text)
    return json.loads(text)


class ActionContent:
    """Helper class to manage action content and model instantiation."""
    
    _model_registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register_model(cls, action_name: str, model_class: Type[BaseModel]):
        """Register a model class for a specific action name."""
        cls._model_registry[action_name] = model_class
    
    @classmethod
    def get_model_class(cls, action_name: str) -> Optional[Type[BaseModel]]:
        """Get the registered model class for an action name."""
        return cls._model_registry.get(action_name)
    
    @classmethod
    def parse(cls, action_name: str, content: str) -> Any:
        """Parse action content and instantiate appropriate model if registered."""
        try:
            data = parse_json_like(content)
            model_class = cls.get_model_class(action_name)
            if model_class:
                return model_class(**data)
            return data
        except Exception as e:
            # If parsing fails, return the raw content
            return content


class ParsedBlock(BaseModel):
    """Base class for parsed blocks."""
    tag: str
    content: str | dict | BaseModel | List['ParsedBlock'] = ""
    attributes: Dict[str, str] = Field(default_factory=dict)


def walk_xml_tree(root: ET.Element) -> Generator[ET.Element, None, None]:
    """Walk through XML tree in pre-order traversal."""
    yield root
    for child in root:
        yield from walk_xml_tree(child)


def parse_block(element: ET.Element) -> ParsedBlock:
    """Parse a single XML element into a ParsedBlock."""
    # Get tag and attributes
    tag = element.tag
    attributes = dict(element.attrib)
    
    # Handle leaf nodes (no children)
    if len(element) == 0:
        text = clean_xml_text(element.text or "")
        
        # If this is an action block, try to parse its content
        if tag.lower() == "action":
            action_name = attributes.get("name", "")
            content = ActionContent.parse(action_name, text)
        else:
            content = text
            
        return ParsedBlock(
            tag=tag,
            content=content,
            attributes=attributes
        )
    
    # Handle nodes with children
    children = []
    for child in element:
        children.append(parse_block(child))
        
    return ParsedBlock(
        tag=tag,
        content=children,
        attributes=attributes
    )


def parse_blocks(text: str) -> ParsedBlock:
    """
    Parse XML-structured block output into a ParsedBlock object.
    Uses ElementTree for robust XML parsing.
    
    Args:
        text: The XML-structured text to parse
        
    Returns:
        ParsedBlock object containing the parsed content
        
    Raises:
        ValueError: If the XML is malformed
    """
    try:
        # Wrap the text in a root element to ensure valid XML
        wrapped_text = f"<root>{text}</root>"
        root = ET.fromstring(wrapped_text)
        
        # Parse the entire tree
        result = parse_block(root)
        
        # If root has only one child, return that child instead
        if isinstance(result.content, list) and len(result.content) == 1:
            return result.content[0]
            
        return result
        
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML structure: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing blocks: {e}") 