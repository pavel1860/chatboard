import json
from typing import Any, Dict, List, Type, Optional
from xml.etree import ElementTree as ET
from pydantic import BaseModel
import re


def clean_xml_text(text: str) -> str:
    """Clean and normalize XML text content."""
    return re.sub(r'\s+', ' ', text).strip()


def parse_json_like(text: str) -> dict:
    """
    Convert a JSON-like string with unquoted keys into a valid JSON object.
    Handles common formatting issues in LLM outputs.
    """
    # Remove any leading/trailing whitespace
    s = text.strip()
    
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    
    # Quote unquoted keys
    s = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', s)
    
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON-like content: {e}")


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


class ParsedAction(BaseModel):
    """Represents a parsed action with its content."""
    name: str
    id: str
    content: Any


class ParsedOutput(BaseModel):
    """Represents the complete parsed output structure."""
    observation: str = ""
    thought: str = ""
    actions: List[ParsedAction] = []


def parse_blocks(text: str) -> ParsedOutput:
    """
    Parse XML-structured block output into a ParsedOutput object.
    Uses ElementTree for robust XML parsing.
    
    Args:
        text: The XML-structured text to parse
        
    Returns:
        ParsedOutput object containing the parsed content
        
    Raises:
        ValueError: If the XML is malformed or required sections are missing
    """
    try:
        # Wrap the text in a root element to ensure valid XML
        wrapped_text = f"<root>{text}</root>"
        root = ET.fromstring(wrapped_text)
        
        # Initialize output sections
        observation = ""
        thought = ""
        actions = []
        
        # Parse each section
        for element in root:
            tag = element.tag.lower()
            
            if tag == "observation":
                observation = clean_xml_text(element.text or "")
                
            elif tag == "thought":
                thought = clean_xml_text(element.text or "")
                
            elif tag == "actions":
                # Parse each action element
                for action_elem in element.findall("Action"):
                    name = action_elem.get("name", "")
                    action_id = action_elem.get("id", "")
                    content = clean_xml_text(action_elem.text or "")
                    
                    # Parse the content and instantiate appropriate model
                    parsed_content = ActionContent.parse(name, content)
                    
                    actions.append(ParsedAction(
                        name=name,
                        id=action_id,
                        content=parsed_content
                    ))
        
        return ParsedOutput(
            observation=observation,
            thought=thought,
            actions=actions
        )
        
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML structure: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing blocks: {e}") 