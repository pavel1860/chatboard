from enum import Enum
from typing import Literal


class VersioningStrategy(str, Enum):
    """Strategy for versioning models"""
    NONE = "none"           # No versioning (User, Config, etc.)
    SNAPSHOT = "snapshot"   # One artifact per object (Block, Span, Log)
    EVENT_SOURCED = "event_sourced"  # New artifact per change (user models)
    
    
ArtifactKind = Literal["model", "block", "span", "log"]