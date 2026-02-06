"""
Low-level Pattern Matching Utilities for Tree Data Structures

These are standalone utility functions that can be used independently
or by the Tree class methods.

Pattern Syntax:
- * : matches exactly one path segment (e.g., "1.*" matches "1.0", "1.1")
- ** : matches zero or more path segments (e.g., "1.**" matches all descendants)
- {a,b,c} : matches any of the options (e.g., "1.{0,1,2}" matches "1.0", "1.1", "1.2")
- [0-9] : matches any digit (e.g., "1.[0-5]" matches "1.0" through "1.5")
- literal : matches exact text
"""

import re
from typing import Pattern


def parse_path_level(path: str) -> int:
    """
    Parse the depth level of a path

    Examples:
        parse_path_level("1") -> 1
        parse_path_level("1.1.2") -> 3
    """
    return len(path.split('.'))


def is_pattern(path: str) -> bool:
    """
    Check if a path string contains pattern syntax

    Examples:
        is_pattern("1.2") -> False
        is_pattern("1.*") -> True
        is_pattern("1.**") -> True
        is_pattern("1.{0,1}") -> True
    """
    return bool(re.search(r'[*\[\{]', path))


def path_to_regex(pattern: str) -> Pattern[str]:
    """
    Convert a path pattern to a regular expression

    Examples:
        path_to_regex("1.*") -> re.compile(r'^1\.[^.]+$')
        path_to_regex("1.**") -> re.compile(r'^1\..*$')
        path_to_regex("**.message") -> re.compile(r'^.*\.message$')
        path_to_regex("1.{0,1,2}") -> re.compile(r'^1\.(0|1|2)$')
    """
    # Escape special regex characters except our pattern syntax
    regex_str = pattern
    regex_str = regex_str.replace('.', '\\.')  # Escape dots first
    regex_str = regex_str.replace('**', '§DOUBLESTAR§')  # Temporarily replace **
    regex_str = regex_str.replace('*', '[^.]+')  # * matches one segment
    regex_str = regex_str.replace('§DOUBLESTAR§', '.*')  # ** matches any segments

    # Handle choice patterns {a,b,c} -> (a|b|c)
    regex_str = re.sub(r'\{([^}]+)\}', lambda m: f"({m.group(1).replace(',', '|')})", regex_str)

    # Character classes [0-9] stay as-is

    return re.compile(f'^{regex_str}$')


def match_path(pattern: str, path: str) -> bool:
    """
    Check if a path matches a pattern

    Examples:
        match_path("1.*", "1.0") -> True
        match_path("1.*", "1.0.message") -> False
        match_path("**.message", "1.0.message") -> True
        match_path("1.{0,1}", "1.1") -> True
    """
    if not is_pattern(pattern):
        return pattern == path
    return bool(path_to_regex(pattern).match(path))


def get_parent_path(path: str) -> str | None:
    """
    Get the parent path

    Examples:
        get_parent_path("1.1.2") -> "1.1"
        get_parent_path("1") -> None
    """
    parts = path.split('.')
    if len(parts) <= 1:
        return None
    return '.'.join(parts[:-1])


def is_ancestor(path_a: str, path_b: str) -> bool:
    """
    Check if path_a is an ancestor of path_b

    Examples:
        is_ancestor("1", "1.1.2") -> True
        is_ancestor("1.1", "1.2") -> False
    """
    return path_b.startswith(path_a + '.')
