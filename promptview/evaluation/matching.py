"""Path pattern matching for value evaluators."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..prompt.span_tree import Value, SpanTree
    from .models import EvaluatorConfig


def match_ltree_pattern(path: str, pattern: str) -> bool:
    """
    Match an LTREE path against a pattern.

    Patterns:
    - "1" - Exact match for path "1"
    - "1.2" - Exact match for path "1.2"
    - "1.*" - Match any direct child of "1" (e.g., "1.0", "1.1", but not "1.0.0")
    - "1.*{1,}" - Match "1" and any descendants (e.g., "1.0", "1.0.0", "1.0.0.1")
    - "*" - Match any top-level path (e.g., "1", "2", "3")
    - "*.*" - Match any value that is a child of a top-level path
    - "*.0" - Match first child of any parent

    Args:
        path: The LTREE path to check (e.g., "1.2.3")
        pattern: The LTREE pattern (e.g., "1.*")

    Returns:
        True if the path matches the pattern
    """
    # Handle exact match
    if pattern == path:
        return True

    # Handle wildcard patterns
    if "*" not in pattern:
        return False

    # Convert LTREE pattern to simple matching logic
    # For MVP, we'll support common patterns

    # "*" - Match any single-level path
    if pattern == "*":
        return "." not in path

    # "*.*" - Match any two-level path
    if pattern == "*.*":
        parts = path.split(".")
        return len(parts) == 2

    # "1.*" - Match direct children of "1"
    if pattern.endswith(".*") and not pattern.startswith("*"):
        prefix = pattern[:-2]  # Remove ".*"
        parts = path.split(".")
        if len(parts) == len(prefix.split(".")) + 1:
            return path.startswith(prefix + ".")
        return False

    # "*.0" - Match specific child of any parent
    if pattern.startswith("*."):
        suffix = pattern[2:]  # Remove "*."
        parts = path.split(".")
        if len(parts) >= 2:
            return parts[-1] == suffix or path.endswith("." + suffix)
        return False

    # "1.*{1,}" - Match "1" and any descendants
    if "{" in pattern:
        # Extract base path
        base = pattern.split(".*")[0]
        return path == base or path.startswith(base + ".")

    return False


def match_value_to_evaluators(
    value: "Value",
    span: "SpanTree",
    evaluators: list["EvaluatorConfig"]
) -> list["EvaluatorConfig"]:
    """
    Match a value to evaluators based on path patterns, tags, span name, and value name.

    Args:
        value: The Value object to match
        span: The parent SpanTree that contains this value
        evaluators: List of evaluator configurations

    Returns:
        List of matching evaluator configurations
    """
    matched = []

    for evaluator in evaluators:
        # Check path pattern
        if evaluator.path_pattern:
            if not match_ltree_pattern(value.path, evaluator.path_pattern):
                continue

        # Check tags (span must have at least one of the specified tags)
        if evaluator.tags:
            if not any(tag in span.root.tags for tag in evaluator.tags):
                continue

        # Check span name (exact match)
        if evaluator.span_name:
            if span.name != evaluator.span_name:
                continue

        # Check value name (exact match)
        if evaluator.value_name:
            if value.name != evaluator.value_name:
                continue

        # All criteria matched
        matched.append(evaluator)

    return matched
