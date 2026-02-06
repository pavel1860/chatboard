import json
import pytest
from typing import Any, List, Dict

from chatboard.model.query_types import (
    validate_query
)

# Example queries from the provided examples
EXAMPLE_QUERIES = [
    [["name", "==", "bbb"]],
    [["name", "==", "bbb"], "and", ["age", ">", 10], "and", ["age", "<", 20]],
    [["name", "==", "test"], "and", ["age", ">=", 18], "and", ["age", "<=", 65]],
    [["price", ">", 100], "and", ["price", "<", 1000], "and", ["category", "==", "electronics"]],
    [["status", "==", "active"], "and", ["created", ">", 1000], "and", ["created", "<", 2000]],
    [["score", ">", 0.5], "and", ["score", "<", 0.7]],
    [["date", ">", "2024-01-01"], "and", ["date", "<", "2024-01-02"]],
    [["datetime", ">", "2024-01-01T00:00:00"], "and", ["datetime", "<", "2024-01-01T00:00:01"]]
]

def test_simple_query_validation():
    """Test validation of simple queries"""
    # Valid simple query
    query = [["name", "==", "bbb"]]
    assert validate_query(query)
    
    # Invalid simple query - wrong operator
    invalid_query = [["name", "invalid", "bbb"]]
    assert not validate_query(invalid_query)
    
    # Invalid simple query - wrong structure
    invalid_query = [["name", "=="]]  # Missing value
    assert not validate_query(invalid_query)
    
    # Invalid simple query - not a list
    invalid_query = "not a list"
    assert not validate_query(invalid_query)

def test_complex_query_validation():
    """Test validation of complex queries"""
    # Valid complex query
    query = [
        ["name", "==", "bbb"], 
        "and", 
        ["age", ">", 10], 
        "and", 
        ["age", "<", 20]
    ]
    assert validate_query(query)
    
    # Invalid complex query - wrong logical operator
    invalid_query = [
        ["name", "==", "bbb"], 
        "invalid", 
        ["age", ">", 10]
    ]
    assert not validate_query(invalid_query)
    
    # Invalid complex query - condition at odd index
    invalid_query = [
        ["name", "==", "bbb"], 
        ["age", ">", 10], 
        "and"
    ]
    assert not validate_query(invalid_query)
    
    # Invalid complex query - operator at even index
    invalid_query = [
        "and",
        ["name", "==", "bbb"], 
        ["age", ">", 10]
    ]
    assert not validate_query(invalid_query)
