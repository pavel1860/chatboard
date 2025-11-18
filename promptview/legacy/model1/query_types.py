from typing import Any, List, Literal, Tuple, TypedDict, Union, cast

# Define the comparison operators
ComparisonOperator = Literal["==", "!=", ">", "<", ">=", "<="]

# Define the logical operators that can connect conditions
LogicalOperator = Literal["and", "or"]

# Define a single condition as a 3-element list: [field_name, operator, value]
# Using List instead of Tuple for easier compatibility with the examples
Condition = List[Any]  # [str, ComparisonOperator, Any]

# Define a query element which can be either a condition or a logical operator
QueryElement = Union[Condition, LogicalOperator]

# Define a simple query with just one condition
SimpleQuery = List[Condition]

# Define a complex query with multiple conditions connected by logical operators
# This is a list of elements that alternates between conditions and logical operators
ComplexQuery = List[QueryElement]

# The general query type that can be either simple or complex
QueryListType = Union[SimpleQuery, ComplexQuery]


def validate_query(query: Any) -> bool:
    """
    Validate that a query matches the expected structure.
    
    Args:
        query: The query to validate
        
    Returns:
        True if the query is valid, False otherwise
    """
    if not isinstance(query, list) or not query:
        return False
    
    # Simple query with just one condition
    if len(query) == 1:
        return (isinstance(query[0], list) and 
                len(query[0]) == 3 and 
                isinstance(query[0][0], str) and 
                isinstance(query[0][1], str) and 
                query[0][1] in ("==", "!=", ">", "<", ">=", "<="))
    
    # Complex query with multiple conditions
    for i, element in enumerate(query):
        if i % 2 == 0:  # Even indices should be conditions
            if not (isinstance(element, list) and 
                    len(element) == 3 and 
                    isinstance(element[0], str) and 
                    isinstance(element[1], str) and 
                    element[1] in ("==", "!=", ">", "<", ">=", "<=")):
                return False
        else:  # Odd indices should be operators
            if not (isinstance(element, str) and element in ("and", "or")):
                return False
    
    return True

