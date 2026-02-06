import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, Iterator, List, Literal, Protocol, Self, Type, TypeVar, Union, get_origin, get_args
import numpy as np
from pydantic.fields import FieldInfo
import datetime as dt

from chatboard.model.base_namespace import Namespace, QuerySet

if TYPE_CHECKING:  # pragma: nocoverage
    from .model import Model
    from .base_namespace import NSFieldInfo




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




class FieldOp(Enum):
    RANGE = "range"
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOTIN = "notin"
    NULL = "null"
    
    

class QueryOp(Enum):
    AND = "and"
    OR = "or"    


class QueryFilter:
    
    def __init__(self, left, operator, right):
        self._left = left
        self._right = right
        self._operator = operator
        # print("INIT", left, right, operator)
        if operator not in [QueryOp.AND, QueryOp.OR]:
            if isinstance(self._left, PropertyComparable):
                self.field = self._left
                self.value = self._right
            elif isinstance(self._right, PropertyComparable):
                self.field = self._right
                self.value = self._left
            else:
                raise ValueError("No PropertyComparable found")

    
    def is_datetime(self):
        if self.field.type == dt.datetime:
            return True
        elif get_origin(self.field.type) == UnionType or get_origin(self.field.type) == Union:
            # Optional[datetime]
            union_args = get_args(self.field.type)
            if len(union_args) == 2 and dt.datetime in union_args and type(None) in union_args:
                return True
        return False
    
        
    def __and__(self, other):
        if other is None:
            return self
        return QueryFilter(self, QueryOp.AND, other)
    
    def __or__(self, other):
        if other is None:
            return self
        return QueryFilter(self, QueryOp.OR, other)
    
    
    def _set_values(self):
        if isinstance(self._left, FieldComparable):
            self._field = self._left
            self._value = self._right
        elif isinstance(self._right, FieldComparable):
            self._field = self._right
            self._value = self._left
        else:
            raise ValueError("No FieldComparable found")     
        
    def __repr__(self):
        def recurse(obj):
            if isinstance(obj, QueryFilter):
                return repr(obj)
            return repr(obj)

        left_repr = recurse(self._left)
        right_repr = recurse(self._right)

        return f"({left_repr} {self._operator.value.upper()} {right_repr})"


    def model_dump(self):
        left = self._left.model_dump() if hasattr(self._left, "model_dump") else self._left
        right = self._right.model_dump() if hasattr(self._right, "model_dump") else self._right
        return {
            "left": left,
            "operator": self._operator.value,
            "right": right,
            "_type": "filter"
        }
        

class RangeFilter:
    def __init__(self, ge=None, le=None, gt=None, lt=None):
        types = [type(v) for v in [ge, le, gt, lt] if v is not None]
        if not types:
            raise ValueError("At least one value must be provided")
        for t in types:
            if t != types[0]:
                raise ValueError("All values must be of the same type")
        self.value = types[0]
        self._ge = ge
        self._le = le
        self._gt = gt
        self._lt = lt        
    
    @property
    def ge(self):
        if isinstance(self._ge, QuerySetField):
            return self._ge.get()
        return self._ge
    
    @property
    def le(self):
        if isinstance(self._le, QuerySetField):
            return self._le.get()
        return self._le
    
    @property
    def gt(self):
        if isinstance(self._gt, QuerySetField):
            return self._gt.get()
        return self._gt
    
    @property
    def lt(self):
        if isinstance(self._lt, QuerySetField):
            return self._lt.get()
        return self._lt
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["ge", "le", "gt", "lt"]:            
            if value is not None and type(value) != self.value:
                raise ValueError(f"Value must be of type {self.value}")
        super().__setattr__(name, value)
        
    
        
    def __repr__(self):
        return f"RangeFilter(ge={self.ge}, le={self.le}, gt={self.gt}, lt={self.lt})"
    
    
    def model_dump(self):
        return {
            "ge": self.ge,
            "le": self.le,
            "gt": self.gt,
            "lt": self.lt,
            "_type": "range"
        }


class QueryField:
    pass

class PropertyComparable:
    
    def __init__(self, field_name: str, type: Type[Any] | None = None):
        self._filters = {}
        self.name = field_name
        self.type = type
        self._query_filter = None
        
    def _validate_type(self, other):
        if isinstance(other, PropertyComparable):
            other_type = other.type
        else:
            other_type = type(other)
        if self.type is not None and self.type != other_type:
            origin = get_origin(self.type)
            if origin == UnionType or origin == Union:
                union_args = get_args(self.type)
                if other_type not in union_args:
                    raise ValueError(f"Cannot compare {self.name} with {other}. Expected one of {union_args} got {other_type}")
            else:
                raise ValueError(f"Cannot compare {self.name} with {other}. Expected {self.type} got {other_type}")
        # if not isinstance(other, FieldComparable):
            # raise ValueError(f"Cannot compare {self._field_name} with {other}")

    def __gt__(self, other):
        self._validate_type(other)
        if self._query_filter is None:
            self._query_filter = QueryFilter(self, FieldOp.RANGE, RangeFilter(gt=other))
        elif self._query_filter._operator == FieldOp.RANGE:
            self._query_filter._right.gt = other
        else:
            raise ValueError(f"Cannot compare {self.name} with {other}")
        return self._query_filter
            

    def __ge__(self, other):
        print(self.name, "GE", other)
        self._validate_type(other)
        if self._query_filter is None:
            self._query_filter = QueryFilter(self, FieldOp.RANGE, RangeFilter(ge=other))
        elif self._query_filter._operator == FieldOp.RANGE:
            self._query_filter._right.ge = other
        else:
            raise ValueError(f"Cannot compare {self.name} with {other}")
        return self._query_filter
        

    def __lt__(self, other):
        self._validate_type(other)
        if self._query_filter is None:
            self._query_filter = QueryFilter(self, FieldOp.RANGE, RangeFilter(lt=other))
        elif self._query_filter._operator == FieldOp.RANGE:
            self._query_filter._right.lt = other
        else:
            raise ValueError(f"Cannot compare {self.name} with {other}")
        return self._query_filter

    def __le__(self, other):
        print(self.name, "LE", other)
        self._validate_type(other)
        if self._query_filter is None:
            self._query_filter = QueryFilter(self, FieldOp.RANGE, RangeFilter(le=other))        
        elif self._query_filter._operator == FieldOp.RANGE:
            self._query_filter._right.le = other
        else:
            raise ValueError(f"Cannot compare {self.name} with {other}")
        return self._query_filter
    
    def __eq__(self, other):
        self._validate_type(other)
        return QueryFilter(self, FieldOp.EQ, other)    
    
    def __ne__(self, other):
        self._validate_type(other)
        return QueryFilter(self, FieldOp.NE, other)    
    
    def isin(self, *args):
        if type(args[0]) == list:
            other = args[0]
        elif type(args) == tuple:
            other = list(args)
        else:
            raise ValueError(f"Invalid value for any: {args}")
        return QueryFilter(self, FieldOp.IN, other)
    
    def notin(self, *args):
        if type(args[0]) == list:
            other = args[0]
        elif type(args) == tuple:
            other = list(args)
        else:
            raise ValueError(f"Invalid value for any: {args}")
        return QueryFilter(self, FieldOp.NOTIN, other)

    def isnull(self):
        return QueryFilter(self, FieldOp.NULL, None)
    
    def __repr__(self):
        return f"Property({self.name})"
    
    def model_dump(self):
        return {
            "name": self.name,
            "type": self.type.__name__ if self.type else None,
            "_type": "property"
        }
        


class FieldComparable(PropertyComparable):
    
    def __init__(self, field_name: str, field_info: "NSFieldInfo"):
        self._field_info = field_info
        super().__init__(field_name, field_info.data_type) 

    def __repr__(self):
        return f"Field({self.name})"
    
class QuerySetField(PropertyComparable):
    def __init__(self, field_name: str, field_info: "NSFieldInfo", alias: str):
        self._field_info = field_info
        super().__init__(field_name, field_info.data_type) 
        self.alias = alias
    
    def __repr__(self):
        return f"QuerySetField({self.name})"
        
    def get(self):
        return f"{self.alias}.{self.name}"
    
    
class SelectField(PropertyComparable):
    def __init__(self, field_name: str, field_info: "NSFieldInfo"):
        self._field_info = field_info
        super().__init__(field_name, field_info.data_type) 
        self._label = None
        
        
    def __repr__(self):
        if self._label:
            return f"SelectField({self.name}, label={self._label})"
        else:
            return f"SelectField({self.name})"
        
    def label(self, label: str):
        self._label = label
        return self
    
    def get(self):
        if self._label:
            return f'"{self._label}" AS "{self._field_info.name}"'
        else:
            return f'"{self._field_info.name}"'
        
        
        
MODEL = TypeVar("MODEL", bound="Model")   
FIELD_INFO = TypeVar("FIELD_INFO", bound="NSFieldInfo")

class ModelFilterProxy(Generic[MODEL]):
    def __init__(self, cls: Type[MODEL]):
        """
        Initialize the ORM object with a table name and fields (columns).

        :param cls: class for the proxy
        :param fields: Dictionary of field names and their values
        """   
        self._cls = cls
        fields = cls.model_fields
        self._fields = fields

        # Dynamically add properties for each field to support Intellisense
        for field_name in fields.keys():
            self._add_property(field_name)

    def _add_property(self, field_name):
        """
        Add a property dynamically for a given field name.
        This ensures Intellisense recognizes the field.
        """
        def getter(self):
            return self._fields[field_name]

        def setter(self, value):
            self._fields[field_name] = value

        # Dynamically add the property to the class
        setattr(self.__class__, field_name, property(getter, setter))

    def __dir__(self):
        """
        Add dynamic fields to the list of available attributes for Intellisense.
        """
        return list(super().__dir__()) + list(self._fields.keys())

    def get_metadata(self):
        """
        Return metadata about the fields (e.g., names, types).
        """
        return {
            "fields": list(self._fields.keys()),
            "field_values": self._fields
        }    

        
        


FusionType = Literal["rff", "dbsf"]

# class VectorQuerySet:

class AllVecs:
    def __bool__(self):
        return True
    
ALL_VECS = AllVecs()


class QueryProxyAny:
    
    def __getattr__(self, name):  
        return PropertyComparable(name)
        
    


class QueryProxy(Generic[MODEL, FIELD_INFO]):
    # model: Type[MODEL]
    
    def __init__(self, model: Type[MODEL], namespace: Namespace[MODEL, FIELD_INFO]):
        self.model = model
        self.namespace = namespace
        fields = model.model_fields
        self._fields = fields
        # for field_name in fields.keys():
        #     self._add_property(field_name)
        # self.__annotations__ = model.model_fields
        
    # def _add_property(self, field_name):
    #     """
    #     Add a property dynamically for a given field name.
    #     This ensures Intellisense recognizes the field.
    #     """
    #     def getter(self):
    #         return self._fields[field_name]

    #     def setter(self, value):
    #         self._fields[field_name] = value

    #     # Dynamically add the property to the class
    #     setattr(self.__class__, field_name, property(getter, setter))
    def _add_property(self, field_name):
        """
        Add a property dynamically for a given field name.
        This ensures Intellisense recognizes the field.
        """
        def getter(self):
            # return self._fields[field_name]
            return FieldComparable(field_name, self._fields[field_name])

        def setter(self, value):
            self._fields[field_name] = value

        # Dynamically add the property to the class
        setattr(self.__class__, field_name, property(getter, setter))
        
    def __dir__(self) -> Iterable[str]:
        # return list(super().__dir__()) + list(self.model.model_fields.keys())
        return list(super().__dir__()) + list(self._fields.keys())
    
    def __getattr__(self, name):
        # if name == "model":
        #     return self.model        
        # print("GET ATTR", name)
        # if field_info:= self.model.model_fields.get(name, None):            
        if field_info:= self.namespace.get_field(name):
            return FieldComparable(name, field_info)
        else:
            raise AttributeError(f"{self.model.__name__} has no attribute {name}")


class QuerySetProxy(QueryProxy[MODEL, FIELD_INFO]):
    
    def __init__(self, query_set: "QuerySet[MODEL]"):
        super().__init__(query_set.model_class, query_set.namespace)
        self.query_set = query_set
        
        
    def __getattr__(self, name):
        if field_info:= self.namespace.get_field(name):
            return QuerySetField(name, field_info, alias=self.query_set.alias)
        else:
            raise AttributeError(f"{self.model.__name__} has no attribute {name}")


class SelectFieldProxy(QueryProxy[MODEL, FIELD_INFO]):
    
    def __init__(self, model: Type[MODEL], namespace: Namespace[MODEL, FIELD_INFO]):
        super().__init__(model, namespace)
        
    def __getattr__(self, name):
        if sel_field:= self.namespace.get_field(name):
            return SelectField(name, sel_field)
        else:
            raise AttributeError(f"{self.model.__name__} has no attribute {name}")
 
        
        
        


def parse_query_params(conditions: QueryListType) -> QueryFilter | None:
    """Parse query parameters string into a QueryFilter object
    
    example:
    conditions = [
        ["test", "==", 1],
        ["test", ">=", 1],
        ["test", "<=", 1],
        ["test", ">", 1],
        ["test", "<", 1],
    ]
    """
    # Split into individual conditions
    # conditions = query_params.split('&')
    
    # Group conditions by field name
    field_conditions = {}
    for condition in conditions:
        field = None
        value = None
        operator = None
        
        if len(condition) != 3:
            raise ValueError(f"Invalid condition: {condition}")
        field, operator, value = condition
        # Parse operators and values
        if operator == "==":
            field_conditions[field] = {'eq': value}
        elif operator == ">=":
            if field not in field_conditions:
                field_conditions[field] = {'range': {}}
            field_conditions[field]['range']['ge'] = parse_value(field, value)
        elif operator == "<=":
            if field not in field_conditions:
                field_conditions[field] = {'range': {}}
            field_conditions[field]['range']['le'] = parse_value(field, value)
        elif operator == ">":
            if field not in field_conditions:
                field_conditions[field] = {'range': {}}
            field_conditions[field]['range']['gt'] = parse_value(field, value)
        elif operator == "<":
            if field not in field_conditions:
                field_conditions[field] = {'range': {}}
            field_conditions[field]['range']['lt'] = parse_value(field, value)

    # Convert conditions to QueryFilter objects
    filters = []
    for field, conditions in field_conditions.items():
        field_type = get_field_type(field)
        if 'eq' in conditions:
            filters.append(QueryFilter(
                PropertyComparable(field, field_type),
                FieldOp.EQ,
                conditions['eq']
            ))
        elif 'range' in conditions:
            range_params = conditions['range']
            filters.append(QueryFilter(
                PropertyComparable(field, field_type),
                FieldOp.RANGE,
                RangeFilter(**range_params)
            ))

    # Combine filters with AND
    result = filters[0]
    for f in filters[1:]:
        result = result & f
        
    return result


def get_field_type(field: str) -> Type:
    """Determine field type based on field name"""
    if field in ['score']:
        return float
    elif field in ['date', 'datetime']:
        return dt.datetime
    elif field in ['name', 'category', 'status']:
        return str
    else:
        return int

def parse_value(field: str, value: str) -> Any:
    """Parse string value into appropriate type based on field name"""
    field_type = get_field_type(field)
    if field_type == float:
        return float(value)
    elif field_type == dt.datetime:
        try:
            return dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            return dt.datetime.strptime(value, '%Y-%m-%d')
    elif field_type == str:
        return value
    else:
        return int(value)






# def parse_query_params(model_cls: Any, curr_filter: QueryFilter | None = None, **kwargs):
            
#     for k, v in kwargs.items():
#         if isinstance(v, QueryFilter):
#             curr_filter = curr_filter & v if curr_filter else v
#         else:
#             if curr_filter is None:
#                 curr_filter = QueryFilter(FieldComparable(k, model_cls.model_fields[k]), FieldOp.EQ, v)
#             else:
#                 curr_filter = curr_filter & QueryFilter(FieldComparable(k, model_cls.model_fields[k]), FieldOp.EQ, v)
#     return curr_filter