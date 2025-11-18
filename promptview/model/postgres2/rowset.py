# promptview/model3/sql/rowset.py
from typing import Generic, Optional, Type, TypeVar
from ..sql.queries import Table

T = TypeVar("T")  # Model subtype

class RowsetNode(Generic[T]):
    """
    A named rowset (CTE/subquery) that produces rows of model T (optionally typed).
    This is backend-agnostic; compilers lower it to WITH/CTE (PG) or CALL { ... } (Neo4j).
    """
    def __init__(
        self,
        name: str,
        query,                     # SelectQuery | RawSQL | backend node
        *,
        model: Optional[Type[T]] = None,
        key: Optional[str] = None, # name of PK-like column in this rowset
        recursive: bool = False,
        materialized: Optional[str] = None  # "MATERIALIZED"/"NOT MATERIALIZED"/None (PG only)
    ):
        self.name = name
        self.query = query
        self.model = model
        self.key = key
        self.recursive = recursive
        self.materialized = materialized

    # sugar for manual usage
    def as_table(self, alias: str | None = None) -> Table:
        return Table(self.name, alias=alias)

    # override the key if the CTE exposes a different column name
    def with_key(self, key: str) -> "RowsetNode[T]":
        self.key = key
        return self
