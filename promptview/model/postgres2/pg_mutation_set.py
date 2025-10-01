from typing import Generic, Type, TypeVar

from ..model3 import Model
from ..sql.queries import DeleteQuery, InsertQuery, UpdateQuery, SelectQuery
from .pg_query_set import PgSelectQuerySet

MODEL = TypeVar("MODEL", bound=Model)








class PgMutationSet(Generic[MODEL]):
    
    def __init__(self, model_class: Type[MODEL]):
        self.model_class = model_class
        self.namespace = model_class.get_namespace()
        
        
    # def get(self):
    #     return SelectQuery(
    #         select="*",
    #         from_table=self.model_class.get_namespace().table_name,
    #     )
    
    def create(self):
        