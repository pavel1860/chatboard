# pg_relation.py
from typing import Optional, Type, Union

class PgRelation:
    def __init__(
        self,
        name: str,
        primary_key: str,
        foreign_key: str,
        foreign_cls: Union[Type, str],  # may be ForwardRef/str until finalized
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
        is_one_to_one: bool = False,
        relation_model: Optional[Type] = None  # for M:N with metadata
    ):
        self.name = name
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self.foreign_cls = foreign_cls
        self.on_delete = on_delete
        self.on_update = on_update
        self.is_one_to_one = is_one_to_one
        self.relation_model = relation_model  # Subclass of RelationModel if M:N w/ metadata

    def __repr__(self):
        return f"<PgRelation {self.name}: {self.primary_key}->{self.foreign_cls}.{self.foreign_key}>"
