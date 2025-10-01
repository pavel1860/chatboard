from typing import TYPE_CHECKING, Any
from uuid import UUID



if TYPE_CHECKING:
    from promptview.model.postgres2.pg_namespace import PgNamespace
    from promptview.model.base.base_namespace import BaseNamespace
    



def insert_query(namespace: "BaseNamespace", data: dict[str, Any]) -> tuple[str, list[Any]]:
    fields = []
    placeholders = []
    values = []
    param_idx = 1

    for field in namespace.iter_fields():
        value = data.get(field.name, field.default)
        if field.is_primary_key and value is None:
            continue
        if value is None:
            if not field.is_optional and not field.is_primary_key:
                raise ValueError(f"Missing required field: '{field.name}' on Model '{namespace._model_cls.__name__}'")
        serialized = field.serialize(value)
        fields.append(f'"{field.name}"')
        placeholders.append(field.get_placeholder(param_idx))
        values.append(serialized)
        param_idx += 1

    if not fields:
        sql = f'INSERT INTO "{namespace.name}" DEFAULT VALUES RETURNING *;'
    else:
        sql = f"""
        INSERT INTO "{namespace.name}" ({", ".join(fields)})
        VALUES ({", ".join(placeholders)})
        RETURNING *;
        """
    return sql, values



def update_query(id: int | str | UUID, namespace: "BaseNamespace", data: dict[str, Any]) -> tuple[str, list[Any]]:
    set_clauses = []
    values = []

    index = 1
    for field in namespace.iter_fields():
        if field.is_primary_key:
            continue  # primary key goes in WHERE clause
        if field.name not in data:
            continue

        value = data[field.name]
        serialized = field.serialize(value)
        placeholder = field.get_placeholder(index)
        set_clauses.append(f'"{field.name}" = {placeholder}')
        values.append(serialized)
        index += 1

    # WHERE clause for primary key
    pk_field = namespace.primary_key_field
    where_placeholder = f"${index}"
    set_clause = ", ".join(set_clauses)
    values.append(pk_field.serialize(id))

    sql = f"""
    UPDATE "{namespace.name}"
    SET {set_clause}
    WHERE "{pk_field.name}" = {where_placeholder}
    RETURNING *;
    """
    return sql, values



