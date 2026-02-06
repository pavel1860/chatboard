from typing import TYPE_CHECKING, Any, Literal
# from chatboard.block.block import block, Block
from chatboard.block import BlockChunk

if TYPE_CHECKING:
    from chatboard.model.postgres.namespace import PostgresNamespace, PgFieldInfo



# @block()
# def create_table_block_from_namespace(blk: Block, namespace: "PostgresNamespace"):
#     with blk(f"CREATE TABLE IF NOT EXISTS {namespace.table_name}", style="func-col") as blk:
#         for field in namespace.iter_fields():
#             blk /= field.name, field.sql_type
#             blk += "NULL" if field.is_optional or field.is_foreign_key else "NOT NULL"            



# @block()
# def create_table_block(blk: Block, name: str, *fields: "PgFieldInfo"):
#     with blk(f"CREATE TABLE IF NOT EXISTS {name}", style="func-col") as blk:
#         for field in fields:
#             blk /= f'"{field.name}"', field.sql_type
#             if field.is_primary_key:
#                 blk += "PRIMARY KEY"
#                 if field.sql_type == "UUID":
#                     blk += "DEFAULT uuid_generate_v4()"
#             else:
#                 # blk += "NULL" if field.is_optional and not field.is_foreign_key else "NOT NULL"            
#                 blk += "NULL" if field.is_optional else "NOT NULL"
            



def create_table_block(name: str, *fields: "PgFieldInfo"):
    with BlockChunk(f"CREATE TABLE IF NOT EXISTS {name}", style="col-tuple", vwrap=("(\n", "\n)"), vsep=",\n") as blk:
    # with Block(f"CREATE TABLE IF NOT EXISTS {name}", style="col-tuple") as blk:
        for field in fields:
            with blk() as f:
                f += f'"{field.name}"', field.sql_type
                if field.is_primary_key:
                    f += "PRIMARY KEY"
                    if field.sql_type == "UUID":
                        f += "DEFAULT uuid_generate_v4()"
                else:
                    # blk += "NULL" if field.is_optional and not field.is_foreign_key else "NOT NULL"            
                    f += "NULL" if field.is_optional else "NOT NULL"                
    return blk