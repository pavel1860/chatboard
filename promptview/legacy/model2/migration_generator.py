import importlib
import sys

def discover_and_summarize_models(module: str | None = None):
    """
    Discover all registered ORM models and print a summary of their schema.
    If module is provided, import it to ensure models are registered.
    """
    if module:
        print(f"Importing models from module: {module}")
        importlib.import_module(module)
    else:
        print("Importing default model registry (promptview.model2.namespace_manager)")
        import promptview.model.namespace_manager as nm

    try:
        from promptview.model.namespace_manager import NamespaceManager
    except ImportError:
        print("Could not import NamespaceManager. Make sure your PYTHONPATH is set correctly.")
        sys.exit(1)

    namespaces = NamespaceManager.get_all_namespaces()
    print(f"Discovered {len(namespaces)} namespaces:")

    for ns in namespaces:
        print(f"\nTable: {getattr(ns, 'table_name', getattr(ns, 'name', None))}")

        # Fields
        print("  Fields:")
        for field in ns.iter_fields():
            field_desc = f"    - {field.name}: {getattr(field, 'field_type', None)}"
            if getattr(field, 'is_primary_key', False):
                field_desc += " [PK]"
            if getattr(field, 'is_optional', False):
                field_desc += " [NULLABLE]"
            if getattr(field, 'default', None) is not None:
                field_desc += f" [DEFAULT={field.default}]"
            if getattr(field, 'is_enum', False):
                field_desc += f" [ENUM={getattr(field, 'enum_name', None)}]"
            if getattr(field, 'index', False):
                field_desc += " [INDEX]"
            if getattr(field, 'unique', False):
                field_desc += " [UNIQUE]"
            if getattr(field, 'is_vector', False):
                field_desc += " [VECTOR]"
            print(field_desc)

        # Indexes (including composite)
        indexes = getattr(ns, 'indexes', [])
        if indexes:
            print("  Indexes:")
            for idx in indexes:
                idx_desc = f"    - {getattr(idx, 'name', None)}: columns={getattr(idx, 'columns', None)}"
                if getattr(idx, 'unique', False):
                    idx_desc += " [UNIQUE]"
                print(idx_desc)

        # Enums
        enums = []
        for field in ns.iter_fields():
            if getattr(field, 'is_enum', False):
                enums.append(getattr(field, 'enum_name', None))
        if enums:
            print(f"  Enums: {', '.join(set([str(e) for e in enums if e]))}")

        # Relations
        if hasattr(ns, 'iter_relations'):
            rels = list(ns.iter_relations())
            if rels:
                print("  Relations:")
                for rel in rels:
                    print(f"    - {rel.name}: {rel.primary_key} -> {rel.foreign_cls.__name__ if hasattr(rel.foreign_cls, '__name__') else rel.foreign_cls} ({rel.foreign_key})")

        # Vector fields (for Qdrant)
        vector_fields = [f.name for f in ns.iter_fields() if getattr(f, 'is_vector', False)]
        if vector_fields:
            print(f"  Vector fields: {', '.join(vector_fields)}")

        # Indexes, enums, etc. can be added here as needed 

def generate_migration_code(
    namespaces,
    revision: str,
    down_revision: str | None,
    backend: str = "postgres"
) -> str:
    """
    Generate the migration file content as a string.
    """
    lines = [
        "from alembic import op",
        "import sqlalchemy as sa",
        "import sqlalchemy.dialects.postgresql as pg",
        "",
        f'revision = "{revision}"',
        f'down_revision = {repr(down_revision)}',
        "branch_labels = None",
        "depends_on = None",
        "",
        "def upgrade():"
    ]

    # 1. Create tables (no raw SQL for enums)
    for ns in namespaces:
        table_name = getattr(ns, "table_name", ns.name)
        lines.append(f"    op.create_table('{table_name}',")
        for field in ns.iter_fields():
            sql_type = getattr(field, 'sql_type', 'String')
            # Handle dialect-specific types
            if sql_type.upper() == "JSONB":
                col_type = "pg.JSONB"
            else:
                col_type = f"sa.{sql_type}"
            # Handle enum columns
            if getattr(field, "is_enum", False):
                enum_name = getattr(field, "enum_name", None)
                enum_values = getattr(field, "get_enum_values_safe", lambda: [])()
                if enum_name and enum_values:
                    col_def = f"        sa.Column('{field.name}', sa.Enum({', '.join([repr(v) for v in enum_values])}, name='{enum_name}'),"
                else:
                    col_def = f"        sa.Column('{field.name}', {col_type},"
            # Handle integer PKs with autoincrement
            elif field.name == 'id' and getattr(field, 'is_primary_key', False) and sql_type.upper() in ('SERIAL', 'INTEGER', 'INT'):
                col_def = f"        sa.Column('id', sa.Integer, primary_key=True, nullable=False, autoincrement=True),"
                lines.append(col_def)
                continue
            else:
                col_def = f"        sa.Column('{field.name}', {col_type},"
            if getattr(field, "is_primary_key", False) and field.name != 'id':
                col_def += " primary_key=True,"
            if getattr(field, "is_optional", False):
                col_def += " nullable=True,"
            else:
                col_def += " nullable=False,"
            if getattr(field, "default", None) is not None:
                col_def += f" server_default=sa.text('{field.default}'),"
            col_def = col_def.rstrip(',') + "),"
            lines.append(col_def)
        lines.append("    )")

    # 2. Create indexes
    for ns in namespaces:
        for field in ns.iter_fields():
            if getattr(field, "index", False):
                idx_name = f"ix_{ns.table_name}_{field.name}"
                lines.append(f"    op.create_index('{idx_name}', '{ns.table_name}', ['{field.name}'])")
        for idx in getattr(ns, "indexes", []):
            idx_cols = getattr(idx, "columns", [])
            unique = getattr(idx, "unique", False)
            lines.append(f"    op.create_index('{idx.name}', '{ns.table_name}', {idx_cols}, unique={unique})")

    # 3. Create foreign keys
    for ns in namespaces:
        if hasattr(ns, "iter_relations"):
            for rel in ns.iter_relations():
                # lines.append(
                #     f"    op.create_foreign_key("
                #     f"'{rel.name}_fk', '{ns.table_name}', '{rel.foreign_cls.__name__.lower()}s', "
                #     f"['{rel.primary_key}'], ['{rel.foreign_key}'], "
                #     f"ondelete='{rel.on_delete}', onupdate='{rel.on_update}')"
                # )
                lines.append(
                    f"    op.create_foreign_key("
                    f"'{rel.name}_fk', '{rel.foreign_namespace.table_name}', '{ns.table_name}', "
                    f"['{rel.foreign_key}'], ['{rel.primary_key}'], "
                    f"ondelete='{rel.on_delete}', onupdate='{rel.on_update}')"
                )

    # 4. Qdrant support (if backend == 'qdrant' or 'both')
    if backend in ("qdrant", "both"):
        lines.append("    # Qdrant collection creation would go here")

    # Downgrade
    lines.append("")
    lines.append("def downgrade():")

    # Drop indexes (reverse order)
    for ns in reversed(namespaces):
        for field in ns.iter_fields():
            if getattr(field, "index", False):
                idx_name = f"ix_{ns.table_name}_{field.name}"
                lines.append(f"    op.drop_index('{idx_name}', table_name='{ns.table_name}')")
        for idx in getattr(ns, "indexes", []):
            lines.append(f"    op.drop_index('{idx.name}', table_name='{ns.table_name}')")

    # Drop tables (reverse order)
    for ns in reversed(namespaces):
        table_name = getattr(ns, "table_name", ns.name)
        lines.append(f"    op.drop_table('{table_name}')")

    # Drop enums (reverse order, deduplicated)
    enum_names = []
    for ns in namespaces:
        for field in ns.iter_fields():
            if getattr(field, "is_enum", False):
                enum_name = getattr(field, "enum_name")
                if enum_name and enum_name not in enum_names:
                    enum_names.append(enum_name)
    for enum_name in reversed(enum_names):
        lines.append(f"    op.execute('DROP TYPE IF EXISTS {enum_name};')")

    return '\n'.join(lines) 