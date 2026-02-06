import argparse
from chatboard.model.migration_generator import discover_and_summarize_models, generate_migration_code


def main():
    parser = argparse.ArgumentParser(description="Generate migration file from ORM models.")
    parser.add_argument('--output', type=str, required=True, help='Path to write the migration file')
    parser.add_argument('--revision', type=str, required=False, help='Revision id (default: auto-generated)')
    parser.add_argument('--down-revision', type=str, required=False, help='Previous revision id')
    parser.add_argument('--backend', type=str, choices=['postgres', 'qdrant', 'both'], default='postgres', help='Target backend(s)')
    parser.add_argument('--module', type=str, required=False, help='Module path to import models from')

    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    # Discover and summarize models
    if args.module:
        import importlib
        importlib.import_module(args.module)
    from chatboard.model.namespace_manager import NamespaceManager
    namespaces = NamespaceManager.get_all_namespaces()
    discover_and_summarize_models(args.module)

    # Generate migration code
    revision = args.revision or "auto_revision_id"
    down_revision = args.down_revision or None
    migration_code = generate_migration_code(
        namespaces,
        revision=revision,
        down_revision=down_revision,
        backend=args.backend
    )

    # Write to output file
    with open(args.output, 'w') as f:
        f.write(migration_code)
    print(f"Migration file written to {args.output}")

if __name__ == '__main__':
    main() 