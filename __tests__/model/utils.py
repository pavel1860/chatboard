


def normalize_sql(sql: str) -> str:
    import sqlparse
    return sqlparse.format(sql.strip(), reindent=True, keyword_case="upper")


def assert_sql(query, expected_sql: str, expected_params: list, compiler_cls=None):
    from chatboard.model.postgres.sql.compiler import Compiler

    compiler = (compiler_cls or Compiler)()
    actual_sql, actual_params = compiler.compile(query)

    formatted_actual = normalize_sql(actual_sql)
    formatted_expected = normalize_sql(expected_sql)

    print("\n⏎ Formatted SQL:\n", formatted_actual)
    print("\n🔑 Params:", actual_params)

    if formatted_actual != formatted_expected:
        print("\n Expected SQL:\n", formatted_expected)
        
    assert formatted_actual == formatted_expected, "SQL does not match"
    assert actual_params == expected_params, "Parameters do not match"

