




import re
import textwrap

def format_function_multiline(sql: str, function_name: str, indent: int = 8) -> str:
    """
    Formats function arguments onto new lines with consistent indentation.
    e.g., json_build_object(
              'id',
              p.id,
              ...
          )
    """
    pattern = re.compile(
        rf"{function_name}\(([^()]*?(?:\([^()]*?\)[^()]*)*?)\)",
        re.DOTALL
    )

    def split_args(arg_str):
        args = []
        current = ''
        depth = 0
        for char in arg_str:
            if char == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                current += char
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
        if current:
            args.append(current.strip())
        return args

    indent_spaces = ' ' * indent

    def replacer(match):
        args = split_args(match.group(1))
        joined = ",\n".join(f"{indent_spaces}{arg}" for arg in args)
        return f"{function_name}(\n{joined}\n)"

    previous = None
    while previous != sql:
        previous = sql
        sql = pattern.sub(replacer, sql)

    return sql





# def print_sql(sql: str, new_line: bool = False):
#     import sqlparse

#     # Step 1: Format everything else
#     sql = sqlparse.format(sql, reindent=True, keyword_case='upper')

#     # Step 2: Apply function formatting AFTER sqlparse
#     if new_line:
#         for fn in ["COALESCE", "json_agg", "jsonb_build_object"]:
#             sql = format_function_multiline(sql, fn, indent=10)

#     print(sql)



def print_sql(sql: str, only_first: bool = False):
    if only_first:
        import sqlparse
        import re
        # Step 1: Basic formatting with sqlparse
        formatted = sqlparse.format(
            sql,
            reindent=True,
            keyword_case='upper',
            indent_width=4,
            use_space_around_operators=True,
        )
        return formatted
    sql = format_sql(sql)
    print(sql)




# def format_sql(sql: str) -> str:
#     import sqlparse
#     # First, use sqlparse to format the basic structure
#     formatted = sqlparse.format(
#         sql,
#         reindent=True,
#         keyword_case='upper',
#         indent_width=4,
#         use_space_around_operators=True,
#     )

#     # Now handle specific nested functions like COALESCE, json_agg, jsonb_build_object
#     # This regex finds jsonb_build_object calls and pretty prints them line-by-line
#     def format_jsonb(match):
#         inner = match.group(1)
#         # Split key-value pairs and format them
#         parts = re.split(r",\s*'", inner)
#         formatted_parts = []
#         for i, part in enumerate(parts):
#             part = "'" + part if i != 0 else part  # re-add removed quote
#             formatted_parts.append(f"\n            {part.strip()}")
#         return "jsonb_build_object(" + ",".join(formatted_parts) + "\n        )"

#     formatted = re.sub(
#         r"jsonb_build_object\((.*?)\)",
#         format_jsonb,
#         formatted,
#         flags=re.DOTALL
#     )

#     # Format COALESCE and json_agg similarly
#     formatted = re.sub(
#         r"COALESCE\((json_agg.*?)\)",
#         lambda m: "COALESCE(\n        " + m.group(1) + "\n    )",
#         formatted,
#         flags=re.DOTALL
#     )

#     formatted = re.sub(
#         r"json_agg\((DISTINCT.*?)\s+FILTER",
#         lambda m: "json_agg(\n            " + m.group(1) + "\n        ) FILTER",
#         formatted,
#         flags=re.DOTALL
#     )

#     return formatted




def format_sql(sql: str) -> str:
    import sqlparse
    import re
    # Step 1: Basic formatting with sqlparse
    formatted = sqlparse.format(
        sql,
        reindent=True,
        keyword_case='upper',
        indent_width=4,
        use_space_around_operators=True,
    )

    # Step 2: Format jsonb_build_object nicely
    def format_jsonb(match):
        inner = match.group(1)
        parts = re.split(r",\s*'", inner)
        formatted_parts = []
        for i, part in enumerate(parts):
            part = "'" + part if i != 0 else part
            formatted_parts.append(f"\n                {part.strip()}")
        return "jsonb_build_object(" + ",".join(formatted_parts) + "\n            )"

    formatted = re.sub(
        r"jsonb_build_object\((.*?)\)",
        format_jsonb,
        formatted,
        flags=re.DOTALL
    )

    # Step 3: Format json_agg(... FILTER (...)) including DISTINCT if present
    def format_json_agg_with_filter(match):
        inner = match.group(1)
        filter_clause = match.group(2)
        s = "json_agg(\n"
        # s += f"            {inner.strip()}\n"
        s += textwrap.indent(inner.strip(), "            ")
        s += f") FILTER {filter_clause}"
        return s
        # return (
        #     "json_agg(\n"
        #     f"            {inner.strip()}\n"
        #     f"        ) FILTER {filter_clause}"
        # )

    formatted = re.sub(
        r"json_agg\((.*?)\)\s+FILTER\s+(\(.*?\))",
        format_json_agg_with_filter,
        formatted,
        flags=re.DOTALL
    )

    # Step 4: Indent COALESCE with new lines
    formatted = re.sub(
        r"COALESCE\(\s*(json_agg\(.*?\)\s+FILTER\s+\(.*?\))\s*,\s*('.*?')\s*\)",
        lambda m: "COALESCE(\n        " + m.group(1).strip() + ",\n        " + m.group(2) + "\n    )",
        formatted,
        flags=re.DOTALL
    )

    return formatted



