# graph_ra.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import itertools

# ========= Types =========

class SqlType:
    def __init__(self, name: str): self.name = name
    def __repr__(self): return self.name

INT = SqlType("INTEGER")
BIGINT = SqlType("BIGINT")
TEXT = SqlType("TEXT")
BOOL = SqlType("BOOLEAN")
NUMERIC = SqlType("NUMERIC")
TIMESTAMPTZ = SqlType("TIMESTAMPTZ")

# ========= Graph Primitives =========

_node_id_counter = itertools.count(1)

@dataclass(frozen=True)
class NodeId:
    value: int

def new_node_id() -> NodeId:
    return NodeId(next(_node_id_counter))

# ========= Schema =========

@dataclass(eq=False)   # <- identity equality; object hash works
class Table:
    name: str
    columns: dict[str, "Column"] = field(default_factory=dict)

    def col(self, name: str, type_, nullable: bool=True) -> "Column":
        c = Column(self, name, type_, nullable)
        self.columns[name] = c
        return c

    @property
    def c(self):
        # dot-access: users.c.id
        return type("Cols", (), self.columns)

@dataclass(frozen=True)
class Column:
    table: Table
    name: str
    type_: SqlType
    nullable: bool = True

# ========= Expressions (terms & predicates) =========

class Term: ...
@dataclass(frozen=True)  # column reference term
class Col(Term): col: Column
@dataclass(frozen=True)
class Lit(Term): value: Any
@dataclass(frozen=True)  # scalar subquery term (1 row, 1 column)
class Scalar(Term): rel: "Relation"

class Pred:
    def __and__(self, other: "Pred") -> "Pred": return Bool("AND", self, other)
    def __or__(self, other: "Pred") -> "Pred":  return Bool("OR", self, other)
    def __invert__(self) -> "Pred":            return Not(self)

@dataclass(frozen=True)
class Bin(Pred): op: str; left: Term; right: Term
@dataclass(frozen=True)
class Bool(Pred): op: str; left: Pred; right: Pred
@dataclass(frozen=True)
class Not(Pred): inner: Pred
@dataclass(frozen=True)
class InList(Pred): term: Term; values: List[Any]
@dataclass(frozen=True)
class Exists(Pred): rel: "Relation"; correlation: Optional[Pred]=None

def t(x: Any) -> Term:
    # ✅ Correct wrapping: Columns become Col(...), not Lit(...)
    if isinstance(x, Term):
        return x
    if isinstance(x, Column):
        return Col(x)
    return Lit(x)


# Short predicate helpers (nice ergonomics)
def EQ(a: Column, b: Any) -> Pred: return Bin("=", Col(a), t(b))
def NE(a: Column, b: Any) -> Pred: return Bin("<>", Col(a), t(b))
def LT(a: Column, b: Any) -> Pred: return Bin("<", Col(a), t(b))
def LE(a: Column, b: Any) -> Pred: return Bin("<=", Col(a), t(b))
def GT(a: Column, b: Any) -> Pred: return Bin(">", Col(a), t(b))
def GE(a: Column, b: Any) -> Pred: return Bin(">=", Col(a), t(b))
def LIKE(a: Column, s: str) -> Pred: return Bin("LIKE", Col(a), Lit(s))
def ILIKE(a: Column, s: str) -> Pred: return Bin("ILIKE", Col(a), Lit(s))
@dataclass(frozen=True)
class InList(Pred):
    term: Term
    values: List[Any]  # should be primitives, not Columns



@dataclass(frozen=True)
class Func(Term):
    name: str
    args: List[Term]

def FUNC(name: str, *args: Any) -> Func:
    return Func(name, [t(a) for a in args])   # uses your existing t() to wrap args

# 2) Aliased select item (so Project can select expressions with AS ...)
@dataclass(frozen=True)
class SelectItem:
    term: Term
    alias: Optional[str] = None

def COL(c: Column, alias: Optional[str] = None) -> SelectItem:
    return SelectItem(Col(c), alias)

def EXPR(term: Term, alias: str) -> SelectItem:
    return SelectItem(term, alias)

# 3) Correlated scalar subquery. Use when the inner SELECT needs to reference outer tables.
@dataclass(frozen=True)
class ScalarSubquery(Term):
    rel: Relation
    where: Optional[Pred] = None

# ========= Relation graph (FBP-style) =========

class Relation:
    """Base class for all relation nodes. Immutable; composing yields a DAG."""
    __slots__ = ("id",)
    def __init__(self):
        # Bypass frozen dataclass __setattr__ in subclasses
        object.__setattr__(self, "id", new_node_id())

    # ---- Fluent composition helpers ----
    def select(self, pred: Pred) -> "Relation": return Select(self, pred)
    def project(self, *cols: Column) -> "Relation": return Project(self, [Col(c) for c in cols])
    def join(self, other: "Relation", on: Pred) -> "Relation": return Join(self, other, on)
    def cross(self, other: "Relation") -> "Relation": return Cross(self, other)
    def union(self, other: "Relation", all: bool=False) -> "Relation": return Union(self, other, all)
    def diff(self, other: "Relation") -> "Relation": return Diff(self, other)
    def intersect(self, other: "Relation") -> "Relation": return Intersect(self, other)
    def groupby(self, keys: Sequence[Column], aggs: Sequence["Agg"]) -> "Relation":
        return GroupBy(self, [Col(k) for k in keys], aggs)
    def orderby(self, *keys: Tuple[Column, str]) -> "Relation":
        return OrderBy(self, [(Col(c), dir_) for (c, dir_) in keys])
    def limit(self, n: int, offset: Optional[int]=None) -> "Relation": return Limit(self, n, offset)
    def rename(self, alias: str) -> "Relation": return Rename(self, alias)

# Node types:

@dataclass(frozen=True)
class From(Relation):
    table: Table
    alias: Optional[str]=None
    def __init__(self, table: Table, alias: Optional[str]=None):
        super().__init__()
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "alias", alias)

@dataclass(frozen=True)
class Subquery(Relation):
    rel: Relation
    alias: Optional[str]=None
    def __init__(self, rel: Relation, alias: Optional[str]=None):
        super().__init__()
        object.__setattr__(self, "rel", rel)
        object.__setattr__(self, "alias", alias)

@dataclass(frozen=True)
class Select(Relation):
    rel: Relation
    pred: Pred
    def __init__(self, rel: Relation, pred: Pred):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "pred", pred)

@dataclass(frozen=True)
class Project(Relation):
    rel: Relation
    attrs: Sequence[Col]
    def __init__(self, rel: Relation, attrs: Sequence[Col]):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "attrs", tuple(attrs))

@dataclass(frozen=True)
class Rename(Relation):
    rel: Relation
    alias: str
    def __init__(self, rel: Relation, alias: str):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "alias", alias)

@dataclass(frozen=True)
class Union(Relation):
    left: Relation; right: Relation; all: bool=False
    def __init__(self, left: Relation, right: Relation, all: bool=False):
        super().__init__(); object.__setattr__(self, "left", left); object.__setattr__(self, "right", right); object.__setattr__(self, "all", all)

@dataclass(frozen=True)
class Diff(Relation):
    left: Relation; right: Relation
    def __init__(self, left: Relation, right: Relation):
        super().__init__(); object.__setattr__(self, "left", left); object.__setattr__(self, "right", right)

@dataclass(frozen=True)
class Intersect(Relation):
    left: Relation; right: Relation
    def __init__(self, left: Relation, right: Relation):
        super().__init__(); object.__setattr__(self, "left", left); object.__setattr__(self, "right", right)

@dataclass(frozen=True)
class Cross(Relation):
    left: Relation; right: Relation
    def __init__(self, left: Relation, right: Relation):
        super().__init__(); object.__setattr__(self, "left", left); object.__setattr__(self, "right", right)

@dataclass(frozen=True)
class Join(Relation):
    left: Relation; right: Relation; on: Pred
    def __init__(self, left: Relation, right: Relation, on: Pred):
        super().__init__(); object.__setattr__(self, "left", left); object.__setattr__(self, "right", right); object.__setattr__(self, "on", on)

@dataclass(frozen=True)
class Agg:
    func: str               # SUM, COUNT, AVG, MIN, MAX
    term: Optional[Term]    # None → COUNT(*)
    alias: str

def COUNT(alias: str) -> Agg: return Agg("COUNT", None, alias)
def SUM(col: Column, alias: str) -> Agg: return Agg("SUM", Col(col), alias)
def AVG(col: Column, alias: str) -> Agg: return Agg("AVG", Col(col), alias)
def MIN(col: Column, alias: str) -> Agg: return Agg("MIN", Col(col), alias)
def MAX(col: Column, alias: str) -> Agg: return Agg("MAX", Col(col), alias)

@dataclass(frozen=True)
class GroupBy(Relation):
    rel: Relation; keys: Sequence[Col]; aggs: Sequence[Agg]
    def __init__(self, rel: Relation, keys: Sequence[Col], aggs: Sequence[Agg]):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "keys", tuple(keys)); object.__setattr__(self, "aggs", tuple(aggs))

@dataclass(frozen=True)
class OrderBy(Relation):
    rel: Relation; keys: Sequence[Tuple[Col, str]]
    def __init__(self, rel: Relation, keys: Sequence[Tuple[Col,str]]):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "keys", tuple(keys))

@dataclass(frozen=True)
class Limit(Relation):
    rel: Relation; n: int; offset: Optional[int]=None
    def __init__(self, rel: Relation, n: int, offset: Optional[int]=None):
        super().__init__(); object.__setattr__(self, "rel", rel); object.__setattr__(self, "n", n); object.__setattr__(self, "offset", offset)

# WITH/CTE (let-binding) and CTE references
@dataclass(frozen=True)
class With(Relation):
    bindings: Dict[str, Relation]  # name -> subgraph
    body: Relation
    materialized: Optional[Dict[str, bool]] = None  # PG ≥ 12: MATERIALIZED / NOT MATERIALIZED
    def __init__(self, bindings: Dict[str, Relation], body: Relation, materialized: Optional[Dict[str,bool]]=None):
        super().__init__(); object.__setattr__(self, "bindings", dict(bindings)); object.__setattr__(self, "body", body); object.__setattr__(self, "materialized", materialized or {})

@dataclass(frozen=True)
class CTE(Relation):
    name: str
    alias: Optional[str]=None
    def __init__(self, name: str, alias: Optional[str]=None):
        super().__init__(); object.__setattr__(self, "name", name); object.__setattr__(self, "alias", alias)


@dataclass(frozen=True)
class Cast(Term):
    term: Term
    type_sql: str




# ========= SQL Compiler (PostgreSQL, parameterized) =========




class SQL:
    def __init__(self, text: str, params: List[Any]):
        self.text, self.params = text, params
    def __repr__(self):
        return f"SQL({self.text!r}, params={self.params})"

class Compiler:
    def __init__(self):
        self.params: List[Any] = []
        self.alias_counter = 0

    # ---------- public ----------
    def compile(self, rel: Relation) -> SQL:
        self.params.clear()
        self.alias_counter = 0
        sql, _env = self._emit(rel, env={}, top_level=True)  # <-- top-level render
        return SQL(sql, list(self.params))

    # ---------- helpers ----------
    def _q(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'
    def _p(self, v: Any) -> str:
        # 🚫 Columns must never be bound as parameters
        if isinstance(v, Column):
            raise TypeError(
                f"Attempted to bind Column {v} as a parameter. "
                "This usually means t() is wrapping Columns as Literals. "
                "Ensure t() returns Col(x) for Column inputs."
            )
        self.params.append(v)
        return f"${len(self.params)}"
    def _alias(self, base: str = "t") -> str:
        self.alias_counter += 1; return f"{base}{self.alias_counter}"
    def _indent(self, s: str, n: int = 2) -> str:
        pad = " " * n; return "\n".join(pad + line if line else line for line in s.splitlines())
    def _join_lines(self, *lines: str) -> str:
        return "\n".join([ln for ln in lines if ln is not None and ln != ""])

    # ---------- term/predicate ----------
    def _term(self, term: Term, env: Dict[Table, str]) -> str:
        if isinstance(term, Col):
            alias = env.get(term.col.table)
            if not alias:
                alias = self._alias(term.col.table.name[:1] or "t")
            return f"{alias}.{self._q(term.col.name)}"
        if isinstance(term, Lit):
            return self._p(term.value)
        if isinstance(term, Scalar):
            inner_sql, _ = self._emit(term.rel, env.copy(), top_level=False)
            return f"(SELECT * FROM {inner_sql})"
        if isinstance(term, Func):
            args_sql = ", ".join(self._term(a, env) for a in term.args)
            return f"{term.name.upper()}({args_sql})"

        if isinstance(term, ScalarSubquery):
            inner_sql, inner_env = self._emit(term.rel, env.copy(), top_level=False)
            if term.where is None:
                return f"(SELECT * FROM {inner_sql})"
            # correlated WHERE: merge envs so outer columns resolve
            merged_env = {**inner_env, **env}
            where_sql = self._pred(term.where, merged_env)
            inner_block = self._join_lines("SELECT *", f"FROM {inner_sql}", f"WHERE {where_sql}")
            return f"({inner_block})"
        if isinstance(term, Cast):
            return f"{self._term(term.term, env)}::{term.type_sql}"
        raise TypeError(f"Unknown term {term}")

    def _pred(self, pred: Pred, env: Dict[Table, str]) -> str:
        if isinstance(pred, Bin):
            return f"{self._term(pred.left, env)} {pred.op} {self._term(pred.right, env)}"
        if isinstance(pred, Bool):
            return f"({self._pred(pred.left, env)} {pred.op} {self._pred(pred.right, env)})"
        if isinstance(pred, Not):
            return f"(NOT {self._pred(pred.inner, env)})"
        if isinstance(pred, InList):
            items = ", ".join(self._p(v) for v in pred.values)
            return f"{self._term(pred.term, env)} IN ({items})"
        if isinstance(pred, Exists):
            inner_sql, inner_env = self._emit(pred.rel, env.copy(), top_level=False)
            where = f"\nWHERE {self._pred(pred.correlation, inner_env)}" if pred.correlation else ""
            return "EXISTS (" + self._join_lines("SELECT 1", f"FROM {inner_sql}{where}") + ")"
        raise TypeError(f"Unknown predicate {pred}")

    # ---------- SELECT formatter ----------
    def _fmt_select(self, select_list: str, from_src: str,
                    *, where: Optional[str]=None, group_by: Optional[str]=None,
                    order_by: Optional[str]=None, limit: Optional[str]=None,
                    wrap_alias: Optional[str]=None, top_level: bool=False) -> str:
        lines = [f"SELECT {select_list}", f"FROM {from_src}"]
        if where:    lines.append(f"WHERE {where}")
        if group_by: lines.append(f"GROUP BY {group_by}")
        if order_by: lines.append(f"ORDER BY {order_by}")
        if limit:    lines.append(f"LIMIT {limit}")
        block = self._join_lines(*lines)
        if top_level:
            return block
        # wrap as subquery
        al = f" AS {wrap_alias}" if wrap_alias else ""
        return f"(\n{self._indent(block)}\n){al}"
        
    def _collapse_env(self, src_env: dict, wrapper_alias: str) -> dict:
        new = dict(src_env)
        for tbl in list(src_env.keys()):
            new[tbl] = wrapper_alias
        return new

    # ---------- emit relations (returns (sql, env)) ----------
    def _emit(self, r: Relation, env: Dict[Table, str], top_level: bool) -> Tuple[str, Dict[Table, str]]:
        # FROM base table
        if isinstance(r, From):
            alias = r.alias or self._alias(r.table.name[:1] or "t")
            new_env = env.copy(); new_env[r.table] = alias
            return f'{self._q(r.table.name)} AS {alias}', new_env

        if isinstance(r, Subquery):
            inner_sql, inner_env = self._emit(r.rel, env.copy(), top_level=False)
            alias = r.alias or self._alias("sq")
            wrapped = f"(\n{self._indent(self._join_lines('SELECT *', f'FROM {inner_sql}'))}\n) AS {alias}"
            return wrapped, self._collapse_env(inner_env, alias)

        # Select
        if isinstance(r, Select):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            where = self._pred(r.pred, src_env)
            if top_level:
                return self._fmt_select("*", src_sql, where=where, top_level=True), src_env
            alias = self._alias("sel")
            wrapped = self._fmt_select("*", src_sql, where=where, wrap_alias=alias, top_level=False)
            return wrapped, self._collapse_env(src_env, alias)
        
        # Project
        if isinstance(r, Project):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            def _sel(item: SelectItem) -> str:
                base = self._term(item.term, src_env)
                return f'{base} AS "{item.alias}"' if item.alias else base
            cols = ", ".join(_sel(x) for x in r.attrs) if r.attrs else "*"
            if top_level:
                return self._fmt_select(cols, src_sql, top_level=True), src_env
            alias = self._alias("pr")
            wrapped = self._fmt_select(cols, src_sql, wrap_alias=alias, top_level=False)
            # collapse env after wrapping
            return wrapped, self._collapse_env(src_env, alias)

        if isinstance(r, Rename):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            # Replace trailing alias if present; otherwise append new alias
            if src_sql.strip().startswith("(") and " AS " in src_sql.strip():
                pre, _, _old = src_sql.strip().rpartition(" AS ")
                return (f"{pre} AS {r.alias}", src_env)
            return (f"{src_sql} AS {r.alias}", src_env)

        if isinstance(r, Union):
            l_sql, _ = self._emit(r.left, env.copy(), top_level=False)
            r_sql, _ = self._emit(r.right, env.copy(), top_level=False)
            op = "UNION ALL" if r.all else "UNION"
            if top_level:
                text = self._join_lines(
                    self._join_lines("SELECT *", f"FROM {l_sql}"),
                    op,
                    self._join_lines("SELECT *", f"FROM {r_sql}"),
                )
                return text, env
            alias = self._alias("u")
            text = self._join_lines(
                "(",
                self._indent(self._join_lines("SELECT *", f"FROM {l_sql}")),
                f") {op} (",
                self._indent(self._join_lines("SELECT *", f"FROM {r_sql}")),
                f") AS {alias}"
            )
            return text, env

        if isinstance(r, Diff) or isinstance(r, Intersect):
            l_sql, _ = self._emit(r.left, env.copy(), top_level=False)
            r_sql, _ = self._emit(r.right, env.copy(), top_level=False)
            op = "EXCEPT" if isinstance(r, Diff) else "INTERSECT"
            if top_level:
                text = self._join_lines(
                    self._join_lines("SELECT *", f"FROM {l_sql}"),
                    op,
                    self._join_lines("SELECT *", f"FROM {r_sql}"),
                )
                return text, env
            alias = self._alias("d" if op == "EXCEPT" else "i")
            text = self._join_lines(
                "(",
                self._indent(self._join_lines("SELECT *", f"FROM {l_sql}")),
                f") {op} (",
                self._indent(self._join_lines("SELECT *", f"FROM {r_sql}")),
                f") AS {alias}"
            )
            return text, env

        if isinstance(r, Cross):
            l_sql, l_env = self._emit(r.left, env.copy(), top_level=False)
            r_sql, r_env = self._emit(r.right, env.copy(), top_level=False)
            merged = {**l_env, **r_env}
            block = self._join_lines("SELECT *", f"FROM {l_sql}", f"CROSS JOIN {r_sql}")
            if top_level:
                return block, merged
            alias = self._alias("x")
            return f"(\n{self._indent(block)}\n) AS {alias}", merged

        # Join
        if isinstance(r, Join):
            l_sql, l_env = self._emit(r.left, env.copy(), top_level=False)
            r_sql, r_env = self._emit(r.right, env.copy(), top_level=False)
            merged = {**l_env, **r_env}
            on = self._pred(r.on, merged)
            block = self._join_lines("SELECT *", f"FROM {l_sql}", f"JOIN {r_sql}", f"  ON {on}")
            if top_level:
                return block, merged
            alias = self._alias("j")
            wrapped = f"(\n{self._indent(block)}\n) AS {alias}"
            return wrapped, self._collapse_env(merged, alias)

        if isinstance(r, GroupBy):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            key_sqls = [self._term(k, src_env) for k in r.keys]
            sel = []
            if key_sqls: sel.extend(key_sqls)
            for a in r.aggs:
                if a.func.upper() == "COUNT" and a.term is None:
                    sel.append(f'COUNT(*) AS "{a.alias}"')
                else:
                    sel.append(f'{a.func.upper()}({self._term(a.term, src_env)}) AS "{a.alias}"')
            select_list = ", ".join(sel) if sel else "*"
            gby = ", ".join(key_sqls) if key_sqls else None
            if top_level:
                return self._fmt_select(select_list, src_sql, group_by=gby, top_level=True), src_env
            alias = self._alias("g")
            wrapped = self._fmt_select(select_list, src_sql, group_by=gby, wrap_alias=alias, top_level=False)
            return wrapped, self._collapse_env(src_env, alias)

        # OrderBy
        if isinstance(r, OrderBy):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            keys = ", ".join(f"{self._term(k, src_env)} {d}" for (k, d) in r.keys)
            if top_level:
                return self._fmt_select("*", src_sql, order_by=keys, top_level=True), src_env
            alias = self._alias("o")
            wrapped = self._fmt_select("*", src_sql, order_by=keys, wrap_alias=alias, top_level=False)
            return wrapped, self._collapse_env(src_env, alias)

        # Limit
        if isinstance(r, Limit):
            src_sql, src_env = self._emit(r.rel, env.copy(), top_level=False)
            lim = str(r.n) + (f"\nOFFSET {r.offset}" if r.offset else "")
            if top_level:
                return self._fmt_select("*", src_sql, limit=lim, top_level=True), src_env
            alias = self._alias("l")
            wrapped = self._fmt_select("*", src_sql, limit=lim, wrap_alias=alias, top_level=False)
            return wrapped, self._collapse_env(src_env, alias)

        if isinstance(r, With):
            # Build each CTE
            cte_blocks = []
            for name, expr in r.bindings.items():
                body_sql, _ = self._emit(expr, env.copy(), top_level=False)
                cte_blocks.append(self._join_lines(
                    f'{self._q(name)} AS (',
                    self._indent(self._join_lines("SELECT *", f"FROM {body_sql}")),
                    ')'
                ))
            ctes = ",\n".join(cte_blocks)
            body_sql, body_env = self._emit(r.body, env.copy(), top_level=False)
            with_block = self._join_lines(
                "WITH",
                self._indent(ctes, 2),
                self._join_lines("SELECT *", f"FROM {body_sql}")
            )
            if top_level:
                return with_block, body_env
            alias = self._alias("w")
            wrapped = f"(\n{self._indent(with_block)}\n) AS {alias}"
            # collapse body env to wrapper alias
            return wrapped, self._collapse_env(body_env, alias)

        if isinstance(r, CTE):
            alias = r.alias or self._alias(r.name[:1] or "c")
            return f"{self._q(r.name)} AS {alias}", env

        raise TypeError(f"Unknown relation node {r}")
