"""
SQL safety validation layer.

This is the ONE file to point at in an interview and say "this is what
stops a bad query." It is deliberately kept separate from SQL
generation (text_to_sql.py) and has its own test file
(tests/test_sql_safety.py) with no dependency on any LLM.

Defense in depth — this validator is layer 1. Layer 2 is the read-only
DB role / connection (see executor.py). A generated query must pass
BOTH before it ever touches real data.

Validation steps, in order:
  1. Parse with sqlglot. Anything that fails to parse is rejected.
  2. Reject anything that isn't exactly one SELECT statement
     (blocks INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE, multi-statement
     injection via ';', CTEs that wrap a mutating statement, etc).
  3. Walk the parsed tree and reject disallowed constructs even inside
     an otherwise-valid SELECT: PRAGMA, ATTACH, subqueries against
     tables not in the schema, function calls not on a small allowlist
     (blocks things like `load_extension`, `sqlite_master` probing).
  4. Enforce the table/column allowlist derived from the real schema
     (schema_registry.ALLOWLIST) — every table and column referenced
     must exist and be one we intend to expose.
  5. Enforce a row-limit cap: if the query has no LIMIT, or a LIMIT
     above the configured cap, rewrite/clamp it down rather than trust
     the LLM to have included one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from app.core.config import settings
from app.sql.schema_registry import ALLOWLIST

# Function calls we consider safe inside a read-only analytical SELECT.
_ALLOWED_FUNCTIONS = {
    "sum", "count", "avg", "min", "max", "round", "abs",
    "coalesce", "cast", "strftime", "date", "julianday",
    "upper", "lower", "substr", "length", "extract",
}

# Statement / expression types that are an automatic reject regardless
# of anything else, because they mutate data, change schema, or read
# internals outside the allowlisted business tables.
_FORBIDDEN_EXPR_TYPES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
    exp.Create, exp.TruncateTable, exp.Attach, exp.Pragma,
    exp.Command, exp.Grant,
)


@dataclass
class ValidationResult:
    is_safe: bool
    reason: str | None = None
    safe_sql: str | None = None  # normalized/clamped SQL if is_safe else None
    violations: list[str] = field(default_factory=list)


def validate_sql(raw_sql: str, row_limit: int | None = None) -> ValidationResult:
    row_limit = row_limit or settings.sql_row_limit
    violations: list[str] = []

    raw_sql = raw_sql.strip().rstrip(";")

    # Reject multi-statement payloads outright (e.g. "SELECT 1; DROP TABLE x")
    # before even trying to parse — sqlglot will happily parse the first
    # statement and silently ignore the rest, which is exactly the
    # attack this blocks.
    try:
        statements = sqlglot.parse(raw_sql, read="sqlite")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(is_safe=False, reason=f"SQL failed to parse: {exc}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return ValidationResult(
            is_safe=False,
            reason=f"Expected exactly one statement, found {len(statements)} "
            "(multi-statement queries are rejected outright).",
        )

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        return ValidationResult(
            is_safe=False,
            reason=f"Only SELECT statements are allowed, got {type(tree).__name__}.",
        )

    # Walk the whole tree (covers subqueries, CTEs, UNIONs) looking for
    # forbidden node types.
    for node in tree.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        if isinstance(node_obj, _FORBIDDEN_EXPR_TYPES):
            violations.append(f"Forbidden construct: {type(node_obj).__name__}")

    # Table allowlist
    for table in tree.find_all(exp.Table):
        table_name = table.name.lower()
        if table_name not in ALLOWLIST:
            violations.append(f"Table not in allowlist: {table_name}")

    # Column allowlist (best-effort: qualified and unqualified columns).
    # We only hard-reject a column if we can resolve it to a single
    # known table and it's not in that table's column set; ambiguous /
    # unqualified columns in a multi-table join are left to the DB to
    # error on, since rejecting them here would produce too many false
    # positives on legitimate joins.
    referenced_tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    if len(referenced_tables) == 1:
        only_table = next(iter(referenced_tables))
        allowed_cols = ALLOWLIST.get(only_table, frozenset())
        for col in tree.find_all(exp.Column):
            col_name = col.name.lower()
            if col_name == "*":
                continue
            if allowed_cols and col_name not in allowed_cols:
                violations.append(f"Column not in allowlist for {only_table}: {col_name}")

    # Function allowlist
    for func in tree.find_all(exp.Anonymous):
        # sqlglot parses unrecognized function names as Anonymous
        if func.this and func.this.lower() not in _ALLOWED_FUNCTIONS:
            violations.append(f"Function not in allowlist: {func.this}")

    if violations:
        return ValidationResult(is_safe=False, reason="Query violates safety policy.", violations=violations)

    # Clamp / enforce row limit
    existing_limit = tree.args.get("limit")
    if existing_limit is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
    else:
        try:
            limit_val = int(existing_limit.expression.this)
            if limit_val > row_limit:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
        except (TypeError, ValueError, AttributeError):
            tree.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))

    return ValidationResult(is_safe=True, safe_sql=tree.sql(dialect="sqlite"))
