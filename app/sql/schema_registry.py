"""
Single source of truth for "what tables/columns exist". Both the
Text-to-SQL prompt (what schema we show the LLM) and the safety
validator's allowlist (what tables/columns we PERMIT) are derived from
here, from the same SQLAlchemy models — so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.sql.models import Base


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: frozenset[str]


def build_allowlist() -> dict[str, frozenset[str]]:
    """table_name -> frozenset of permitted column names."""
    allowlist: dict[str, frozenset[str]] = {}
    for mapper in Base.registry.mappers:
        table = mapper.local_table
        allowlist[table.name] = frozenset(c.name for c in table.columns)
    return allowlist


def build_schema_prompt() -> str:
    """Human/LLM-readable schema description for the Text-to-SQL system prompt."""
    lines = []
    for mapper in Base.registry.mappers:
        table = mapper.local_table
        cols = ", ".join(f"{c.name} {c.type}" for c in table.columns)
        lines.append(f"TABLE {table.name} ({cols})")
    return "\n".join(lines)


ALLOWLIST = build_allowlist()
SCHEMA_PROMPT = build_schema_prompt()
