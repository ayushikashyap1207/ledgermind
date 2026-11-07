"""
Text-to-SQL generation. Deliberately dumb and separate from safety.py —
this module's only job is "question + schema -> candidate SQL string".
It is never trusted on its own; every output is passed to
safety.validate_sql() before anything touches the database (see
app/sql/pipeline.py for the wiring).

There is intentionally no hardcoded fallback here. If the configured LLM
provider fails, the exception propagates so callers can fail closed
(return an execution error / abstain) instead of answering with SQL that
was never generated for the question.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.llm import LLMClient, get_llm
from app.sql.schema_registry import SCHEMA_PROMPT

SYSTEM_PROMPT = """You are a SQL generator for a financial database (SQLite dialect).

Given the schema below and a user question, output ONLY a single SELECT
statement. No explanation, no markdown fences, no semicolon-separated
statements. If the question cannot be answered with a read-only SELECT
against this schema, output exactly: NOT_ANSWERABLE

IMPORTANT ACCOUNTING RULES:
- account_type values are lowercase: asset, liability, equity, revenue, expense.
- The transactions table represents the transaction's debit-side account through
  transactions.account_id.
- The general_ledger table contains the individual debit and credit legs of
  each transaction.
- For revenue totals, query general_ledger and SUM(credit) for accounts whose
  account_type = 'revenue'.
- For expense totals, query general_ledger and SUM(debit) for accounts whose
  account_type = 'expense'.
- For account-type financial totals, prefer general_ledger because it preserves
  debit/credit direction.
- Do not assume that transactions.account_id represents the revenue account
  when answering revenue questions.

QUERY RULES:
- "Overdue" invoices means invoices.status = 'overdue'. Do not infer overdue
  from due_date unless the user explicitly asks for invoices past their due date.
- In a join, qualify every column with its table alias (for example
  i.vendor_id and v.vendor_id). Never leave a column ambiguous.
- Never use SELECT *. Select only the columns needed to answer the question.
- For ranking questions ("which vendor has the highest ..."), include the
  ranked metric in the SELECT with an alias, for example:
  SELECT v.vendor_name, SUM(i.amount) AS total ... ORDER BY total DESC LIMIT 1
- Return exactly one SELECT statement. Do not add a semicolon or any explanation.

Schema:
{schema}
""".strip()

@dataclass
class SQLGenerationResult:
    question: str
    raw_sql: str | None
    not_answerable: bool


def _parse_completion(question: str, completion: str) -> SQLGenerationResult:
    completion = (completion or "").strip()

    # Strip accidental markdown fences — small models love to add these
    # even when told not to.
    if completion.startswith("```"):
        completion = completion.strip("`")
        if completion.lower().startswith("sql"):
            completion = completion[3:].strip()

    if not completion or completion.upper() == "NOT_ANSWERABLE":
        return SQLGenerationResult(question=question, raw_sql=None, not_answerable=True)

    return SQLGenerationResult(question=question, raw_sql=completion, not_answerable=False)


def generate_sql(question: str, llm: LLMClient | None = None) -> SQLGenerationResult:
    llm = llm or get_llm()
    system = SYSTEM_PROMPT.format(schema=SCHEMA_PROMPT)
    completion = llm.complete(system=system, user=question)
    return _parse_completion(question, completion)


REPAIR_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

REPAIR MODE:
The previous SELECT failed when executed. Use the database error and the
failed SQL to produce a corrected query that answers the same question.
Preserve the question's intent. Return only one corrected SELECT, or
NOT_ANSWERABLE if the question truly cannot be answered.
"""
)


def repair_sql(
    question: str,
    failed_sql: str,
    error: str,
    llm: LLMClient | None = None,
) -> SQLGenerationResult:
    """One execution-guided repair attempt for a failed generated query."""
    llm = llm or get_llm()
    system = REPAIR_SYSTEM_PROMPT.format(schema=SCHEMA_PROMPT)
    user = (
        f"Question: {question}\n\n"
        f"Failed SQL:\n{failed_sql}\n\n"
        f"Database error:\n{error}"
    )
    completion = llm.complete(system=system, user=user)
    return _parse_completion(question, completion)
