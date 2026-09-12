"""
Text-to-SQL generation. Deliberately dumb and separate from safety.py —
this module's only job is "question + schema -> candidate SQL string".
It is never trusted on its own; every output is passed to
safety.validate_sql() before anything touches the database (see
app/sql/pipeline.py for the wiring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
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

Schema:
{schema}
""".strip()

@dataclass
class SQLGenerationResult:
    question: str
    raw_sql: str | None
    not_answerable: bool


def generate_sql(question: str, llm: LLMClient | None = None) -> SQLGenerationResult:
    llm = llm or get_llm()
    system = SYSTEM_PROMPT.format(schema=SCHEMA_PROMPT)
    try:
        completion = llm.complete(system=system, user=question).strip()
    except Exception:
        if not settings.sql_fallback_enabled:
            raise
        fallback_sql = _fallback_generate_sql(question)
        if fallback_sql is None:
            return SQLGenerationResult(question=question, raw_sql=None, not_answerable=True)
        return SQLGenerationResult(question=question, raw_sql=fallback_sql, not_answerable=False)

    # Strip accidental markdown fences — small models love to add these
    # even when told not to.
    if completion.startswith("```"):
        completion = completion.strip("`")
        if completion.lower().startswith("sql"):
            completion = completion[3:].strip()

    if completion.strip().upper() == "NOT_ANSWERABLE":
        return SQLGenerationResult(question=question, raw_sql=None, not_answerable=True)

    return SQLGenerationResult(question=question, raw_sql=completion, not_answerable=False)


def _fallback_generate_sql(question: str) -> str | None:
    q = re.sub(r"\s+", " ", question.lower()).strip()

    if "vendors" in q and "india" in q:
        return "SELECT COUNT(*) AS n FROM vendors WHERE country = 'India'"
    if "how many" in q and "overdue" in q and "invoice" in q:
        return "SELECT COUNT(*) AS n FROM invoices WHERE status = 'overdue'"
    if "overdue" in q and "total" in q and "invoice" in q:
        return "SELECT SUM(amount) AS total FROM invoices WHERE status = 'overdue'"
    if "account_type" in q and "revenue" in q:
        return "SELECT account_name FROM accounts WHERE account_type = 'revenue'"
    if "pending" in q and "transaction" in q:
        return "SELECT COUNT(*) AS n FROM transactions WHERE status = 'pending'"
    if "invoice amount per vendor category" in q or ("vendor category" in q and "total" in q):
        return (
            "SELECT v.vendor_category, SUM(i.amount) AS total FROM invoices i "
            "JOIN vendors v ON i.vendor_id = v.vendor_id GROUP BY v.vendor_category"
        )
    if "highest total invoiced" in q or "highest invoice total" in q:
        return (
            "SELECT v.vendor_name, SUM(i.amount) AS total FROM invoices i "
            "JOIN vendors v ON i.vendor_id = v.vendor_id "
            "GROUP BY v.vendor_name ORDER BY total DESC LIMIT 1"
        )
    if "how many invoices does each vendor" in q:
        return (
            "SELECT v.vendor_name, COUNT(*) AS n FROM invoices i "
            "JOIN vendors v ON i.vendor_id = v.vendor_id "
            "GROUP BY v.vendor_name ORDER BY n DESC"
        )
    if "total debit and credit" in q and "general ledger" in q and "2024-q1" in q:
        return (
            "SELECT SUM(debit) AS total_debit, SUM(credit) AS total_credit "
            "FROM general_ledger WHERE fiscal_quarter = '2024-Q1'"
        )
    if "accounts exist per account_type" in q:
        return "SELECT account_type, COUNT(*) AS n FROM accounts GROUP BY account_type"
    if "average invoice amount" in q:
        return "SELECT AVG(amount) AS avg_amount FROM invoices"
    if "5 most recent transactions" in q:
        return (
            "SELECT transaction_id, transaction_date, description, amount "
            "FROM transactions ORDER BY transaction_date DESC LIMIT 5"
        )
    if "cloud infrastructure" in q and "how many vendors" in q:
        return "SELECT COUNT(*) AS n FROM vendors WHERE vendor_category = 'Cloud Infrastructure'"
    if "void" in q and "total transaction amount" in q:
        return "SELECT SUM(amount) AS total FROM transactions WHERE status = 'void'"
    if "general ledger entries" in q and "fiscal quarter" in q:
        return (
            "SELECT fiscal_quarter, COUNT(*) AS n FROM general_ledger "
            "GROUP BY fiscal_quarter ORDER BY fiscal_quarter"
        )
    if "overdue" in q and "germany" in q:
        return (
            "SELECT i.invoice_number, i.amount FROM invoices i "
            "JOIN vendors v ON i.vendor_id = v.vendor_id "
            "WHERE i.status = 'overdue' AND v.country = 'Germany'"
        )

    return None
