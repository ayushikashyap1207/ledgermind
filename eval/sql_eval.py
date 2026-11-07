"""
Phase 1 acceptance test: 15 fixed questions against the schema.

Correctness is measured by comparing EXECUTION RESULTS against a golden
SQL query per question, not by string-matching generated SQL — two
different SELECTs can be equally correct. This also means the golden
queries themselves are a real, standalone regression test for the
schema/executor even with no LLM in the loop (see `--golden-only`).

Run:
    python -m eval.sql_eval                 # full: generate_sql via LLM_PROVIDER, then compare
    python -m eval.sql_eval --golden-only    # sanity-check the golden queries execute cleanly
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

from app.sql.executor import execute_validated_sql
from app.sql.pipeline import answer_from_sql
from app.sql.safety import validate_sql


@dataclass
class GoldenQuestion:
    id: str
    question: str
    golden_sql: str


GOLDEN_QUESTIONS: list[GoldenQuestion] = [
    GoldenQuestion("q01", "How many vendors are based in India?",
                    "SELECT COUNT(*) AS n FROM vendors WHERE country = 'India'"),
    GoldenQuestion("q02", "What is the total amount of all overdue invoices?",
                    "SELECT SUM(amount) AS total FROM invoices WHERE status = 'overdue'"),
    GoldenQuestion("q03", "List all account names with account_type 'revenue'.",
                    "SELECT account_name FROM accounts WHERE account_type = 'revenue'"),
    GoldenQuestion("q04", "How many transactions are currently pending?",
                    "SELECT COUNT(*) AS n FROM transactions WHERE status = 'pending'"),
    GoldenQuestion("q05", "What is the total invoice amount per vendor category?",
                    "SELECT v.vendor_category, SUM(i.amount) AS total FROM invoices i "
                    "JOIN vendors v ON i.vendor_id = v.vendor_id GROUP BY v.vendor_category"),
    GoldenQuestion("q06", "Which vendor has the highest total invoiced amount?",
                    "SELECT v.vendor_name, SUM(i.amount) AS total FROM invoices i "
                    "JOIN vendors v ON i.vendor_id = v.vendor_id "
                    "GROUP BY v.vendor_name ORDER BY total DESC LIMIT 1"),
    GoldenQuestion("q07", "How many invoices does each vendor have, ordered by count descending?",
                    "SELECT v.vendor_name, COUNT(*) AS n FROM invoices i "
                    "JOIN vendors v ON i.vendor_id = v.vendor_id "
                    "GROUP BY v.vendor_name ORDER BY n DESC"),
    GoldenQuestion("q08", "What is the total debit and credit recorded in the general ledger for fiscal quarter 2024-Q1?",
                    "SELECT SUM(debit) AS total_debit, SUM(credit) AS total_credit "
                    "FROM general_ledger WHERE fiscal_quarter = '2024-Q1'"),
    GoldenQuestion("q09", "How many accounts exist per account_type?",
                    "SELECT account_type, COUNT(*) AS n FROM accounts GROUP BY account_type"),
    GoldenQuestion("q10", "What is the average invoice amount across all invoices?",
                    "SELECT AVG(amount) AS avg_amount FROM invoices"),
    GoldenQuestion("q11", "List the 5 most recent transactions by date.",
                    "SELECT transaction_id, transaction_date, description, amount "
                    "FROM transactions ORDER BY transaction_date DESC LIMIT 5"),
    GoldenQuestion("q12", "How many vendors are in the 'Cloud Infrastructure' category?",
                    "SELECT COUNT(*) AS n FROM vendors WHERE vendor_category = 'Cloud Infrastructure'"),
    GoldenQuestion("q13", "What is the total transaction amount for transactions with status 'void'?",
                    "SELECT SUM(amount) AS total FROM transactions WHERE status = 'void'"),
    GoldenQuestion("q14", "How many general ledger entries were recorded per fiscal quarter?",
                    "SELECT fiscal_quarter, COUNT(*) AS n FROM general_ledger "
                    "GROUP BY fiscal_quarter ORDER BY fiscal_quarter"),
    GoldenQuestion("q15", "Which invoices are overdue and issued by a vendor in 'Germany'?",
                    "SELECT i.invoice_number, i.amount FROM invoices i "
                    "JOIN vendors v ON i.vendor_id = v.vendor_id "
                    "WHERE i.status = 'overdue' AND v.country = 'Germany'"),
]


def _normalize_rows(rows: list[dict], columns: list[str] | None = None) -> list[tuple]:
    """Order-independent, type-loose comparison: sort rows by their
    string representation so column-order/row-order differences between
    an LLM's SELECT and the golden SELECT don't cause false negatives."""
    if columns is not None:
        normed = [tuple(str(row.get(column)) for column in columns) for row in rows]
    else:
        normed = [tuple(str(v) for v in row.values()) for row in rows]
    return sorted(normed)


def _project(rows: list[dict], columns: list[str]) -> list[dict]:
    return [{column: row.get(column) for column in columns} for row in rows]


def results_match(
    generated_columns: list[str],
    generated_rows: list[dict],
    golden_columns: list[str],
    golden_rows: list[dict],
) -> bool:
    """Compare result sets without punishing a legitimate projection.

    Two different SELECTs can answer the same question. Accept:
    1. an exact row match (regardless of column aliases);
    2. a generated superset of the golden columns (project it down);
    3. a generated subset of the golden columns (project the golden down),
       e.g. "which vendor" returning the vendor name without the total.
    Value mismatches and missing/extra rows still fail.
    """
    if _normalize_rows(generated_rows) == _normalize_rows(golden_rows):
        return True

    if set(golden_columns).issubset(set(generated_columns)):
        projected = _project(generated_rows, golden_columns)
        if _normalize_rows(projected, golden_columns) == _normalize_rows(
            golden_rows, golden_columns
        ):
            return True

    if generated_columns and set(generated_columns).issubset(set(golden_columns)):
        projected_golden = _project(golden_rows, generated_columns)
        if _normalize_rows(generated_rows, generated_columns) == _normalize_rows(
            projected_golden, generated_columns
        ):
            return True

    return False


def run_golden_only() -> dict:
    """Executes each golden query directly — proves the eval harness
    and schema are sound, independent of any LLM."""
    results = []
    for gq in GOLDEN_QUESTIONS:
        validation = validate_sql(gq.golden_sql)
        ok = validation.is_safe
        row_count = None
        error = None
        if ok:
            try:
                res = execute_validated_sql(validation.safe_sql)
                row_count = res.row_count
            except Exception as exc:  # noqa: BLE001
                ok = False
                error = str(exc)
        results.append({"id": gq.id, "question": gq.question, "passed_safety": validation.is_safe,
                         "row_count": row_count, "error": error})
    passed = sum(1 for r in results if r["row_count"] is not None)
    return {"mode": "golden_only", "passed": passed, "total": len(GOLDEN_QUESTIONS), "results": results}


def run_full_eval() -> dict:
    """Real acceptance test: generate SQL via the configured LLM
    provider, validate + execute it, and compare the result set against
    the golden query's result set."""
    results = []
    for gq in GOLDEN_QUESTIONS:
        golden_validation = validate_sql(gq.golden_sql)
        golden_result = (
            execute_validated_sql(golden_validation.safe_sql)
            if golden_validation.is_safe
            else None
        )
        golden_rows = golden_result.rows if golden_result else []
        golden_columns = golden_result.columns if golden_result else []

        try:
            answer = answer_from_sql(gq.question)
        except Exception as exc:  # noqa: BLE001
            results.append({
                "id": gq.id,
                "question": gq.question,
                "status": "execution_error",
                "generated_sql": None,
                "correct": False,
                "error": str(exc),
            })
            continue

        correct = False
        if answer.status == "ok" and answer.result is not None:
            correct = results_match(
                generated_columns=answer.result.columns,
                generated_rows=answer.result.rows,
                golden_columns=golden_columns,
                golden_rows=golden_rows,
            )

        results.append({
            "id": gq.id, "question": gq.question, "status": answer.status,
            "generated_sql": answer.generated_sql, "correct": correct,
            "error": answer.error,
            "repair_attempted": answer.repair_attempted,
            "repair_succeeded": answer.repair_succeeded,
        })

    passed = sum(1 for r in results if r["correct"])
    return {"mode": "full", "passed": passed, "total": len(GOLDEN_QUESTIONS), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-only", action="store_true")
    args = parser.parse_args()

    report = run_golden_only() if args.golden_only else run_full_eval()
    print(json.dumps(report, indent=2, default=str))
    print(f"\n{report['passed']}/{report['total']} passed ({report['mode']} mode)", file=sys.stderr)
