"""
Phase 1 acceptance: at least 5 malicious/off-schema questions, each
rejected BEFORE execution. Run: python -m eval.sql_safety_eval
"""

from __future__ import annotations

import json

from app.sql.safety import validate_sql

MALICIOUS_CASES = [
    ("drop_table", "DROP TABLE transactions"),
    ("delete_mutation", "DELETE FROM invoices WHERE invoice_id = 1"),
    ("update_mutation", "UPDATE accounts SET account_name = 'hacked'"),
    ("stacked_injection", "SELECT * FROM accounts; DROP TABLE accounts;"),
    ("off_schema_table", "SELECT * FROM users"),
    ("pragma_probe", "PRAGMA table_info(accounts)"),
    ("sqlite_master_probe", "SELECT * FROM sqlite_master"),
]

if __name__ == "__main__":
    results = []
    for name, sql in MALICIOUS_CASES:
        v = validate_sql(sql)
        results.append({"case": name, "sql": sql, "rejected": not v.is_safe, "reason": v.reason})
    passed = sum(1 for r in results if r["rejected"])
    print(json.dumps(results, indent=2))
    print(f"\n{passed}/{len(results)} malicious queries correctly rejected before execution")
