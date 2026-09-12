"""
Tests for app/sql/safety.py in isolation — no LLM, no live DB. This is
the file that proves the safety layer works regardless of what the
Text-to-SQL model does or doesn't generate correctly.
"""

from app.sql.safety import validate_sql

# ---------------------------------------------------------------------
# Queries that MUST be accepted
# ---------------------------------------------------------------------

GOOD_QUERIES = [
    "SELECT * FROM transactions WHERE amount > 1000",
    "SELECT account_name, account_type FROM accounts",
    "SELECT SUM(amount) FROM transactions WHERE status = 'posted'",
    "SELECT v.vendor_name, SUM(i.amount) as total FROM invoices i "
    "JOIN vendors v ON i.vendor_id = v.vendor_id GROUP BY v.vendor_name",
    "SELECT fiscal_quarter, SUM(credit) - SUM(debit) FROM general_ledger GROUP BY fiscal_quarter",
    "select * from vendors where country = 'India'",
    "SELECT COUNT(*) FROM invoices WHERE status = 'overdue'",
]


def test_valid_selects_are_accepted():
    for q in GOOD_QUERIES:
        result = validate_sql(q)
        assert result.is_safe, f"expected safe: {q} -> {result.reason} {result.violations}"
        assert result.safe_sql is not None
        assert "limit" in result.safe_sql.lower()


# ---------------------------------------------------------------------
# Queries that MUST be rejected (this is the "at least 5 malicious /
# off-schema test questions" the phase-1 acceptance criteria call for)
# ---------------------------------------------------------------------

MALICIOUS_QUERIES = {
    "drop_table": "DROP TABLE transactions",
    "update_mutation": "UPDATE accounts SET account_name = 'hacked' WHERE account_id = 1",
    "delete_mutation": "DELETE FROM invoices WHERE invoice_id = 1",
    "stacked_query_injection": "SELECT * FROM accounts; DROP TABLE accounts;",
    "insert_mutation": "INSERT INTO vendors (vendor_name) VALUES ('evil')",
    "off_schema_table": "SELECT * FROM users",
    "sqlite_master_probe": "SELECT * FROM sqlite_master",
    "pragma_probe": "PRAGMA table_info(accounts)",
    "attach_database": "ATTACH DATABASE '/etc/passwd' AS pwned",
    "off_schema_column": "SELECT ssn FROM accounts",
}


def test_malicious_and_off_schema_queries_are_rejected():
    for name, q in MALICIOUS_QUERIES.items():
        result = validate_sql(q)
        assert not result.is_safe, f"expected rejection for {name}: {q}"
        assert result.reason or result.violations


def test_multi_statement_is_rejected_before_execution_not_after():
    # The dangerous half must never even reach the parser as "valid":
    # this must fail at validate_sql(), not surface as a DB-level error
    # after something already executed.
    result = validate_sql("SELECT * FROM accounts; DELETE FROM accounts;")
    assert not result.is_safe


def test_row_limit_is_enforced_when_missing():
    result = validate_sql("SELECT * FROM transactions", row_limit=50)
    assert result.is_safe
    assert "limit 50" in result.safe_sql.lower()


def test_row_limit_is_clamped_when_llm_requests_too_many():
    result = validate_sql("SELECT * FROM transactions LIMIT 100000", row_limit=50)
    assert result.is_safe
    assert "limit 50" in result.safe_sql.lower()


def test_existing_small_limit_is_respected_not_overridden():
    result = validate_sql("SELECT * FROM transactions LIMIT 5", row_limit=500)
    assert result.is_safe
    assert "limit 5" in result.safe_sql.lower()


def test_unparseable_sql_is_rejected():
    result = validate_sql("SELEKT * FORM transactions WHERE")
    assert not result.is_safe


def test_non_select_statement_types_all_rejected():
    for q in [
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE accounts ADD COLUMN backdoor TEXT",
        "TRUNCATE TABLE transactions",
    ]:
        result = validate_sql(q)
        assert not result.is_safe, f"expected rejection: {q}"
