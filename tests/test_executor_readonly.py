"""
Proves layer 2 of defense-in-depth independently of layer 1: even a
query that somehow bypassed validate_sql() cannot write, because the
connection itself is opened read-only.
"""

import pytest

from app.sql.executor import execute_validated_sql
from app.sql.safety import validate_sql


def test_select_executes_and_returns_rows():
    validation = validate_sql("SELECT * FROM accounts")
    assert validation.is_safe
    result = execute_validated_sql(validation.safe_sql)
    assert result.row_count > 0
    assert "account_name" in result.columns


def test_connection_itself_refuses_writes_even_if_validator_is_bypassed():
    # Deliberately skip validate_sql() to prove the SECOND layer holds
    # on its own — this simulates "what if the validator had a bug".
    with pytest.raises(PermissionError):
        execute_validated_sql("DELETE FROM accounts")


def test_row_limit_actually_caps_returned_rows():
    validation = validate_sql("SELECT * FROM transactions", row_limit=10)
    assert validation.is_safe
    result = execute_validated_sql(validation.safe_sql)
    assert result.row_count <= 10
