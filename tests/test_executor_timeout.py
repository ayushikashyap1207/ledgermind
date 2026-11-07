import time

import pytest

from app.sql.executor import QueryResultTooLarge, QueryTimeout, execute_validated_sql
from app.sql.safety import validate_sql


def test_sqlite_query_timeout_is_enforced():
    query = (
        "SELECT SUM(a.amount) FROM transactions a "
        "CROSS JOIN transactions b "
        "CROSS JOIN transactions c "
        "CROSS JOIN transactions d"
    )
    validation = validate_sql(query)
    assert validation.is_safe, validation.violations

    started = time.monotonic()
    with pytest.raises(QueryTimeout):
        execute_validated_sql(validation.safe_sql, timeout_seconds=0.05)

    assert time.monotonic() - started < 5.0


def test_result_row_cap_is_enforced():
    query = (
        "SELECT a.transaction_id FROM transactions a "
        "CROSS JOIN transactions b"
    )

    with pytest.raises(QueryResultTooLarge):
        execute_validated_sql(query)
