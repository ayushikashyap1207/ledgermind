from eval.sql_eval import results_match


def test_exact_result_match():
    assert results_match(
        generated_columns=["name", "amount"],
        generated_rows=[{"name": "A", "amount": 10}, {"name": "B", "amount": 20}],
        golden_columns=["name", "amount"],
        golden_rows=[{"name": "B", "amount": 20}, {"name": "A", "amount": 10}],
    )


def test_generated_superset_of_columns_is_projected():
    assert results_match(
        generated_columns=["id", "name", "amount"],
        generated_rows=[{"id": 1, "name": "A", "amount": 10}],
        golden_columns=["name", "amount"],
        golden_rows=[{"name": "A", "amount": 10}],
    )


def test_generated_subset_of_columns_is_projected():
    assert results_match(
        generated_columns=["vendor_name"],
        generated_rows=[{"vendor_name": "Acme"}],
        golden_columns=["vendor_name", "total"],
        golden_rows=[{"vendor_name": "Acme", "total": 999}],
    )


def test_different_single_column_alias_matches_by_value():
    assert results_match(
        generated_columns=["SUM(amount)"],
        generated_rows=[{"SUM(amount)": 100}],
        golden_columns=["total"],
        golden_rows=[{"total": 100}],
    )


def test_wrong_value_still_fails():
    assert not results_match(
        generated_columns=["total"],
        generated_rows=[{"total": 99}],
        golden_columns=["total"],
        golden_rows=[{"total": 100}],
    )


def test_missing_rows_still_fails():
    assert not results_match(
        generated_columns=["vendor_name"],
        generated_rows=[{"vendor_name": "Acme"}],
        golden_columns=["vendor_name", "total"],
        golden_rows=[
            {"vendor_name": "Acme", "total": 10},
            {"vendor_name": "Beta", "total": 20},
        ],
    )
