"""Unit tests for the pure-function per-table aggregator."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.features.aggregate import aggregate_table, apply_pit_filter
from src.features.table_config import TableSpec


@pytest.fixture
def base() -> pl.DataFrame:
    return pl.DataFrame({
        "case_id": [1, 2, 3],
        "date_decision": [date(2024, 6, 1), date(2024, 6, 15), date(2024, 7, 1)],
        "WEEK_NUM": [22, 24, 26],
        "target": [0, 1, 0],
    })


def test_pit_filter_drops_future_rows(base: pl.DataFrame) -> None:
    df = pl.DataFrame({
        "case_id": [1, 1, 2, 2, 3],
        "num_group1": [0, 1, 0, 1, 0],
        "creationdate_885D": [
            date(2024, 1, 1),    # keep — before decision
            date(2024, 7, 1),    # drop — after decision (2024-06-01)
            date(2024, 5, 1),    # keep
            date(2024, 6, 10),   # keep — before decision (2024-06-15)
            date(2024, 8, 1),    # drop — after decision (2024-07-01)
        ],
        "amount_A": [100, 200, 300, 400, 500],
    })
    out = apply_pit_filter(df, base, "creationdate_885D")
    assert out.height == 3
    assert sorted(out["case_id"].to_list()) == [1, 2, 2]


def test_pit_filter_keeps_nulls(base: pl.DataFrame) -> None:
    """Null dates are treated as unknown → kept. The alternative (drop) would
    throw away data conservatively; keeping preserves more signal."""
    df = pl.DataFrame({
        "case_id": [1, 1],
        "num_group1": [0, 1],
        "creationdate_885D": [None, date(2024, 1, 1)],
        "amount_A": [100, 200],
    }, schema={"case_id": pl.Int64, "num_group1": pl.Int64, "creationdate_885D": pl.Date, "amount_A": pl.Int64})
    out = apply_pit_filter(df, base, "creationdate_885D")
    assert out.height == 2


def test_aggregate_numeric_and_string(base: pl.DataFrame) -> None:
    spec = TableSpec(
        name="applprev_1",
        pattern="{split}_applprev_1_*.parquet",
        depth=1,
        pit_date_col=None,  # skip PIT for this test — focus on aggregation
    )
    df = pl.DataFrame({
        "case_id":    [1, 1, 1, 2, 2],
        "num_group1": [0, 1, 2, 0, 1],
        "amount_A":   [100.0, 200.0, 300.0, 50.0, 150.0],
        "status_M":   ["A", "B", "A", "A", "C"],
    })
    out = aggregate_table(df, spec, base).sort("case_id")

    assert out.height == 2
    assert out["applprev_1___row_count"].to_list() == [3, 2]
    assert out["applprev_1__amount_A_sum"].to_list() == [600.0, 200.0]
    assert out["applprev_1__amount_A_mean"].to_list() == [200.0, 100.0]
    assert out["applprev_1__amount_A_min"].to_list() == [100.0, 50.0]
    assert out["applprev_1__amount_A_max"].to_list() == [300.0, 150.0]
    assert out["applprev_1__status_M_nunique"].to_list() == [2, 2]
    assert out["applprev_1__status_M_notnull"].to_list() == [3, 2]


def test_aggregate_drops_num_group_meta(base: pl.DataFrame) -> None:
    """num_group1 / num_group2 should never leak into feature columns."""
    spec = TableSpec(name="t", pattern="x", depth=1, pit_date_col=None)
    df = pl.DataFrame({
        "case_id":    [1, 1],
        "num_group1": [0, 1],
        "num_group2": [0, 0],
        "amount_A":   [10.0, 20.0],
    })
    out = aggregate_table(df, spec, base)
    assert not any("num_group" in c for c in out.columns)
