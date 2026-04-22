"""Pure-function per-table aggregation logic.

Implemented in Polars so we can unit-test on tiny dataframes without spinning up
Spark. The PySpark version in `glue_feature_job.py` uses equivalent operations
(group_by + .agg) — keeping the logic in one place would require an abstraction
(Pandas-on-Spark / Ibis / DuckDB) that's more overhead than value for Phase 2.

Contract shared by both implementations:
    INPUT:  wide depth>=1 dataframe + base(case_id, date_decision)
    OUTPUT: one row per case_id, columns renamed `{table}__{col}_{agg}`
"""
from __future__ import annotations

import polars as pl

from src.features.table_config import (
    CASE_ID_COL,
    DATE_DECISION_COL,
    TableSpec,
)

NUMERIC_AGGS = ("count", "mean", "sum", "min", "max")


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_string(dtype: pl.DataType) -> bool:
    return dtype in (pl.Utf8, pl.Categorical)


def apply_pit_filter(
    df: pl.DataFrame,
    base: pl.DataFrame,
    date_col: str,
) -> pl.DataFrame:
    """Drop rows where `date_col` is strictly after base.date_decision."""
    if date_col not in df.columns:
        raise KeyError(f"{date_col} not in columns; cannot PIT-filter")
    base_dates = base.select([CASE_ID_COL, DATE_DECISION_COL])
    return (
        df.join(base_dates, on=CASE_ID_COL, how="inner")
          .filter(
              pl.col(date_col).is_null()
              | (pl.col(date_col) <= pl.col(DATE_DECISION_COL))
          )
          .drop(DATE_DECISION_COL)
    )


def aggregate_table(
    df: pl.DataFrame,
    spec: TableSpec,
    base: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate a depth>=1 table to case_id grain.

    - PIT-filter if `spec.pit_date_col` is set.
    - Numeric cols → count / mean / sum / min / max.
    - String/categorical cols → n_unique / non_null_count.
    - `num_group1` / `num_group2` are metadata, dropped before aggregation.
    """
    if spec.depth == 0:
        raise ValueError(f"aggregate_table called on depth-0 spec {spec.name}")

    if spec.pit_date_col:
        df = apply_pit_filter(df, base, spec.pit_date_col)

    drop = {CASE_ID_COL, "num_group1", "num_group2"}
    feature_cols = [c for c in df.columns if c not in drop]

    exprs: list[pl.Expr] = [pl.len().alias(f"{spec.name}___row_count")]
    for col in feature_cols:
        dtype = df.schema[col]
        base_expr = pl.col(col)
        prefix = f"{spec.name}__{col}"
        if _is_numeric(dtype):
            exprs.extend([
                base_expr.mean().alias(f"{prefix}_mean"),
                base_expr.sum().alias(f"{prefix}_sum"),
                base_expr.min().alias(f"{prefix}_min"),
                base_expr.max().alias(f"{prefix}_max"),
            ])
        elif _is_string(dtype):
            exprs.extend([
                base_expr.n_unique().alias(f"{prefix}_nunique"),
                base_expr.is_not_null().sum().alias(f"{prefix}_notnull"),
            ])
        # silently skip date/binary/list types for now — Phase 2 baseline

    return df.group_by(CASE_ID_COL).agg(exprs)
