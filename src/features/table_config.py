"""Single source of truth for the Home Credit table layout.

Every script (local Polars, Glue PySpark, tests, ingestion) reads from this
module so the feature engineering contract stays consistent across runtimes.

Table depth semantics (Kaggle convention):
    depth 0 — already one row per case_id, join directly onto the spine.
    depth 1 — many rows per case_id linked by `num_group1`.
              Aggregate to case_id grain before joining.
    depth 2 — many rows per (case_id, num_group1), linked additionally by
              `num_group2`. We simplify and aggregate directly to case_id
              (ignoring the intermediate grain); grandmaster solutions do the
              two-level aggregation but the marginal gain is small.

`pit_date_col` — which `_D` column, if any, represents the event time for
PIT filtering. Rows whose date is AFTER base.date_decision are dropped before
aggregation so we don't leak future information into training features.
Some tables have no date column at all (`other_1`, `applprev_2`,
`credit_bureau_a_2`); we aggregate everything and accept mild leakage for
Phase 2. Phase 2+ can revisit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    name: str
    pattern: str          # glob pattern under parquet_files/<split>/
    depth: int            # 0, 1, or 2
    pit_date_col: str | None


# Order matters only for log readability. Listed by depth then alphabetical.
TABLE_CONFIG: tuple[TableSpec, ...] = (
    # Depth 0 — no aggregation, left-join directly onto base
    TableSpec("static",           "{split}_static_0_*.parquet",        0, None),
    TableSpec("static_cb",        "{split}_static_cb_0.parquet",       0, None),

    # Depth 1 — aggregate to case_id
    TableSpec("applprev_1",       "{split}_applprev_1_*.parquet",      1, "creationdate_885D"),
    TableSpec("credit_bureau_a_1","{split}_credit_bureau_a_1_*.parquet", 1, "refreshdate_3813885D"),
    TableSpec("credit_bureau_b_1","{split}_credit_bureau_b_1.parquet", 1, "contractdate_551D"),
    TableSpec("debitcard_1",      "{split}_debitcard_1.parquet",       1, "openingdate_857D"),
    TableSpec("deposit_1",        "{split}_deposit_1.parquet",         1, "openingdate_313D"),
    TableSpec("other_1",          "{split}_other_1.parquet",           1, None),
    TableSpec("person_1",         "{split}_person_1.parquet",          1, "empl_employedfrom_271D"),
    TableSpec("tax_registry_a_1", "{split}_tax_registry_a_1.parquet",  1, "recorddate_4527225D"),
    TableSpec("tax_registry_b_1", "{split}_tax_registry_b_1.parquet",  1, "deductiondate_4917603D"),
    TableSpec("tax_registry_c_1", "{split}_tax_registry_c_1.parquet",  1, "processingdate_168D"),

    # Depth 2 — aggregate directly to case_id (simplified from two-level agg)
    TableSpec("applprev_2",       "{split}_applprev_2.parquet",        2, None),
    TableSpec("credit_bureau_a_2","{split}_credit_bureau_a_2_*.parquet", 2, None),
    TableSpec("credit_bureau_b_2","{split}_credit_bureau_b_2.parquet", 2, "pmts_date_1107D"),
    TableSpec("person_2",         "{split}_person_2.parquet",          2, "empls_employedfrom_796D"),
)

# Sanity constants — keep in sync with the spine.
SPINE_FILE = "{split}_base.parquet"
CASE_ID_COL = "case_id"
DATE_DECISION_COL = "date_decision"
WEEK_COL = "WEEK_NUM"
TARGET_COL = "target"


def pattern_for(spec: TableSpec, split: str = "train") -> str:
    """Resolve the `{split}` placeholder in a table's glob pattern."""
    return spec.pattern.format(split=split)


def depth0_specs() -> tuple[TableSpec, ...]:
    return tuple(s for s in TABLE_CONFIG if s.depth == 0)


def aggregated_specs() -> tuple[TableSpec, ...]:
    return tuple(s for s in TABLE_CONFIG if s.depth >= 1)
