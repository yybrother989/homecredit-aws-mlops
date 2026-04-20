"""Feature engineering for the Home Credit baseline.

Layout produced by Kaggle:
    parquet_files/train/
        train_base.parquet              # 1 row per case_id, has target + WEEK_NUM
        train_static_0_{0,1}.parquet    # 168 cols, depth-0 (1 row/case), sharded
        train_static_cb_0.parquet       # 53 cols, depth-0 (1 row/case)
        train_applprev_1_{0,1}.parquet  # depth-1, multi-row per case_id (needs agg)
        train_applprev_2.parquet        # depth-2, same
        train_credit_bureau_a_{1,2}_*   # depth-1 / depth-2 shards
        train_person_{1,2}.parquet
        ...

Baseline strategy:
    - Keep `train_base` as the spine (case_id, target, WEEK_NUM, date_decision).
    - Concat + left-join all **depth-0** tables (`train_static_0_*`, `train_static_cb_0`).
    - Skip depth-1/2 tables for now (Phase 2 will aggregate them in a Glue job).
    - Split by WEEK_NUM — last 20% of weeks → validation — preserves time structure
      for the Kaggle stability metric.

Run:
    uv run python src/features/build_features.py \
        --input-dir  data/raw/parquet_files/train \
        --output-dir data/processed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEPTH0_GROUPS = {
    # logical name      -> glob pattern relative to input dir
    "static":    "train_static_0_*.parquet",
    "static_cb": "train_static_cb_0.parquet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-weeks-frac", type=float, default=0.2,
                   help="Last N fraction of weeks used for validation split")
    return p.parse_args()


def load_depth0_group(input_dir: Path, pattern: str) -> pl.DataFrame:
    """Concat all shards matching a pattern; ensures one row per case_id."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {input_dir}")
    log.info("  %s: %d shard(s), %s", pattern, len(files), [f.name for f in files])
    df = pl.concat([pl.read_parquet(f) for f in files])
    if df.select(pl.col("case_id").n_unique()).item() != df.height:
        raise ValueError(f"{pattern}: not unique per case_id (depth>0?)")
    return df


def cast_for_lightgbm(df: pl.DataFrame) -> pl.DataFrame:
    """LightGBM wants numeric or category; convert object/string to Categorical."""
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
        elif dtype == pl.Date or dtype == pl.Datetime:
            # Convert dates to days-since-epoch (int) so LightGBM can split on them
            df = df.with_columns(pl.col(col).cast(pl.Int32).alias(col))
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train"
    val_out = args.output_dir / "validation"
    train_out.mkdir(exist_ok=True)
    val_out.mkdir(exist_ok=True)

    # 1. Spine
    base_path = args.input_dir / "train_base.parquet"
    log.info("Loading spine: %s", base_path)
    base = pl.read_parquet(base_path)
    log.info("  base shape=%s, weeks=%d", base.shape, base["WEEK_NUM"].n_unique())

    # 2. Left-join each depth-0 group
    for group_name, pattern in DEPTH0_GROUPS.items():
        log.info("Joining %s", group_name)
        dfg = load_depth0_group(args.input_dir, pattern)
        # Avoid column-name collisions across groups
        rename_map = {c: f"{group_name}__{c}" for c in dfg.columns if c != "case_id"}
        dfg = dfg.rename(rename_map)
        base = base.join(dfg, on="case_id", how="left")
        log.info("  → shape=%s", base.shape)

    # 3. Cast object/date cols to LightGBM-friendly types
    log.info("Casting columns for LightGBM")
    base = cast_for_lightgbm(base)

    # 4. Time-based split
    weeks = sorted(base["WEEK_NUM"].unique().to_list())
    cutoff_idx = int(len(weeks) * (1 - args.val_weeks_frac))
    val_weeks = set(weeks[cutoff_idx:])
    log.info("Split: %d train weeks, %d val weeks (val = week %d..%d)",
             cutoff_idx, len(weeks) - cutoff_idx, min(val_weeks), max(val_weeks))

    train_df = base.filter(~pl.col("WEEK_NUM").is_in(val_weeks))
    val_df = base.filter(pl.col("WEEK_NUM").is_in(val_weeks))
    log.info("  train=%s, val=%s", train_df.shape, val_df.shape)

    log.info("Writing parquet")
    train_df.write_parquet(train_out / "train.parquet")
    val_df.write_parquet(val_out / "validation.parquet")
    log.info("Done.")


if __name__ == "__main__":
    main()
