"""SageMaker Processing step 1: pull features from Athena, time-split, write parquet.

Runs inside a SageMaker Processing Job container (sklearn base image + the
requirements file at src/training/lightgbm_requirements.txt).

Inputs (via container args):
    --athena-database     default: homecredit_ml
    --athena-table        default: features
    --athena-staging-uri  s3://artifacts/athena/ (temp CTAS output)
    --output-dir          /opt/ml/processing/output  (mapped to two S3 paths)
    --val-weeks-frac      0.2

Outputs (two ProcessingOutput channels):
    /opt/ml/processing/output/train/train.parquet
    /opt/ml/processing/output/validation/validation.parquet

Why CTAS: the default awswrangler path streams CSV — we hit IncompleteRead
on 5 GB. CTAS materializes parquet to S3 staging, then pyarrow reads it;
~10× faster, survives network hiccups. (Proven in Phase 2.)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import awswrangler as wr

# Custom container ships ABI-consistent numpy/pandas/pyarrow/awswrangler;
# AWS_DEFAULT_REGION baked in via Dockerfile but defensive setdefault here
# matches local dev where it might be unset.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--athena-database", default="homecredit_ml")
    p.add_argument("--athena-table", default="features")
    p.add_argument("--athena-staging-uri", required=True)
    p.add_argument("--output-dir", default="/opt/ml/processing/output")
    p.add_argument("--val-weeks-frac", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)

    log.info("Pulling %s.%s via Athena CTAS (staging=%s)",
             args.athena_database, args.athena_table, args.athena_staging_uri)
    df = wr.athena.read_sql_query(
        sql=f"SELECT * FROM {args.athena_table}",
        database=args.athena_database,
        s3_output=args.athena_staging_uri,
        ctas_approach=True,
    )
    log.info("  pulled shape=%s", df.shape)

    # Drop FS bookkeeping / Athena-lowercase quirks — match training contract.
    drop = [c for c in df.columns if c in {"api_invocation_time", "write_time", "is_deleted", "event_time"}]
    if drop:
        df = df.drop(columns=drop)
    # Athena lowercases identifiers; training expects `WEEK_NUM` uppercase.
    df = df.rename(columns={c: "WEEK_NUM" for c in df.columns if c.lower() == "week_num"})

    weeks = sorted(df["WEEK_NUM"].unique().tolist())
    cutoff = int(len(weeks) * (1 - args.val_weeks_frac))
    val_weeks = set(weeks[cutoff:])
    log.info("Time split: %d train weeks, %d val weeks (val=%d..%d)",
             cutoff, len(weeks) - cutoff, min(val_weeks), max(val_weeks))

    train_df = df[~df["WEEK_NUM"].isin(val_weeks)]
    val_df = df[df["WEEK_NUM"].isin(val_weeks)]

    train_df.to_parquet(out / "train" / "train.parquet", index=False)
    val_df.to_parquet(out / "validation" / "validation.parquet", index=False)
    log.info("Wrote train=%s, val=%s", train_df.shape, val_df.shape)


if __name__ == "__main__":
    main()
