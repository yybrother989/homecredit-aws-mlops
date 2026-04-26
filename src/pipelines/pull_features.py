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
import subprocess
import sys
from pathlib import Path

# SageMaker Processing containers don't inherit AWS_DEFAULT_REGION from the
# job's run environment — boto3 calls need it set explicitly or the resolver
# raises NoRegionError. The job runs in us-west-2 by definition (we deploy
# only there), so hard-coding is fine for this project.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")

# The SageMaker sklearn 1.2-1 container ships pandas 1.1.3 / numpy 1.24.1 /
# scipy 1.8.0 — too old for awswrangler. requirements.txt-based installs by
# FrameworkProcessor leave a mixed binary state where pyarrow/pandas were
# upgraded but their compiled extensions still reference the original numpy,
# causing `numpy.core.multiarray failed to import`. The bulletproof fix is
# `--force-reinstall` of the whole data-stack as a single consistent group
# BEFORE the first `import` of any of them.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "--upgrade", "--force-reinstall",
    "numpy==1.26.4",
    "pandas==2.2.3",
    "pyarrow==15.0.2",
    "awswrangler==3.6.0",
])

import awswrangler as wr  # noqa: E402

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
