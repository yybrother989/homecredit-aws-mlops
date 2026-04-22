"""AWS Glue 5.0 PySpark job: feature engineering for Home Credit.

Contract (matches src/features/aggregate.py):
  INPUT  — s3://RAW_BUCKET/parquet_files/{SPLIT}/*.parquet    (Kaggle tree)
  OUTPUT — s3://PROCESSED_BUCKET/features/{SPLIT}/part-*.parquet
            one row per case_id, partitioned by WEEK_NUM

Steps:
  1. Read spine (`{split}_base.parquet`).
  2. Concat shards + left-join depth-0 tables (already 1 row/case).
  3. For each depth>=1 table: concat shards, optional PIT filter, aggregate to
     case_id grain, left-join onto the spine.
  4. Write wide output partitioned by WEEK_NUM.

Glue job parameters (set by the CDK stack, overridable at StartJobRun):
  --RAW_BUCKET        homecredit-raw-<acct>-usw2
  --PROCESSED_BUCKET  homecredit-processed-<acct>-usw2
  --SPLIT             train | test     (default: train)

Note on the table config duplication: this file redefines TABLE_CONFIG rather
than importing from src.features.table_config because Glue runs a flat
package. If this grows, upload the source as `--extra-py-files` via the CDK
and switch to a proper import. For Phase 2 the duplication is acceptable.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Table config (mirror of src/features/table_config.py — keep in sync manually)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSpec:
    name: str
    pattern: str
    depth: int
    pit_date_col: str | None


TABLE_CONFIG: tuple[TableSpec, ...] = (
    TableSpec("static",            "{split}_static_0_*.parquet",        0, None),
    TableSpec("static_cb",         "{split}_static_cb_0.parquet",       0, None),
    TableSpec("applprev_1",        "{split}_applprev_1_*.parquet",      1, "creationdate_885D"),
    TableSpec("credit_bureau_a_1", "{split}_credit_bureau_a_1_*.parquet", 1, "refreshdate_3813885D"),
    TableSpec("credit_bureau_b_1", "{split}_credit_bureau_b_1.parquet", 1, "contractdate_551D"),
    TableSpec("debitcard_1",       "{split}_debitcard_1.parquet",       1, "openingdate_857D"),
    TableSpec("deposit_1",         "{split}_deposit_1.parquet",         1, "openingdate_313D"),
    TableSpec("other_1",           "{split}_other_1.parquet",           1, None),
    TableSpec("person_1",          "{split}_person_1.parquet",          1, "empl_employedfrom_271D"),
    TableSpec("tax_registry_a_1",  "{split}_tax_registry_a_1.parquet",  1, "recorddate_4527225D"),
    TableSpec("tax_registry_b_1",  "{split}_tax_registry_b_1.parquet",  1, "deductiondate_4917603D"),
    TableSpec("tax_registry_c_1",  "{split}_tax_registry_c_1.parquet",  1, "processingdate_168D"),
    TableSpec("applprev_2",        "{split}_applprev_2.parquet",        2, None),
    TableSpec("credit_bureau_a_2", "{split}_credit_bureau_a_2_*.parquet", 2, None),
    TableSpec("credit_bureau_b_2", "{split}_credit_bureau_b_2.parquet", 2, "pmts_date_1107D"),
    TableSpec("person_2",          "{split}_person_2.parquet",          2, "empls_employedfrom_796D"),
)


NUMERIC_TYPES = ("int", "bigint", "long", "float", "double", "decimal", "smallint", "tinyint")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_glob(spark, bucket: str, split: str, pattern: str) -> DataFrame:
    """Read all shards matching a pattern as a single DataFrame."""
    path = f"s3://{bucket}/parquet_files/{split}/{pattern.format(split=split)}"
    return spark.read.parquet(path)


def aggregate_depth_gt_0(df: DataFrame, spec: TableSpec, base_dates: DataFrame) -> DataFrame:
    """Group a depth>=1 table by case_id after optional PIT filter."""
    if spec.pit_date_col:
        date_col = spec.pit_date_col
        if date_col not in df.columns:
            raise KeyError(f"{spec.name}: pit_date_col {date_col} missing")
        df = (
            df.join(F.broadcast(base_dates), on="case_id", how="inner")
              .filter(F.col(date_col).isNull() | (F.col(date_col) <= F.col("date_decision")))
              .drop("date_decision")
        )

    drop_cols = {"case_id", "num_group1", "num_group2"}
    aggs = [F.count(F.lit(1)).alias(f"{spec.name}___row_count")]

    for col_name, type_str in df.dtypes:
        if col_name in drop_cols:
            continue
        prefix = f"{spec.name}__{col_name}"
        if any(type_str.startswith(t) for t in NUMERIC_TYPES):
            aggs += [
                F.mean(col_name).alias(f"{prefix}_mean"),
                F.sum(col_name).alias(f"{prefix}_sum"),
                F.min(col_name).alias(f"{prefix}_min"),
                F.max(col_name).alias(f"{prefix}_max"),
            ]
        elif type_str == "string":
            aggs += [
                F.countDistinct(col_name).alias(f"{prefix}_nunique"),
                F.count(col_name).alias(f"{prefix}_notnull"),
            ]
        # skip date / binary / list types for Phase 2

    return df.groupBy("case_id").agg(*aggs)


def rename_depth0(df: DataFrame, spec: TableSpec) -> DataFrame:
    """Prefix depth-0 columns with table name to avoid collisions on join."""
    return df.select(
        [F.col(c).alias(c if c == "case_id" else f"{spec.name}__{c}") for c in df.columns]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "RAW_BUCKET", "PROCESSED_BUCKET", "SPLIT"])
    sc = SparkContext.getOrCreate()
    glue = GlueContext(sc)
    spark = glue.spark_session
    job = Job(glue)
    job.init(args["JOB_NAME"], args)

    raw = args["RAW_BUCKET"]
    processed = args["PROCESSED_BUCKET"]
    split = args["SPLIT"]

    print(f"[config] raw=s3://{raw}  processed=s3://{processed}  split={split}")

    # 1. Spine
    base = spark.read.parquet(f"s3://{raw}/parquet_files/{split}/{split}_base.parquet")
    base = base.repartition(200, "case_id").cache()
    base_dates = base.select("case_id", "date_decision")
    print(f"[spine] rows={base.count()} cols={len(base.columns)}")

    # 2. Join depth-0 tables directly (already one row per case_id)
    wide = base
    for spec in [s for s in TABLE_CONFIG if s.depth == 0]:
        print(f"[depth-0] {spec.name}")
        df = read_glob(spark, raw, split, spec.pattern)
        df = rename_depth0(df, spec)
        wide = wide.join(df, on="case_id", how="left")

    # 3. Aggregate + left-join depth>=1 tables
    for spec in [s for s in TABLE_CONFIG if s.depth >= 1]:
        print(f"[depth-{spec.depth}] {spec.name}  pit={spec.pit_date_col}")
        df = read_glob(spark, raw, split, spec.pattern)
        agg = aggregate_depth_gt_0(df, spec, base_dates)
        wide = wide.join(agg, on="case_id", how="left")

    # 4. Write partitioned parquet
    out_path = f"s3://{processed}/features/{split}/"
    print(f"[write] → {out_path}")
    (wide
        .repartition("WEEK_NUM")
        .write
        .mode("overwrite")
        .partitionBy("WEEK_NUM")
        .parquet(out_path))

    print(f"[done] wrote wide table, total cols={len(wide.columns)}")
    job.commit()


if __name__ == "__main__":
    main()
