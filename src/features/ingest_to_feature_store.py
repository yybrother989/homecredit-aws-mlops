"""Register the Glue wide-feature parquet as an Athena external table.

Why not `FeatureGroup.ingest()`?
  It pushes records through the PutRecord API, which is ~tens of records per
  second per worker. 1.5M rows × 500 cols timed out in testing (22 GB RAM,
  still running after 15 min). That path is designed for streaming writes,
  not bulk materialization.

What we do instead:
  The SageMaker Feature Group `homecredit-features` is already provisioned
  (by CDK) with its schema contract + Athena table — but its offline store
  is empty. For Phase 2 (training only), we just need the wide feature
  table to be queryable. So we register the Glue output directly as an
  **external Athena table** in a dedicated `homecredit_ml` database,
  bypassing Feature Store's write path.

  The Feature Group remains in place as the canonical schema definition;
  Phase 4 will populate it properly (via SparkDataFrameIngestion from a
  SageMaker Processing job, or Feature Store's batch ingestion API) when
  real-time inference comes online.

Output:
  Athena table `homecredit_ml.features` pointing at
  `s3://homecredit-processed-<acct>-usw2/features/train/`
  partitioned by `WEEK_NUM` (auto-discovered via MSCK REPAIR).

Usage:
    source env.sh
    uv run python src/features/ingest_to_feature_store.py
"""
from __future__ import annotations

import argparse
import logging

import awswrangler as wr
import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ATHENA_DB = "homecredit_ml"
ATHENA_TABLE = "features"


def cfn_output(stack: str, key: str) -> str:
    cf = boto3.client("cloudformation")
    outs = cf.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    for o in outs:
        if o["OutputKey"] == key:
            return o["OutputValue"]
    raise KeyError(f"{key} not in {stack} outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    processed = cfn_output("HomeCreditBaseStack", "ProcessedBucketName")
    artifacts = cfn_output("HomeCreditBaseStack", "ArtifactsBucketName")
    s3_prefix = f"s3://{processed}/features/{args.split}/"
    athena_staging = f"s3://{artifacts}/athena/"

    log.info("Creating Athena DB '%s' if missing…", ATHENA_DB)
    wr.catalog.create_database(name=ATHENA_DB, exist_ok=True)

    log.info("Inspecting parquet schema under %s", s3_prefix)
    # `store_parquet_metadata` infers the schema from parquet files and
    # registers the Glue table in one call — including partition discovery.
    columns_types, partition_types, _ = wr.s3.store_parquet_metadata(
        path=s3_prefix,
        database=ATHENA_DB,
        table=ATHENA_TABLE,
        dataset=True,
        mode="overwrite",
        description="Home Credit wide features (Glue output, one row per case_id)",
        parameters={
            "Project": "HomeCredit",
            "ManagedBy": "ingest_to_feature_store.py",
            "produced_by": "glue_feature_job",
        },
    )
    log.info("  columns=%d  partitions=%s", len(columns_types), list(partition_types))

    log.info("Refreshing partitions (MSCK REPAIR equivalent)…")
    wr.athena.repair_table(
        table=ATHENA_TABLE,
        database=ATHENA_DB,
        s3_output=athena_staging,
    )

    # Row-count sanity check — should match the ~1.5M case_ids in base.
    log.info("Running count query…")
    result = wr.athena.read_sql_query(
        sql=f"SELECT COUNT(*) AS n FROM {ATHENA_TABLE}",
        database=ATHENA_DB,
        s3_output=athena_staging,
    )
    log.info("✓ Athena table ready.  %s.%s  —  %s rows",
             ATHENA_DB, ATHENA_TABLE, result.iloc[0, 0])


if __name__ == "__main__":
    main()
