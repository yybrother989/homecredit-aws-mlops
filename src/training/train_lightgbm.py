"""LightGBM training script.

Runs in three modes:
  * `--feature-source local`   — reads parquet from SM_CHANNEL_{TRAIN,VALIDATION}
    (SageMaker convention; also the local-dev path).
  * `--feature-source parquet` — reads parquet directly from an S3 prefix
    (useful as a shortcut before Feature Store ingestion completes).
  * `--feature-source athena`  — queries the SageMaker Feature Store offline
    store via Athena; uses the Glue Data Catalog table auto-registered by the
    Feature Group. Honors WEEK_NUM-based train/val split on the fly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_VAL_DIR = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--target", type=str, default="target")
    p.add_argument("--week-col", type=str, default="WEEK_NUM")
    p.add_argument("--feature-source", choices=["local", "parquet", "athena"], default="local")
    p.add_argument("--val-weeks-frac", type=float, default=0.2,
                   help="Fraction of weeks (last) used for validation when source != local")
    return p.parse_args()


def load_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


ATHENA_DB = "homecredit_ml"
ATHENA_TABLE = "features"


def load_from_athena(val_weeks_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query the Glue feature output via Athena external table, then time-split.

    The table is registered by `src/features/ingest_to_feature_store.py`.
    Athena lowercases all identifiers; we restore `WEEK_NUM` capitalization
    after the pull so downstream code doesn't branch on case.
    """
    import awswrangler as wr
    import boto3

    cfn = boto3.client("cloudformation")
    outs = cfn.describe_stacks(StackName="HomeCreditBaseStack")["Stacks"][0]["Outputs"]
    artifacts_bucket = next(o["OutputValue"] for o in outs if o["OutputKey"] == "ArtifactsBucketName")
    staging = f"s3://{artifacts_bucket}/athena/"

    log.info("Pulling %s.%s from Athena (staging %s)", ATHENA_DB, ATHENA_TABLE, staging)
    # ctas_approach=True → Athena materializes results as parquet in staging,
    # which awswrangler reads back. For a wide 975-col / 1.5M-row table this
    # is ~3× smaller and 10× faster than the CSV path.
    df = wr.athena.read_sql_query(
        sql=f"SELECT * FROM {ATHENA_TABLE}",
        database=ATHENA_DB,
        s3_output=staging,
        ctas_approach=True,
    )
    log.info("  pulled %s", df.shape)

    rename = {c: "WEEK_NUM" if c.lower() == "week_num" else c for c in df.columns}
    df = df.rename(columns=rename)

    weeks = sorted(df["WEEK_NUM"].unique().tolist())
    cutoff = int(len(weeks) * (1 - val_weeks_frac))
    val_weeks = set(weeks[cutoff:])
    return df[~df["WEEK_NUM"].isin(val_weeks)], df[df["WEEK_NUM"].isin(val_weeks)]


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.feature_source == "local":
        log.info("Loading local parquet from %s and %s", SM_TRAIN_DIR, SM_VAL_DIR)
        return load_parquet_dir(SM_TRAIN_DIR), load_parquet_dir(SM_VAL_DIR)
    if args.feature_source == "athena":
        return load_from_athena(args.val_weeks_frac)
    raise ValueError(f"--feature-source={args.feature_source} not yet implemented")


def stability_metric(y_true: np.ndarray, y_pred: np.ndarray, weeks: np.ndarray) -> float:
    """Kaggle's stability metric: mean(weekly Gini) + slope-penalty - std-penalty.

    Official competition metric; exact form matters for model selection.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred, "w": weeks})
    ginis = []
    for _, g in df.groupby("w"):
        if g["y"].nunique() < 2:
            continue
        auc = roc_auc_score(g["y"], g["p"])
        ginis.append(2 * auc - 1)
    if len(ginis) < 2:
        return float("nan")
    ginis = np.asarray(ginis)
    x = np.arange(len(ginis))
    slope = np.polyfit(x, ginis, 1)[0]
    residuals = ginis - (slope * x + np.polyfit(x, ginis, 1)[1])
    return float(ginis.mean() + 88.0 * min(0.0, slope) - 0.5 * residuals.std())


def main() -> None:
    args = parse_args()

    train_df, val_df = load_data(args)
    log.info("train=%s  val=%s", train_df.shape, val_df.shape)

    drop_cols = [args.target, args.week_col, "case_id", "date_decision"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Any non-numeric column (pandas `object` from parquet, `string[pyarrow]`
    # from Athena CTAS, or explicit `category`) needs categorical encoding.
    obj_cols = [
        c for c in feature_cols
        if str(train_df[c].dtype) in {"object", "string", "category"}
        or str(train_df[c].dtype).startswith("string")
    ]
    if obj_cols:
        log.info("Casting %d non-numeric columns to category: %s...", len(obj_cols), obj_cols[:5])
        for c in obj_cols:
            train_df[c] = train_df[c].astype("category")
            val_df[c] = val_df[c].astype(
                pd.CategoricalDtype(categories=train_df[c].cat.categories)
            )

    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[args.target], categorical_feature=obj_cols or "auto")
    dval = lgb.Dataset(val_df[feature_cols], label=val_df[args.target], reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 5,
        "verbose": -1,
    }

    log.info("Training with params: %s", params)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(100),
        ],
    )

    val_pred = model.predict(val_df[feature_cols])
    auc = roc_auc_score(val_df[args.target], val_pred)
    stab = stability_metric(
        val_df[args.target].to_numpy(),
        val_pred,
        val_df[args.week_col].to_numpy(),
    )
    log.info("Validation AUC=%.4f, Stability=%.4f", auc, stab)

    Path(SM_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(SM_MODEL_DIR) / "model.txt"))
    with open(Path(SM_MODEL_DIR) / "metrics.json", "w") as f:
        json.dump({"val_auc": auc, "stability": stab, "best_iteration": model.best_iteration}, f)
    with open(Path(SM_MODEL_DIR) / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)


if __name__ == "__main__":
    main()
