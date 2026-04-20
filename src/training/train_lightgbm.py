"""LightGBM baseline training script.

Runs locally for debugging and inside a SageMaker Training Job in Phase 3.
SageMaker convention:
  - training data at /opt/ml/input/data/train/
  - model output at /opt/ml/model/
  - hyperparameters injected via CLI args
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
    return p.parse_args()


def load_parquet_dir(path: str) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def stability_metric(y_true: np.ndarray, y_pred: np.ndarray, weeks: np.ndarray) -> float:
    """Kaggle's stability metric: mean(weekly Gini) - penalty*std(weekly Gini) - falling-trend penalty.

    This is the official competition metric; exact form matters for model selection.
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

    log.info("Loading data from %s and %s", SM_TRAIN_DIR, SM_VAL_DIR)
    train_df = load_parquet_dir(SM_TRAIN_DIR)
    val_df = load_parquet_dir(SM_VAL_DIR)

    drop_cols = [args.target, args.week_col, "case_id"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Cast string/object columns to pandas Categorical so LightGBM can split on them.
    obj_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    if obj_cols:
        log.info("Casting %d object columns to category: %s...", len(obj_cols), obj_cols[:5])
        for c in obj_cols:
            train_df[c] = train_df[c].astype("category")
            # Align val categories to train's to avoid unseen-category errors
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
