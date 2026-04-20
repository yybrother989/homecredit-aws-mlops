"""Evaluation step — runs inside SageMaker Processing after training.

Reads model artifact + validation parquet, writes evaluation.json matching the
JsonGet path used by the ConditionStep: metrics.stability.value
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODEL_DIR = Path("/opt/ml/processing/model")
VAL_DIR = Path("/opt/ml/processing/validation")
OUT_DIR = Path("/opt/ml/processing/evaluation")


def stability_metric(y_true, y_pred, weeks) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_pred, "w": weeks})
    ginis = []
    for _, g in df.groupby("w"):
        if g["y"].nunique() < 2:
            continue
        ginis.append(2 * roc_auc_score(g["y"], g["p"]) - 1)
    if len(ginis) < 2:
        return float("nan")
    ginis = np.asarray(ginis)
    x = np.arange(len(ginis))
    slope, intercept = np.polyfit(x, ginis, 1)
    residuals = ginis - (slope * x + intercept)
    return float(ginis.mean() + 88.0 * min(0.0, slope) - 0.5 * residuals.std())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tar = next(MODEL_DIR.glob("*.tar.gz"))
    with tarfile.open(tar) as t:
        t.extractall(MODEL_DIR)
    model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))

    val = pd.concat([pd.read_parquet(f) for f in VAL_DIR.glob("*.parquet")])
    features = json.loads((MODEL_DIR / "feature_names.json").read_text())
    y = val["target"].to_numpy()
    pred = model.predict(val[features])

    auc = roc_auc_score(y, pred)
    stab = stability_metric(y, pred, val["WEEK_NUM"].to_numpy())

    report = {
        "metrics": {
            "auc": {"value": float(auc)},
            "stability": {"value": float(stab)},
        }
    }
    (OUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
