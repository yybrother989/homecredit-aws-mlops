"""SageMaker endpoint handler (SKLearn-container compatible).

Phase 4 wires this into a real-time endpoint; Phase 5 adds Model Monitor capture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = []


def model_fn(model_dir: str):
    global FEATURE_NAMES
    FEATURE_NAMES = json.loads(Path(model_dir, "feature_names.json").read_text())
    return lgb.Booster(model_file=str(Path(model_dir, "model.txt")))


def input_fn(body: str, content_type: str = "application/json") -> pd.DataFrame:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")
    payload = json.loads(body)
    rows = payload["instances"] if "instances" in payload else [payload]
    return pd.DataFrame(rows).reindex(columns=FEATURE_NAMES)


def predict_fn(df: pd.DataFrame, model) -> np.ndarray:
    return model.predict(df)


def output_fn(prediction: np.ndarray, accept: str = "application/json") -> str:
    return json.dumps({"predictions": prediction.tolist()})
