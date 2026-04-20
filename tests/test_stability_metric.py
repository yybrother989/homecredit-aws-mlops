"""Sanity check for the stability metric implementation."""
import numpy as np

from src.training.train_lightgbm import stability_metric


def test_stability_runs_and_is_finite():
    rng = np.random.default_rng(0)
    n = 5000
    weeks = rng.integers(0, 20, size=n)
    y = rng.integers(0, 2, size=n)
    # Give predictions some signal correlated with y.
    p = np.clip(y * 0.6 + rng.normal(0.3, 0.1, size=n), 0, 1)
    s = stability_metric(y, p, weeks)
    assert np.isfinite(s)
