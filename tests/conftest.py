"""Synthetic fixtures so the test suite never needs the network."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


def make_panel(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A two-regime market with persistent switching, in the schema of the
    real panel so the feature and labelling code runs unmodified."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n, name="Date")

    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        p_switch = 0.004 if state[t - 1] == 0 else 0.02
        state[t] = 1 - state[t - 1] if rng.random() < p_switch else state[t - 1]

    vol = np.where(state == 1, 0.022, 0.007)
    drift = np.where(state == 1, -0.0006, 0.0005)
    r = rng.normal(drift, vol)

    spx = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    panel = pd.DataFrame(index=idx)
    panel["SPX"] = spx
    panel["NDX"] = 100 * np.exp(np.cumsum(r * 1.2 + rng.normal(0, 0.004, n)))
    panel["Nikkei"] = 100 * np.exp(np.cumsum(r * 0.8 + rng.normal(0, 0.006, n)))
    panel["RUT"] = 100 * np.exp(np.cumsum(r * 1.1 + rng.normal(0, 0.005, n)))
    panel["Oil"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    panel["USDJPY"] = 110 * np.exp(np.cumsum(-r * 0.3 + rng.normal(0, 0.004, n)))
    panel["DXY"] = 95 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    panel["VIX"] = np.clip(vol * np.sqrt(252) * 100 * np.exp(rng.normal(0, 0.15, n)), 8, 90)
    panel["DGS10"] = np.clip(3 + np.cumsum(rng.normal(0, 0.03, n)), 0.3, 9)
    panel["DGS2"] = np.clip(panel["DGS10"] - 1 + np.cumsum(rng.normal(0, 0.02, n)), 0.05, 9)
    panel["DGS3MO"] = np.clip(panel["DGS2"] - 0.3, 0.01, 9)
    panel["BaaSpread"] = np.clip(1.8 + state * 1.5 + np.cumsum(rng.normal(0, 0.01, n)), 0.4, 8)
    panel["Slope_10Y2Y"] = panel["DGS10"] - panel["DGS2"]
    panel["Slope_10Y3M"] = panel["DGS10"] - panel["DGS3MO"]
    panel["QualitySpread"] = np.clip(0.8 + state * 0.6 + rng.normal(0, 0.05, n), 0.1, 4)
    panel.attrs["true_state"] = state
    return panel


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture(scope="session")
def features(panel) -> pd.DataFrame:
    from regimelab.features import build_features

    return build_features(panel)


@pytest.fixture(scope="session")
def label_frame(features) -> pd.DataFrame:
    from regimelab.features import label_feature_frame

    return label_feature_frame(features)
