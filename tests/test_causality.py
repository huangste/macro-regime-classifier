"""Look-ahead tests.

These are the tests that matter most in this project.  A regime model that
leaks a few days of future information will look excellent in backtest and be
worthless live, and the leak is invisible in ordinary unit tests because every
function returns plausible numbers either way.  The checks here perturb the
future and assert that the past does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.evaluation import newey_west_se, walk_forward_folds
from regimelab.features import build_features
from regimelab.forecasting import any_in_window_target, point_in_time_target
from regimelab.labeling import RegimeLabeler
from tests.conftest import make_panel


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def test_point_in_time_target_is_a_pure_shift():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=float)
    out = point_in_time_target(a, 3)
    assert np.array_equal(out[:5], a[3:])
    assert np.isnan(out[5:]).all()


def test_any_in_window_covers_the_open_interval():
    a = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    out = any_in_window_target(a, 3)
    # from t=0 the window is days 1,2,3 -> contains the 1 at index 3
    assert out[0] == 1
    # from t=1 the window is 2,3,4 -> contains index 3
    assert out[1] == 1
    # from t=4 the window is 5,6,7 -> all zero
    assert out[4] == 0
    # the current day never counts towards its own window
    b = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    assert any_in_window_target(b, 2)[0] == 0


def test_any_in_window_dominates_point_in_time():
    rng = np.random.default_rng(0)
    a = (rng.random(500) < 0.3).astype(float)
    for h in (1, 5, 25):
        pit = point_in_time_target(a, h)
        anyw = any_in_window_target(a, h)
        ok = np.isfinite(pit) & np.isfinite(anyw)
        assert (anyw[ok] >= pit[ok]).all()


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def test_features_do_not_use_future_prices():
    """Corrupt the tail of the panel; features before the cut must not move."""
    base = make_panel(n=1500, seed=3)
    cut = 1200

    perturbed = base.copy()
    rng = np.random.default_rng(99)
    perturbed.iloc[cut:] = perturbed.iloc[cut:] * rng.uniform(1.5, 2.5)

    f0 = build_features(base)
    f1 = build_features(perturbed)

    common = f0.index.intersection(f1.index)
    common = common[common < base.index[cut]]
    assert len(common) > 500
    pd.testing.assert_frame_equal(f0.loc[common], f1.loc[common])


# --------------------------------------------------------------------------
# State estimation
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted(label_frame):
    return RegimeLabeler(n_regimes=3, method="hmm", init="score").fit(label_frame)


def test_filtered_probabilities_are_causal(fitted, label_frame):
    """Filtered posteriors at t must depend only on observations up to t."""
    cut = len(label_frame) - 200
    full = fitted.filtered_proba(label_frame)
    truncated = fitted.filtered_proba(label_frame.iloc[:cut])
    np.testing.assert_allclose(
        full.values[:cut], truncated.values, atol=1e-8,
        err_msg="filtered state estimate changed when future data was added",
    )


def test_smoothed_probabilities_use_information_the_forecaster_cannot_have():
    """The complement of the test above: this documents *why* the forecasting
    path must never consume ``predict_proba``.

    Emissions are deliberately overlapping.  With well-separated states the
    posterior is near-degenerate and smoothing changes nothing, which is why
    the real-data version of this check has to be read at the sample edge
    rather than in the middle of history.
    """
    rng = np.random.default_rng(5)
    n = 1500
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.98 else 1 - state[t - 1]
    # Two conditionally independent, heavily overlapping signals.  Collinear
    # columns would give a singular emission covariance and a degenerate
    # posterior, which would make the test pass or fail for the wrong reason.
    X = pd.DataFrame(
        {
            "SPX_rmom63": rng.normal(np.where(state == 1, -0.5, 0.5), 1.0),
            "VIX_log": rng.normal(np.where(state == 1, 0.5, -0.5), 1.0),
        },
        index=pd.bdate_range("2000-01-03", periods=n),
    )
    lab = RegimeLabeler(n_regimes=2, method="hmm", init="score").fit(X)

    smoothed = lab.predict_proba(X).values
    filtered = lab.filtered_proba(X).values
    assert np.abs(smoothed - filtered).max() > 0.05, (
        "smoothed and filtered posteriors coincided; the causal distinction "
        "would then be vacuous"
    )

    # And the revision is concentrated at the edge of the sample, which is
    # precisely where a live forecast is made.
    cut = n - 300
    trunc = lab.predict_proba(X.iloc[:cut]).values
    edge = np.abs(smoothed[cut - 20:cut] - trunc[-20:]).max()
    assert edge > 1e-4, "smoothed estimates near the cut did not get revised"


# --------------------------------------------------------------------------
# Walk-forward splitting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 5, 25])
def test_folds_are_purged_by_the_horizon(horizon):
    folds = walk_forward_folds(4000, horizon, initial_train=1000, test_block=250)
    assert folds
    for f in folds:
        assert f.train_end + horizon <= f.test_start, "training label leaks into test"
        assert f.train_start < f.train_end < f.test_start < f.test_end


def test_folds_do_not_overlap_and_move_forward():
    folds = walk_forward_folds(4000, 5, initial_train=1000, test_block=250)
    for a, b in zip(folds, folds[1:]):
        assert b.test_start >= a.test_end
        assert b.train_end >= a.train_end


def test_no_fold_runs_past_the_end_of_the_sample():
    n, h = 4000, 25
    for f in walk_forward_folds(n, h, 1000, 250):
        assert f.test_end + h <= n


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def test_hac_se_exceeds_iid_se_under_positive_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=4000)
    x = pd.Series(e).rolling(21).mean().dropna().values   # overlapping windows
    iid = x.std(ddof=1) / np.sqrt(len(x))
    hac = newey_west_se(x, lag=42)
    assert hac > 2 * iid, "HAC variance failed to widen for overlapping data"
