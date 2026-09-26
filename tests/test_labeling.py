"""Tests for the property the original notebook lacked: stable regime labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimelab.labeling import (
    RegimeLabeler,
    empirical_transition_matrix,
    episode_table,
    risk_appetite_score,
)


@pytest.mark.parametrize("method", ["hmm", "gmm", "score"])
def test_canonical_labels_are_ordered_by_risk_appetite(label_frame, method):
    lab = RegimeLabeler(n_regimes=4, method=method, init="score").fit(label_frame)
    s = lab.regime_scores_
    assert np.all(np.diff(s) >= 0), f"{method}: canonical order is not monotone"


@pytest.mark.parametrize("method", ["hmm", "score"])
def test_labels_are_invariant_to_the_random_seed(label_frame, method):
    """The defect that motivated this project: a different seed must not
    change which days are called risk-off."""
    a = RegimeLabeler(n_regimes=4, method=method, init="score", random_state=0).fit(label_frame)
    b = RegimeLabeler(n_regimes=4, method=method, init="score", random_state=123).fit(label_frame)
    ya, yb = a.predict(label_frame), b.predict(label_frame)
    assert np.array_equal(ya.values, yb.values)
    assert np.array_equal(a.binary_labels(ya).values, b.binary_labels(yb).values)


def test_risk_off_states_have_worse_conditions(panel, label_frame):
    lab = RegimeLabeler(n_regimes=4, method="hmm", init="score").fit(label_frame)
    y = lab.predict(label_frame)
    off = lab.binary_labels(y).values.astype(bool)
    vix = panel["VIX"].reindex(label_frame.index).values
    assert vix[off].mean() > vix[~off].mean()
    spread = panel["BaaSpread"].reindex(label_frame.index).values
    assert spread[off].mean() > spread[~off].mean()


def test_risk_off_share_stays_inside_its_bounds(label_frame):
    for k in (2, 3, 4, 5):
        lab = RegimeLabeler(n_regimes=k, method="hmm", init="score").fit(label_frame)
        share = float(lab.binary_labels(lab.predict(label_frame)).mean())
        assert 0.05 < share < 0.75, f"K={k}: degenerate risk-off share {share:.2f}"


def test_risk_score_signs_match_the_economics(label_frame):
    """Raising volatility must lower the score; raising momentum must raise it."""
    z = pd.DataFrame(np.zeros((3, label_frame.shape[1])), columns=label_frame.columns)
    z.loc[1, "VIX_log"] = 3.0
    z.loc[2, "SPX_rmom63"] = 3.0
    s = risk_appetite_score(z, list(z.columns))
    assert s[1] < s[0] < s[2]


def test_transition_matrix_rows_are_distributions():
    y = pd.Series([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 0, 1])
    for h in (1, 3):
        T = empirical_transition_matrix(y, horizon=h, n_regimes=3)
        np.testing.assert_allclose(T.sum(axis=1).values, 1.0, atol=1e-9)
        assert (T.values >= 0).all()


def test_transition_matrix_recovers_a_known_chain():
    rng = np.random.default_rng(0)
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    s = [0]
    for _ in range(200_000):
        s.append(int(rng.random() > A[s[-1], 0]) if s[-1] == 0
                 else int(rng.random() < A[1, 1]))
    T = empirical_transition_matrix(pd.Series(s), horizon=1, n_regimes=2)
    np.testing.assert_allclose(T.values, A, atol=0.02)


def test_episodes_partition_the_history(label_frame):
    lab = RegimeLabeler(n_regimes=3, method="hmm", init="score").fit(label_frame)
    y = lab.predict(label_frame)
    ep = episode_table(y, min_days=1)
    assert ep["days"].sum() == len(y)
    assert (ep["regime"].values[1:] != ep["regime"].values[:-1]).all()


def test_hmm_is_more_persistent_than_the_raw_score_buckets(label_frame):
    """The HMM exists to stop the classification flickering day to day."""
    def switches(method):
        lab = RegimeLabeler(n_regimes=4, method=method, init="score").fit(label_frame)
        v = lab.predict(label_frame).values
        return int((v[1:] != v[:-1]).sum())

    assert switches("hmm") < switches("score") / 3
