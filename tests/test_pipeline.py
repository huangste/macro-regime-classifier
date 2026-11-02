"""The production build: its CPU-saving thread limit must not change results."""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from regimelab.labeling import RegimeLabeler
from regimelab.pipeline import live_forecasts


def test_one_thread_gives_exactly_the_same_forecasts(panel, features, label_frame):
    """build_state runs the forecast fits on one thread to avoid burning CPU
    on thread coordination.  That is only acceptable if the forecasts are
    bit-for-bit what the multi-threaded fits produce."""
    lab = RegimeLabeler(n_regimes=3, method="hmm", init="score").fit(label_frame)
    proba = lab.filtered_proba(label_frame)
    hard = proba.values.argmax(axis=1)
    risk_off = lab.risk_off_mask()[hard].astype(float)

    def run():
        return (live_forecasts(features, panel, lab, proba, risk_off, hard, horizons=(5,))
                .set_index(["horizon", "target", "model"])["prob"].sort_index())

    many = run()
    with threadpool_limits(limits=1):
        one = run()

    assert len(many) > 0 and many.index.equals(one.index)
    assert np.array_equal(many.values, one.values)
