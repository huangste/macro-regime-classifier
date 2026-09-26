"""Production pipeline: fit everything once, cache it, serve it to the app.

Two clearly separated products come out of here.

*The historical classification* is fitted on the whole sample.  That is
legitimate and deliberate: describing what regime the market was in during
2008 is a smoothing problem, not a forecasting one, and using the whole sample
gives the sharpest description.  It is labelled as in-sample everywhere it is
shown.

*The live forecast* uses only causal inputs -- filtered state posteriors and a
transition matrix estimated on data up to the forecast origin -- so the number
quoted for today is produced the same way as the numbers that were scored out
of sample in experiment 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import ARTIFACT_DIR, FORECAST_HORIZONS
from .data import build_panel, download_all
from .features import build_features, label_feature_frame
from .forecasting import (
    APP_COMBOS,
    any_in_window_target,
    build_designs,
    forward_market_quantities,
    model_zoo,
    point_in_time_target,
)
from .labeling import (
    RegimeLabeler,
    empirical_transition_matrix,
    episode_table,
    regime_asset_returns,
    regime_characteristics,
)

STATE_FILE = ARTIFACT_DIR / "model_state.joblib"

# Models that are averaged with the persistence benchmark for display.
BLEND_WITH: tuple[str, ...] = ("hmm_analytic", "compact|logit_reg",
                               "compact|hgb", "full|hgb")


@dataclass
class RegimeModelState:
    """Everything the Streamlit app needs, computed once."""

    built_at: str
    sample: str
    n_regimes: int
    labeler_method: str

    panel: pd.DataFrame
    features: pd.DataFrame

    labels: pd.Series                 # descriptive (smoothed) canonical labels
    proba_smoothed: pd.DataFrame
    proba_filtered: pd.DataFrame
    risk_off: pd.Series               # descriptive binary
    risk_off_filtered: pd.Series      # causal binary (real-time nowcast)
    risk_score: pd.Series             # continuous risk-appetite index

    regime_names: list[str]
    regime_scores: np.ndarray
    risk_off_mask: np.ndarray
    characteristics: pd.DataFrame
    asset_returns: pd.DataFrame
    episodes: pd.DataFrame
    transitions: dict[int, pd.DataFrame]
    expected_durations: pd.Series

    forecasts: pd.DataFrame           # live forward probabilities
    labeler: object = None
    meta: dict = field(default_factory=dict)


def expected_durations(trans_1d: pd.DataFrame) -> pd.Series:
    """Mean sojourn time in trading days implied by the one-step matrix."""
    p_stay = np.clip(np.diag(trans_1d.values), 0, 1 - 1e-9)
    return pd.Series(1.0 / (1.0 - p_stay), index=trans_1d.index, name="expected_days")


def live_forecasts(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    labeler: RegimeLabeler,
    proba_filtered: pd.DataFrame,
    risk_off_filtered: np.ndarray,
    hard_filtered: np.ndarray,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    combos: tuple[tuple[str, str], ...] = APP_COMBOS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Forward probabilities as of the last available date.

    Reported per horizon and per target definition, for the two benchmarks and
    for the learned models, so the app can show them side by side against the
    measured out-of-sample skill of each.
    """
    n_regimes = labeler.n_regimes
    mask = labeler.risk_off_mask().astype(float)
    designs = build_designs(features, proba_filtered)
    zoo = model_zoo(random_state)
    post_now = proba_filtered.values[-1]
    state_now = int(hard_filtered[-1])

    A = labeler.model_.transmat_ if labeler.method == "hmm" else None
    inv = labeler.inverse_order_

    rows = []
    for h in horizons:
        market = forward_market_quantities(panel, features.index, h)
        for kind in ("point_in_time", "any_in_window", "mkt_high_vol"):
            if kind == "point_in_time":
                y_all = point_in_time_target(risk_off_filtered, h)
            elif kind == "any_in_window":
                y_all = any_in_window_target(risk_off_filtered, h)
            else:
                v = market["fwd_vol"]
                y_all = np.where(np.isfinite(v),
                                 (v > np.nanmedian(v)).astype(float), np.nan)
            ok = np.isfinite(y_all)
            idx = np.where(ok)[0]
            y = y_all[idx]

            base = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
            rows.append(dict(horizon=h, target=kind, model="uncond", prob=base))

            if kind == "point_in_time":
                P = empirical_transition_matrix(
                    pd.Series(hard_filtered[idx]), horizon=h, n_regimes=n_regimes).values
                p_off = P @ mask
            else:
                p_off = np.array([
                    y[hard_filtered[idx] == s].mean()
                    if (hard_filtered[idx] == s).any() else base
                    for s in range(n_regimes)
                ])
            rows.append(dict(horizon=h, target=kind, model="persist_markov",
                             prob=float(np.clip(p_off[state_now], 1e-4, 1 - 1e-4))))

            if A is not None and kind != "mkt_high_vol":
                A_can = A[np.ix_(inv, inv)]
                if kind == "point_in_time":
                    p = float((post_now @ np.linalg.matrix_power(A_can, h)) @ mask)
                else:
                    sub = np.arange(n_regimes)[~labeler.risk_off_mask()]
                    if len(sub) == 0:
                        p = 1.0
                    else:
                        stay = np.linalg.matrix_power(A_can[np.ix_(sub, sub)], max(h - 1, 0))
                        p = float(1.0 - ((post_now @ A_can[:, sub]) @ stay).sum())
                rows.append(dict(horizon=h, target=kind, model="hmm_analytic",
                                 prob=float(np.clip(p, 1e-4, 1 - 1e-4))))

            for dname, mname in combos:
                X = designs[dname]
                est = clone(zoo[mname])
                est.fit(X.iloc[idx].values, y.astype(int))
                p = float(est.predict_proba(X.iloc[[-1]].values)[0, 1])
                rows.append(dict(horizon=h, target=kind, model=f"{dname}|{mname}",
                                 prob=float(np.clip(p, 1e-4, 1 - 1e-4))))

    fc = pd.DataFrame(rows)

    # Equal-weight combination with the persistence benchmark.  Experiment 7
    # shows this is what makes the improvement statistically visible at every
    # horizon: the learned models are informative about turns and overconfident
    # about continuation, and averaging keeps the first while damping the
    # second.  The weight is fixed at one half rather than estimated.
    base = fc[fc["model"] == "persist_markov"].set_index(["horizon", "target"])["prob"]
    blends = []
    for m in BLEND_WITH:
        sub = fc[fc["model"] == m]
        for r in sub.itertuples():
            b = base.get((r.horizon, r.target))
            if b is None:
                continue
            blends.append(dict(horizon=r.horizon, target=r.target,
                               model=f"blend50|{m}", prob=0.5 * (r.prob + b)))
    return pd.concat([fc, pd.DataFrame(blends)], ignore_index=True)


def build_state(
    sample: str = "core_1990",
    n_regimes: int = 4,
    labeler_method: str = "hmm",
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    refresh_data: bool = False,
    random_state: int = 42,
) -> RegimeModelState:
    panel = build_panel(download_all(refresh=refresh_data), sample=sample)
    features = build_features(panel)
    L = label_feature_frame(features)

    labeler = RegimeLabeler(n_regimes=n_regimes, method=labeler_method,
                            init="score", random_state=random_state).fit(L)

    labels = labeler.predict(L)
    proba_s = labeler.predict_proba(L)
    proba_f = labeler.filtered_proba(L)
    hard_f = proba_f.values.argmax(axis=1)
    mask = labeler.risk_off_mask()

    risk_off = labeler.binary_labels(labels)
    risk_off_f = pd.Series(mask[hard_f].astype(int), index=L.index,
                           name="risk_off_filtered")

    from .labeling import risk_appetite_score
    score = pd.Series(
        risk_appetite_score(labeler.scaler_.transform(L.values), list(L.columns)),
        index=L.index, name="risk_score")

    chars = regime_characteristics(panel, labels, n_regimes)
    chars.insert(0, "name", labeler.names())
    chars.insert(1, "risk_score", labeler.regime_scores_)

    trans = {h: empirical_transition_matrix(labels, horizon=h, n_regimes=n_regimes)
             for h in (1,) + tuple(horizons)}

    fc = live_forecasts(features, panel, labeler, proba_f, risk_off_f.values.astype(float),
                        hard_f, horizons=horizons, random_state=random_state)

    return RegimeModelState(
        built_at=datetime.now().isoformat(timespec="seconds"),
        sample=sample,
        n_regimes=n_regimes,
        labeler_method=labeler_method,
        panel=panel,
        features=features,
        labels=labels,
        proba_smoothed=proba_s,
        proba_filtered=proba_f,
        risk_off=risk_off,
        risk_off_filtered=risk_off_f,
        risk_score=score,
        regime_names=labeler.names(),
        regime_scores=labeler.regime_scores_,
        risk_off_mask=mask,
        characteristics=chars,
        asset_returns=regime_asset_returns(panel, labels, n_regimes),
        episodes=episode_table(labels, min_days=5),
        transitions=trans,
        expected_durations=expected_durations(trans[1]),
        forecasts=fc,
        labeler=labeler,
        meta={"n_days": len(features),
              "start": str(features.index[0].date()),
              "end": str(features.index[-1].date())},
    )


def save_state(state: RegimeModelState) -> None:
    import joblib

    joblib.dump(state, STATE_FILE, compress=3)
    (ARTIFACT_DIR / "state_meta.json").write_text(
        json.dumps({"built_at": state.built_at, **state.meta}, indent=2),
        encoding="utf-8")


def load_state() -> RegimeModelState | None:
    import joblib

    if not STATE_FILE.exists():
        return None
    try:
        return joblib.load(STATE_FILE)
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover
    import warnings

    warnings.filterwarnings("ignore")
    st = build_state()
    save_state(st)
    print(f"built {st.built_at}  {st.meta}")
    print(st.characteristics.round(3).to_string())
    print()
    print(st.forecasts.pivot_table(index=["target", "horizon"], columns="model",
                                   values="prob").round(3).to_string())
