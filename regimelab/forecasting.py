"""Causal regime forecasting: targets, model zoo and the walk-forward runner.

What is actually being forecast
-------------------------------
At origin ``t`` and horizon ``h`` two different questions can be asked, and the
original notebook did not distinguish them:

``point_in_time``
    P(the economy is in a risk-off state **on day t+h**).  As ``h`` grows this
    decays towards the unconditional base rate, because a single distant day
    is close to a draw from the stationary distribution.

``any_in_window``
    P(**at least one** risk-off day occurs in ``(t, t+h]``).  This increases
    with ``h`` and is the quantity a risk manager actually cares about.

Both are reported.  Confusing them is why horizon curves in regime dashboards
often look nonsensical.

Avoiding look-ahead
-------------------
The target is the *filtered* regime state -- what a real-time nowcast would
have said on day ``t+h`` using only data up to ``t+h``.  The labeller itself is
refitted inside every walk-forward fold on training data only, so neither the
regime definition nor the feature scaling ever sees the evaluation period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import Fold, walk_forward_folds
from .features import label_feature_frame
from .labeling import RegimeLabeler, empirical_transition_matrix

# Economically chosen, deliberately small design.  With an effective sample of
# a few hundred independent observations, a 60-column design is mostly noise.
COMPACT_COLUMNS: tuple[str, ...] = (
    "VIX_log",
    "VIX_vrp",
    "Baa_log",
    "Baa_chg63",
    "SPX_rmom21",
    "SPX_rmom63",
    "SPX_dd252",
    "SPX_logvol21",
    "SPX_volratio",
    "Slope_10Y2Y_level",
    "RUT_rel_SPX_63",
    "SPX_bond_corr63",
)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def point_in_time_target(risk_off: np.ndarray, horizon: int) -> np.ndarray:
    """y[t] = risk_off[t + h]; NaN where undefined."""
    a = np.asarray(risk_off, dtype=float)
    out = np.full(len(a), np.nan)
    if horizon < len(a):
        out[: len(a) - horizon] = a[horizon:]
    return out


def any_in_window_target(risk_off: np.ndarray, horizon: int) -> np.ndarray:
    """y[t] = 1 if any risk-off day falls in (t, t+h]."""
    a = np.asarray(risk_off, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    if n > horizon:
        w = sliding_window_view(a, horizon)       # w[i] = a[i : i+h]
        out[: n - horizon] = w[1 : n - horizon + 1].max(axis=1)
    return out


def forward_market_quantities(panel: pd.DataFrame, index: pd.Index,
                              horizon: int) -> dict[str, np.ndarray]:
    """Forward realised outcomes, used for the model-free control targets.

    These matter because the regime targets are partly *mechanically*
    predictable: a 63-day momentum feature observed at ``t+h`` still shares
    ``63-h`` of its 63 days with the same feature observed at ``t``, so a model
    can appear to forecast the future state while only extrapolating the
    roll-off of its own windows.  Forward return and forward realised
    volatility have no such overlap with anything known at ``t``.
    """
    px = np.log(panel["SPX"]).reindex(index)
    fwd_ret = (px.shift(-horizon) - px).values
    r = px.diff().values
    n = len(index)

    fwd_vol = np.full(n, np.nan)
    if n > horizon:
        w = sliding_window_view(r, horizon)
        fwd_vol[: n - horizon] = np.nanstd(w[1: n - horizon + 1], axis=1)

    return {"fwd_ret": fwd_ret, "fwd_vol": fwd_vol}


def build_target(
    target_kind: str,
    horizon: int,
    risk_off: np.ndarray,
    market: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> np.ndarray:
    """Target vector for the whole series; NaN where undefined.

    Any threshold is estimated on ``train_idx`` only.
    """
    if target_kind == "point_in_time":
        return point_in_time_target(risk_off, horizon)
    if target_kind == "any_in_window":
        return any_in_window_target(risk_off, horizon)
    if target_kind == "mkt_negative_return":
        r = market["fwd_ret"]
        return np.where(np.isfinite(r), (r < 0).astype(float), np.nan)
    if target_kind == "mkt_high_vol":
        v = market["fwd_vol"]
        thr = np.nanmedian(v[train_idx])
        return np.where(np.isfinite(v), (v > thr).astype(float), np.nan)
    raise ValueError(f"unknown target_kind {target_kind!r}")


MODEL_DEFINED_TARGETS = ("point_in_time", "any_in_window")


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def build_designs(features: pd.DataFrame, state_proba: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Design matrices.  Every design includes the current state posterior so
    that a feature-based model *nests* the persistence baseline and any gain
    it shows is genuinely incremental."""
    sp = state_proba.copy()
    sp.columns = [f"state_p{c}" for c in sp.columns]
    compact = [c for c in COMPACT_COLUMNS if c in features.columns]
    return {
        "state": sp,
        "compact": pd.concat([features[compact], sp], axis=1),
        "full": pd.concat([features, sp], axis=1),
    }


def model_zoo(random_state: int = 42) -> dict[str, object]:
    return {
        "logit": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"),
        ),
        "logit_reg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.02, max_iter=3000, solver="lbfgs"),
        ),
        # The original notebook's setting.  Re-weighting the classes optimises
        # a decision rule, not a probability, so it should damage log loss.
        "logit_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs",
                               class_weight="balanced"),
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=100,
            random_state=random_state,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=100, l2_regularization=1.0,
            random_state=random_state,
        ),
    }


DEFAULT_COMBOS: tuple[tuple[str, str], ...] = (
    ("state", "logit"),
    ("compact", "logit"),
    ("compact", "logit_reg"),
    ("compact", "hgb"),
    ("full", "logit_reg"),
    ("full", "rf"),
    ("full", "hgb"),
)

# Subset carried into the live application: the two benchmarks plus the model
# families that showed non-negative out-of-sample skill in experiment 3.
# ``full|rf`` and the unregularised ``compact|logit`` are excluded because
# they were measurably worse than the benchmark, and re-fitting them for the
# app would only slow it down to display a number we have shown not to trust.
APP_COMBOS: tuple[tuple[str, str], ...] = (
    ("state", "logit"),
    ("compact", "logit_reg"),
    ("compact", "hgb"),
    ("full", "hgb"),
)


# --------------------------------------------------------------------------
# Walk-forward runner
# --------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    horizon: int
    target_kind: str
    dates: pd.DatetimeIndex
    y: np.ndarray
    preds: dict[str, np.ndarray]
    state_at_t: np.ndarray
    fold_ids: np.ndarray
    n_folds: int = 0
    meta: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.preds, index=self.dates)
        df.insert(0, "y", self.y)
        df.insert(1, "state_at_t", self.state_at_t)
        df.insert(2, "fold", self.fold_ids)
        return df


def _markov_h_step(labels: np.ndarray, horizon: int, k: int) -> np.ndarray:
    """Empirical P(state_{t+h} | state_t) estimated directly at horizon h."""
    return empirical_transition_matrix(
        pd.Series(labels), horizon=horizon, n_regimes=k
    ).values


def run_walk_forward(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int,
    target_kind: str = "point_in_time",
    n_regimes: int = 4,
    labeler_method: str = "hmm",
    combos: tuple[tuple[str, str], ...] = DEFAULT_COMBOS,
    initial_train: int = 2000,
    test_block: int = 252,
    labeler_n_init: int = 1,
    random_state: int = 42,
    design_features: pd.DataFrame | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Refit labeller and forecasters inside each fold; return pooled OOS predictions.

    ``design_features`` lets an ablation restrict what the *forecasters* see
    while the labeller keeps its full feature set, so the regime definition
    stays fixed across ablation arms and only the predictors change.
    """
    L = label_feature_frame(features)
    design_source = features if design_features is None else design_features
    n = len(features)
    folds = walk_forward_folds(n, horizon, initial_train, test_block)
    if not folds:
        raise RuntimeError("no folds: sample too short for these settings")

    market = forward_market_quantities(panel, features.index, horizon)
    model_defined = target_kind in MODEL_DEFINED_TARGETS

    zoo = model_zoo(random_state)
    names = ["uncond", "persist_markov"] + (
        ["hmm_analytic"] if (labeler_method == "hmm" and model_defined) else []
    ) + [f"{d}|{m}" for d, m in combos]

    acc: dict[str, list[np.ndarray]] = {nm: [] for nm in names}
    ys, dates, states, fids = [], [], [], []

    for fi, fold in enumerate(folds):
        tr = slice(fold.train_start, fold.train_end)
        te = slice(fold.test_start, fold.test_end)

        labeler = RegimeLabeler(
            n_regimes=n_regimes, method=labeler_method,
            random_state=random_state, n_init=labeler_n_init,
        ).fit(L.iloc[tr])

        # Causal state estimates for the whole series under frozen parameters.
        filt = labeler.filtered_proba(L)
        hard = filt.values.argmax(axis=1)
        risk_off_mask = labeler.risk_off_mask()
        risk_off = risk_off_mask[hard].astype(float)

        tr_idx = np.arange(fold.train_start, fold.train_end)
        te_idx = np.arange(fold.test_start, fold.test_end)
        y_all = build_target(target_kind, horizon, risk_off, market, tr_idx)

        designs = build_designs(design_source, filt)
        tr_idx = tr_idx[np.isfinite(y_all[tr_idx])]
        te_idx = te_idx[np.isfinite(y_all[te_idx])]
        if len(te_idx) == 0 or len(np.unique(y_all[tr_idx])) < 2:
            continue

        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # ---- baselines ----------------------------------------------------
        base_rate = float(np.clip(y_tr.mean(), 1e-4, 1 - 1e-4))
        acc["uncond"].append(np.full(len(te_idx), base_rate))

        if target_kind == "point_in_time":
            P = _markov_h_step(hard[tr_idx], horizon, n_regimes)
            p_off = P @ risk_off_mask.astype(float)
        else:
            # For every other target the benchmark is the regime-conditional
            # frequency read straight off the training sample: still "today's
            # state and nothing else", but without assuming the target is a
            # state indicator.
            p_off = np.array([
                y_tr[hard[tr_idx] == s].mean() if (hard[tr_idx] == s).any() else base_rate
                for s in range(n_regimes)
            ])
        acc["persist_markov"].append(np.clip(p_off[hard[te_idx]], 1e-4, 1 - 1e-4))

        if labeler_method == "hmm" and model_defined:
            A = labeler.model_.transmat_
            Ah = np.linalg.matrix_power(A, horizon)
            # transmat_ is in raw component order; map to canonical
            inv = labeler.inverse_order_
            Ah_canon = Ah[np.ix_(inv, inv)]
            post = filt.values[te_idx]
            if target_kind == "point_in_time":
                p = (post @ Ah_canon) @ risk_off_mask.astype(float)
            else:
                # P(no risk-off day in t+1..t+h): one step from any state into
                # the risk-on block, then h-1 steps confined to that block.
                sub = np.arange(n_regimes)[~risk_off_mask]
                if len(sub) == 0:
                    p = np.ones(len(te_idx))
                else:
                    A_can = A[np.ix_(inv, inv)]
                    step_in = A_can[:, sub]
                    stay = np.linalg.matrix_power(A_can[np.ix_(sub, sub)],
                                                  max(horizon - 1, 0))
                    surv = (post @ step_in) @ stay
                    p = 1.0 - surv.sum(axis=1)
            acc["hmm_analytic"].append(np.clip(p, 1e-4, 1 - 1e-4))

        # ---- learned models ----------------------------------------------
        for dname, mname in combos:
            X = designs[dname]
            est = clone(zoo[mname])
            est.fit(X.iloc[tr_idx].values, y_tr.astype(int))
            p = est.predict_proba(X.iloc[te_idx].values)[:, 1]
            acc[f"{dname}|{mname}"].append(np.clip(p, 1e-4, 1 - 1e-4))

        ys.append(y_te)
        dates.append(features.index[te_idx])
        states.append(hard[te_idx])
        fids.append(np.full(len(te_idx), fi))

        if verbose:
            print(f"    fold {fi + 1}/{len(folds)}  train={len(tr_idx)} "
                  f"test={len(te_idx)}  {features.index[te_idx[0]].date()}"
                  f"..{features.index[te_idx[-1]].date()}", end="\r")

    if verbose:
        print()

    return WalkForwardResult(
        horizon=horizon,
        target_kind=target_kind,
        dates=pd.DatetimeIndex(np.concatenate(dates)),
        y=np.concatenate(ys),
        preds={k: np.concatenate(v) for k, v in acc.items() if v},
        state_at_t=np.concatenate(states),
        fold_ids=np.concatenate(fids),
        n_folds=len(folds),
        meta={"n_regimes": n_regimes, "labeler": labeler_method},
    )
