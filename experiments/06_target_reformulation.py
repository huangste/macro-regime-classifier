"""Experiment 6 -- does the *target* rather than the model limit the forecast?

Experiment 3 found that predicting the discrete risk-off state h days ahead
barely beats a persistence benchmark.  Experiment 4 found that predicting
forward realised volatility -- a quantity with no window overlap with the
features at all -- beats the same style of benchmark by a wide and highly
significant margin.

Those two facts together suggest the binding constraint is the *formulation*,
not the feature set or the estimator.  A binary state indicator throws away
almost everything the features know: whichever side of the boundary the market
is on, the label is the same, and the boundary is where all the variance is.

Three formulations of the same question are compared at every horizon:

R1  ``binary``      classify the risk-off indicator at t+h directly.
R2  ``continuous``  regress the *risk-appetite score* at t+h, then read the
                    probability off the predictive distribution.  Same
                    features, same horizon, same event -- only the loss the
                    estimator is trained on changes.
R3  ``vol_state``   classify whether forward realised volatility exceeds its
                    training median.  A different, model-free event.

R3 is also run across every horizon so the application can show its
out-of-sample skill beside the regime forecast.

Output: reports/06_target_reformulation.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regimelab.config import ARTIFACT_DIR, FORECAST_HORIZONS, REPORT_DIR
from regimelab.data import build_panel, download_all
from regimelab.evaluation import summarise_predictions, walk_forward_folds
from regimelab.features import build_features, label_feature_frame
from regimelab.forecasting import (
    COMPACT_COLUMNS,
    build_designs,
    point_in_time_target,
    run_walk_forward,
)
from regimelab.labeling import RegimeLabeler, risk_appetite_score

OUT: list[str] = []
N_REGIMES = 4


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def continuous_reformulation(
    features: pd.DataFrame, horizon: int, n_regimes: int = N_REGIMES,
    initial_train: int = 2000, test_block: int = 252, alphas=(1.0, 10.0, 100.0),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """R2: regress the forward risk score, then convert to a probability.

    The event is defined identically to R1 -- the risk-off indicator implied by
    the same train-fitted labeller -- so the two are directly comparable.  The
    probability comes from a Gaussian predictive distribution whose scale is
    the training residual standard deviation, which is the simplest honest way
    to turn a point forecast into a calibrated-ish probability.
    """
    L = label_feature_frame(features)
    folds = walk_forward_folds(len(features), horizon, initial_train, test_block)

    ys, preds = [], {f"continuous_ridge_a{a:g}": [] for a in alphas}
    preds["binary_reference"] = []

    for fold in folds:
        tr = slice(fold.train_start, fold.train_end)
        labeler = RegimeLabeler(n_regimes=n_regimes, method="hmm",
                                init="score", random_state=42).fit(L.iloc[tr])
        filt = labeler.filtered_proba(L)
        hard = filt.values.argmax(axis=1)
        mask = labeler.risk_off_mask()
        risk_off = mask[hard].astype(float)

        score = risk_appetite_score(
            labeler.scaler_.transform(L.values), list(L.columns))
        y_bin_all = point_in_time_target(risk_off, horizon)
        y_sc_all = np.full(len(score), np.nan)
        y_sc_all[: len(score) - horizon] = score[horizon:]

        tr_idx = np.arange(fold.train_start, fold.train_end)
        te_idx = np.arange(fold.test_start, fold.test_end)
        ok = np.isfinite(y_bin_all) & np.isfinite(y_sc_all)
        tr_idx, te_idx = tr_idx[ok[tr_idx]], te_idx[ok[te_idx]]
        if len(te_idx) == 0 or len(np.unique(y_bin_all[tr_idx])) < 2:
            continue

        # Threshold on the forward score whose exceedance frequency on the
        # training fold matches the risk-off base rate, so "score below cut"
        # and "risk-off" describe the same event at the same frequency.
        off_tr = y_bin_all[tr_idx].astype(bool)
        if off_tr.all() or not off_tr.any():
            continue
        cut = float(np.quantile(y_sc_all[tr_idx], off_tr.mean()))

        cols = [c for c in COMPACT_COLUMNS if c in features.columns]
        sp = filt.copy()
        sp.columns = [f"state_p{c}" for c in sp.columns]
        X = pd.concat([features[cols], sp], axis=1).values

        for a in alphas:
            m = make_pipeline(StandardScaler(), Ridge(alpha=a))
            m.fit(X[tr_idx], y_sc_all[tr_idx])
            resid = y_sc_all[tr_idx] - m.predict(X[tr_idx])
            sd = max(float(resid.std(ddof=1)), 1e-6)
            mu = m.predict(X[te_idx])
            preds[f"continuous_ridge_a{a:g}"].append(
                np.clip(norm.cdf((cut - mu) / sd), 1e-4, 1 - 1e-4))

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.02, max_iter=3000))
        clf.fit(X[tr_idx], y_bin_all[tr_idx].astype(int))
        preds["binary_reference"].append(
            np.clip(clf.predict_proba(X[te_idx])[:, 1], 1e-4, 1 - 1e-4))

        ys.append(y_bin_all[te_idx])

    return np.concatenate(ys), {k: np.concatenate(v) for k, v in preds.items() if v}


def main() -> None:
    say("# Experiment 6 - Reformulating the target")
    say()

    panel = build_panel(download_all(), sample="core_1990")
    feat = build_features(panel)

    # ------------------------------------------------------------------
    say("## R3. Forward volatility state, across all horizons")
    say()
    say("Target: realised S&P volatility over `(t, t+h]` above its "
        "training-fold median. Benchmark unchanged -- the regime-conditional "
        "frequency, i.e. today's state and nothing else.")
    say()
    rows = []
    for h in FORECAST_HORIZONS:
        print(f"  [mkt_high_vol] h={h}")
        res = run_walk_forward(feat, panel, horizon=h,
                               target_kind="mkt_high_vol",
                               n_regimes=N_REGIMES, labeler_method="hmm",
                               initial_train=2000, test_block=252, verbose=True)
        tbl = summarise_predictions(res.y, res.preds, h, baseline="persist_markov")
        tbl.insert(0, "h", h)
        tbl.insert(1, "target", "mkt_high_vol")
        rows.append(tbl)
    vol = pd.concat(rows, ignore_index=True)
    vol.to_csv(ARTIFACT_DIR / "06_volatility_benchmark.csv", index=False)

    best = (vol[vol["model"] != "persist_markov"].sort_values("log_loss")
            .groupby("h", as_index=False).first())
    say(best[["h", "model", "log_loss", "auc", "skill_vs_base", "dm_t", "dm_p"]]
        .round(4).to_markdown(index=False))
    say()
    say("Full table:")
    say()
    say(vol.pivot_table(index="model", columns="h", values="skill_vs_base")
        .round(3).to_markdown())
    say()

    # ------------------------------------------------------------------
    say("## R1 vs R2. Same event, different loss")
    say()
    rows = []
    for h in FORECAST_HORIZONS:
        print(f"  [reformulation] h={h}")
        y, preds = continuous_reformulation(feat, h)
        tbl = summarise_predictions(y, preds, h, baseline="binary_reference")
        tbl.insert(0, "h", h)
        rows.append(tbl)
    ref = pd.concat(rows, ignore_index=True)
    ref.to_csv(ARTIFACT_DIR / "06_reformulation.csv", index=False)
    say(ref.pivot_table(index="model", columns="h", values="log_loss")
        .round(4).to_markdown())
    say()
    say("Log loss, lower is better. `binary_reference` is the regularised "
        "logistic classifier from experiment 3 on the same design matrix; the "
        "`continuous_ridge` rows regress the forward risk-appetite score and "
        "convert to a probability through a Gaussian predictive distribution.")
    say()
    say(ref.pivot_table(index="model", columns="h", values="skill_vs_base")
        .round(4).to_markdown())
    say()
    say("Skill relative to the binary classifier.")
    say()

    path = REPORT_DIR / "06_target_reformulation.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
