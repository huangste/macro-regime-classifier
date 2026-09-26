"""Experiment 2 -- does economic anchoring actually stabilise the labels?

Three failure modes are measured separately, because they have different
causes and different fixes:

S1  *Permutation instability.*  Same data, different seed: do the component
    indices mean the same thing?  Anchoring should fix this outright.

S2  *Vintage instability.*  Refit on data up to T for a sequence of T, which
    is what happens whenever the model is re-run on a later date.  Anchoring
    fixes the naming; it cannot fix genuine changes in the partition, so this
    measures how much of the problem was naming and how much was substance.

S3  *Flicker.*  How often the historical classification switches state.  A
    label that changes twice a week is not a regime.

Output: reports/02_label_stability.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from regimelab.config import REPORT_DIR
from regimelab.data import build_panel, download_all
from regimelab.features import build_features, label_feature_frame
from regimelab.labeling import RegimeLabeler, episode_table, regime_characteristics

OUT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


VARIANTS = {
    # the original notebook's setup: mixture indices used as-is
    "gmm_raw_index": dict(method="gmm", anchored=False, init="random"),
    "gmm_anchored": dict(method="gmm", anchored=True, init="random"),
    "hmm_anchored": dict(method="hmm", anchored=True, init="random"),
    # deterministic score initialisation
    "gmm_anchored_scoreinit": dict(method="gmm", anchored=True, init="score"),
    "hmm_anchored_scoreinit": dict(method="hmm", anchored=True, init="score"),
    "score_only": dict(method="score", anchored=True, init="score"),
}


def fit_variant(L: pd.DataFrame, spec: dict, k: int, seed: int, n_init: int = 10):
    return RegimeLabeler(n_regimes=k, method=spec["method"], init=spec["init"],
                         random_state=seed, n_init=n_init).fit(L)


def labels_of(lab: RegimeLabeler, L: pd.DataFrame, anchored: bool) -> np.ndarray:
    y = lab.predict(L).values
    if anchored:
        return y
    # Undo the canonical re-indexing to recover the estimator's own numbering.
    return lab.inverse_order_[y]


def binary_of(lab: RegimeLabeler, L: pd.DataFrame, anchored: bool) -> np.ndarray:
    y = lab.predict(L)
    if anchored:
        return lab.binary_labels(y).values
    # What the original notebook did: a hard-coded index list, frozen from
    # whichever fit happened to be in front of the analyst that day.
    frozen = {0, 2}
    return np.isin(lab.inverse_order_[y.values], list(frozen)).astype(int)


def main() -> None:
    say("# Experiment 2 - Label stability")
    say()

    panel = build_panel(download_all(), sample="core_1990")
    feat = build_features(panel)
    L = label_feature_frame(feat)
    K = 4
    say(f"Sample: {L.index[0].date()} to {L.index[-1].date()}, {len(L):,} days, "
        f"{L.shape[1]} labelling features, K={K}.")
    say()

    # ------------------------------------------------------------------
    say("## S1. Same data, different seed")
    say()
    seeds = [0, 1, 2, 7, 13, 42]
    rows = []
    for name, spec in VARIANTS.items():
        fits = [fit_variant(L, spec, K, s, n_init=1) for s in seeds]
        ref_lab = labels_of(fits[0], L, spec["anchored"])
        ref_bin = binary_of(fits[0], L, spec["anchored"])
        aris, flips = [], []
        for f in fits[1:]:
            aris.append(adjusted_rand_score(ref_lab, labels_of(f, L, spec["anchored"])))
            flips.append(float(np.mean(binary_of(f, L, spec["anchored"]) != ref_bin)))
        rows.append({
            "variant": name,
            "mean_ARI_vs_seed0": np.mean(aris),
            "min_ARI": np.min(aris),
            "risk_on_off_flip_rate": np.mean(flips),
        })
    s1 = pd.DataFrame(rows)
    say(s1.round(3).to_markdown(index=False))
    say()
    say("`ARI` compares partitions and is permutation-invariant by "
        "construction, so it is the same for raw and anchored variants of the "
        "same estimator -- that is the point. The column that matters is the "
        "flip rate: the share of historical days whose **risk-on/risk-off** "
        "verdict changes when only the random seed changes.")
    say()

    # ------------------------------------------------------------------
    say("## S2. Refitting on later data vintages")
    say()
    cuts = [len(L) - c for c in (1260, 1008, 756, 504, 252, 0)]
    rows = []
    for name, spec in VARIANTS.items():
        full = fit_variant(L, spec, K, 42, n_init=10)
        full_lab = labels_of(full, L, spec["anchored"])
        full_bin = binary_of(full, L, spec["anchored"])
        aris, flips = [], []
        for c in cuts[:-1]:
            v = fit_variant(L.iloc[:c], spec, K, 42, n_init=10)
            lab_v = labels_of(v, L.iloc[:c], spec["anchored"])
            bin_v = binary_of(v, L.iloc[:c], spec["anchored"])
            aris.append(adjusted_rand_score(full_lab[:c], lab_v))
            flips.append(float(np.mean(bin_v != full_bin[:c])))
        rows.append({
            "variant": name,
            "mean_ARI_vs_latest": np.mean(aris),
            "min_ARI": np.min(aris),
            "risk_on_off_flip_rate": np.mean(flips),
            "worst_flip_rate": np.max(flips),
        })
    s2 = pd.DataFrame(rows)
    say("Vintages cut 1, 2, 3, 4 and 5 years off the end and compare the "
        "historical verdict on the overlapping dates:")
    say()
    say(s2.round(3).to_markdown(index=False))
    say()

    # ------------------------------------------------------------------
    say("## S3. Flicker in the historical classification")
    say()
    rows = []
    for name, spec in VARIANTS.items():
        lab = fit_variant(L, spec, K, 42, n_init=10)
        y = lab.predict(L)
        switches = int((y.values[1:] != y.values[:-1]).sum())
        years = (L.index[-1] - L.index[0]).days / 365.25
        ep = episode_table(y, min_days=1)
        rows.append({
            "variant": name,
            "switches_per_year": switches / years,
            "median_episode_days": float(ep["days"].median()),
            "episodes_shorter_than_5d": float((ep["days"] < 5).mean()),
        })
    say(pd.DataFrame(rows).round(2).to_markdown(index=False))
    say()

    # ------------------------------------------------------------------
    say("## Choosing K")
    say()
    rows = []
    for k in (2, 3, 4, 5, 6):
        gm = RegimeLabeler(n_regimes=k, method="gmm", init="random",
                           random_state=42, n_init=10).fit(L)
        bic = gm.model_.bic(gm.scaler_.transform(L.values))
        hm = RegimeLabeler(n_regimes=k, method="hmm", init="score",
                           random_state=42).fit(L)
        y = hm.predict(L)
        # vintage robustness at this K
        flips = []
        for c in (len(L) - 1260, len(L) - 504):
            v = RegimeLabeler(n_regimes=k, method="hmm", init="score",
                              random_state=42).fit(L.iloc[:c])
            flips.append(float(np.mean(
                v.binary_labels(v.predict(L.iloc[:c])).values
                != hm.binary_labels(y).values[:c])))
        chars = regime_characteristics(panel, y, k)
        rows.append({
            "K": k,
            "GMM_BIC": bic,
            "risk_off_states": int(hm.risk_off_mask().sum()),
            "risk_off_share_of_days": float(hm.binary_labels(y).mean()),
            "vintage_flip_rate": float(np.mean(flips)),
            "sharpe_spread_worst_to_best": float(
                chars["SPX_sharpe"].iloc[-1] - chars["SPX_sharpe"].iloc[0]),
        })
    say(pd.DataFrame(rows).round(3).to_markdown(index=False))
    say()
    say("BIC falls monotonically with K -- it always does on strongly "
        "autocorrelated data, because extra components soak up serial "
        "dependence the model does not otherwise represent. It is therefore "
        "not a usable criterion here. K=4 is chosen instead on vintage "
        "stability plus the requirement that the states separate realised "
        "risk-adjusted returns, while still leaving enough days in each state "
        "to estimate transitions.")
    say()

    # ------------------------------------------------------------------
    say("## Economic characteristics of the chosen labelling (HMM, K=4)")
    say()
    lab = RegimeLabeler(n_regimes=K, method="hmm", init="score", random_state=42).fit(L)
    y = lab.predict(L)
    chars = regime_characteristics(panel, y, K)
    chars.insert(0, "name", lab.names())
    chars.insert(1, "risk_score", lab.regime_scores_.round(3))
    say(chars.round(3).to_markdown())
    say()
    say("Canonical index 0 is always the most risk-averse state and index "
        f"{K - 1} the most risk-seeking, by construction, whatever the seed or "
        "the data vintage.")
    say()

    ep = episode_table(y, min_days=15)
    worst = ep[ep["regime"] == 0].sort_values("days", ascending=False).head(12)
    say("Longest episodes of canonical regime 0 (crisis / deleveraging):")
    say()
    say(worst.assign(start=lambda d: d["start"].dt.date,
                     end=lambda d: d["end"].dt.date)[["start", "end", "days"]]
        .to_markdown(index=False))
    say()

    path = REPORT_DIR / "02_label_stability.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
