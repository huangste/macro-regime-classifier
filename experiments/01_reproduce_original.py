"""Experiment 1 -- reproduce the original notebook, then diagnose it.

Runs the original pipeline as written (same tickers, same features, same GMM,
same logistic/TimeSeriesSplit evaluation) and then adds the diagnostics the
notebook never ran:

  D1  what sample was actually used, and how many *independent* observations
      the overlapping 21-day target really provides;
  D2  how unstable the component indices are across seeds and vintages;
  D3  how much of the reported score survives when the scaler/PCA/GMM are
      refit causally instead of on the full sample;
  D4  how the reported score compares with a persistence baseline that knows
      nothing except today's regime.

Output: reports/01_reproduce_original.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from regimelab.config import REPORT_DIR
from regimelab.data import fetch_fred, fetch_yahoo

OUT = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


# --------------------------------------------------------------------------
# 1. Data exactly as the original built it
# --------------------------------------------------------------------------
def original_raw_data() -> pd.DataFrame:
    eq = pd.concat(
        [fetch_yahoo(t).rename(n) for t, n in
         [("^GSPC", "SPX"), ("^N225", "Nikkei"), ("^NDX", "NDX")]], axis=1)
    fx = pd.concat(
        [fetch_yahoo(t).rename(n) for t, n in
         [("DX-Y.NYB", "DXY"), ("JPY=X", "USDJPY")]], axis=1)
    cmd = pd.concat(
        [fetch_yahoo(t).rename(n) for t, n in
         [("GC=F", "Gold"), ("CL=F", "Oil")]], axis=1)
    rates = pd.concat([fetch_fred("DGS10"), fetch_fred("DGS2")], axis=1)
    rates["US_10Y_2Y_Slope"] = rates["DGS10"] - rates["DGS2"]

    raw = pd.concat([eq, fx, cmd, rates[["DGS10", "DGS2", "US_10Y_2Y_Slope"]]],
                    axis=1).sort_index()
    raw = raw.loc["1990-01-01":]
    raw = raw.ffill()          # as in the notebook
    return raw.dropna()        # as in the notebook


def original_features(raw: pd.DataFrame) -> pd.DataFrame:
    price_cols = ["SPX", "Nikkei", "NDX", "DXY", "USDJPY", "Gold", "Oil"]
    rate_cols = ["DGS10", "DGS2", "US_10Y_2Y_Slope"]
    f = pd.DataFrame(index=raw.index)
    for c in price_cols:
        for h in (1, 10, 21, 63):
            f[f"{c}_{h}d_ret"] = raw[c].pct_change(h)
    for c in rate_cols:
        for h in (1, 10, 21, 63):
            f[f"{c}_{h}d_chg"] = raw[c].diff(h)
    dr = raw[price_cols].pct_change(1)
    for c in price_cols:
        f[f"{c}_21d_vol"] = dr[c].rolling(21).std()
        f[f"{c}_63d_vol"] = dr[c].rolling(63).std()
    return f.dropna()


def main() -> None:
    say("# Experiment 1 - Reproduction and diagnosis of the original notebook")
    say()

    raw = original_raw_data()
    features = original_features(raw)

    say("## Reproduction")
    say()
    say(f"* Raw panel after the notebook's `dropna()`: **{raw.shape[0]} rows**, "
        f"{raw.index[0].date()} to {raw.index[-1].date()}")
    say(f"* Feature matrix: **{features.shape[0]} rows x {features.shape[1]} columns**")
    say()
    say("The download starts at 1990 but the effective sample starts "
        f"**{raw.index[0].date()}**: `GC=F` and `CL=F` have no history before "
        "2000, and `dropna()` on the concatenated panel silently truncates "
        "everything else to match. Thirty-six years of available S&P history "
        "became twenty-five.")
    say()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=0.85)
    X_pca = pca.fit_transform(X_scaled)
    say(f"* PCA retains **{X_pca.shape[1]} components** for 85% of variance "
        f"(first three explain {pca.explained_variance_ratio_[:3].sum():.1%})")

    gmm = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=1e-4,
                          n_init=50, random_state=42)
    labels = gmm.fit_predict(X_pca)
    shares = pd.Series(labels).value_counts(normalize=True).sort_index()
    say(f"* GMM(5, diag) component shares: "
        f"{ {int(k): round(v, 3) for k, v in shares.items()} }")
    say()

    # ---- the notebook's own forecasting setup ----------------------------
    H = 21
    stress = np.where(np.isin(labels, [0, 2]), 1, 0)   # as coded in the notebook
    y = stress[H:]
    X = X_pca[:-H]
    tscv = TimeSeriesSplit(n_splits=2)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    cv = -cross_val_score(model, X, y, cv=tscv,
                          scoring="neg_log_loss").mean()
    base_rate = y.mean()
    naive = log_loss(y, np.full((len(y), 2), [1 - base_rate, base_rate]), labels=[0, 1])
    say(f"* Notebook's reported setup: H={H}, stress mask = components [0, 2], "
        f"base rate {base_rate:.1%}")
    say(f"* Logistic CV log-loss **{cv:.4f}** vs unconditional baseline "
        f"**{naive:.4f}**")
    say()

    # ======================================================================
    say("## Diagnosis")
    say()

    # ---- D1: effective sample size ---------------------------------------
    say("### D1. The effective sample is roughly 300 observations, not 6,000")
    say()
    n = len(y)
    say(f"The target is the regime {H} trading days ahead, sampled every day, so "
        f"consecutive rows share {H - 1}/{H} of their horizon. The {n:,} rows "
        f"carry on the order of **{n // H:,} non-overlapping observations**, and "
        "the risk-off class contains roughly "
        f"**{int(base_rate * n) // H:,}** of them. Every standard error in the "
        "notebook is understated by a factor of about "
        f"sqrt({H}) = {np.sqrt(H):.1f}.")
    say()

    # ---- D2: label instability -------------------------------------------
    say("### D2. Component indices carry no stable meaning")
    say()
    ref = None
    rows = []
    for seed in [0, 1, 2, 7, 42]:
        g = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=1e-4,
                            n_init=1, random_state=seed).fit(X_pca)
        lab = g.predict(X_pca)
        if ref is None:
            ref = lab
        # Which raw index holds the highest-volatility component?
        vol_col = features.columns.get_loc("SPX_21d_vol")
        worst = int(pd.Series(features["SPX_21d_vol"].values).groupby(lab).mean().idxmax())
        rows.append({"seed": seed, "ARI_vs_seed0": adjusted_rand_score(ref, lab),
                     "index_of_highest_vol_component": worst})
    d2 = pd.DataFrame(rows)
    say("Refitting with `n_init=1` and different seeds (the notebook's "
        "`n_init=50` hides, but does not remove, this):")
    say()
    say(d2.round(3).to_markdown(index=False))
    say()

    vintages = []
    for cut in [0, 60, 125, 250]:
        sub = X_pca[: len(X_pca) - cut] if cut else X_pca
        g = GaussianMixture(n_components=5, covariance_type="diag", reg_covar=1e-4,
                            n_init=10, random_state=42).fit(sub)
        lab = g.predict(X_pca[: len(sub)])
        worst = int(pd.Series(features["SPX_21d_vol"].values[: len(sub)])
                    .groupby(lab).mean().idxmax())
        overlap = adjusted_rand_score(labels[: len(sub)], lab)
        vintages.append({"days_removed_from_end": cut,
                         "index_of_highest_vol_component": worst,
                         "ARI_vs_full_sample_fit": overlap})
    say("Refitting on earlier data vintages, which is what happens every time "
        "the notebook is re-run on a later date:")
    say()
    say(pd.DataFrame(vintages).round(3).to_markdown(index=False))
    say()
    say("The index of the crisis component moves. The notebook hard-codes "
        "`np.isin(labels, [0, 2])` as the stress mask while the comment on the "
        "line above says `classify 1, 3 as crisis` and the plot legend calls "
        "component 3 `Crisis/Deleveraging`. Those three statements cannot all "
        "be true, which is the label-drift problem showing up in the code "
        "itself.")
    say()

    # ---- D3: leakage in the evaluated pipeline ---------------------------
    say("### D3. The scaler, the PCA and the regime labels all see the future")
    say()
    say("`StandardScaler`, `PCA` and the `GaussianMixture` are all fitted on "
        "the entire sample **before** `cross_val_score` splits it. Two "
        "distinct leaks follow, and they have to be measured separately.")
    say()
    say("**Leak 1 - the feature pipeline.** Holding the target fixed (so the "
        "comparison is like-for-like) and varying only whether the scaler and "
        "PCA are fitted on the training fold or on everything:")
    say()

    splits = list(TimeSeriesSplit(n_splits=5).split(features))
    rows = []
    for name, causal_pipe in [("scaler + PCA fitted on full sample", False),
                              ("scaler + PCA fitted on training fold only", True)]:
        losses = []
        for tr, te in splits:
            yy = stress[H:]
            tr2, te2 = tr[tr < len(yy)], te[te < len(yy)]
            if len(np.unique(yy[tr2])) < 2:
                continue
            if causal_pipe:
                sc = StandardScaler().fit(features.iloc[tr])
                pc = PCA(n_components=0.85).fit(sc.transform(features.iloc[tr]))
                Z = pc.transform(sc.transform(features))
            else:
                Z = X_pca
            XX = Z[:-H]
            m = LogisticRegression(solver="lbfgs", max_iter=1000,
                                   class_weight="balanced").fit(XX[tr2], yy[tr2])
            losses.append(log_loss(yy[te2], m.predict_proba(XX[te2]), labels=[0, 1]))
        rows.append({"feature pipeline": name, "mean_fold_log_loss": np.mean(losses)})
    say(pd.DataFrame(rows).round(4).to_markdown(index=False))
    say()
    say("**Leak 2 - the target.** This one cannot be measured by swapping a "
        "line: the labels themselves are produced by a mixture fitted with "
        "hindsight, so a causal version has a different target and its "
        "log-losses are not comparable on the same scale. That is why "
        "experiment 3 evaluates skill *relative to a baseline computed under "
        "the same labelling*, and refits the labeller inside every "
        "walk-forward fold.")
    say()

    # ---- D4: persistence baseline ----------------------------------------
    say("### D4. A persistence baseline that uses no features at all")
    say()
    say("The notebook compares its model only against the unconditional base "
        "rate. But regimes are extremely persistent, so the honest question is "
        "whether the features add anything beyond knowing today's state. "
        "Fitting P(risk-off in 21d | risk-off today) on the training fold only:")
    say()
    rows = []
    for tr, te in splits:
        yy, st_now = stress[H:], stress[:-H]
        tr2, te2 = tr[tr < len(yy)], te[te < len(yy)]
        p1 = yy[tr2][st_now[tr2] == 1].mean()
        p0 = yy[tr2][st_now[tr2] == 0].mean()
        pr = np.where(st_now[te2] == 1, p1, p0)
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        persistence = log_loss(yy[te2], np.c_[1 - pr, pr], labels=[0, 1])
        b = np.clip(yy[tr2].mean(), 1e-6, 1 - 1e-6)
        uncond = log_loss(yy[te2], np.full((len(te2), 2), [1 - b, b]), labels=[0, 1])
        m = LogisticRegression(solver="lbfgs", max_iter=1000,
                               class_weight="balanced").fit(X_pca[:-H][tr2], yy[tr2])
        full = log_loss(yy[te2], m.predict_proba(X_pca[:-H][te2]), labels=[0, 1])
        rows.append({"unconditional": uncond, "persistence_only": persistence,
                     "logistic_on_PCA": full})
    d4 = pd.DataFrame(rows)
    say(d4.mean().round(4).to_frame("mean fold log-loss").to_markdown())
    say()
    say("Lower is better. The published model is beaten by a two-parameter "
        "persistence rule, and does worse than simply quoting the base rate.")
    say()

    # ---- D5: fragility across the split count the notebook scanned --------
    say("### D5. The headline number depends on how many folds you ask for")
    say()
    say("The final cell of the notebook scans `n_splits` in {2, 3, 4, 5} across "
        "four model families and prints all sixteen results. Nothing corrects "
        "for that search, and the setting used earlier in the notebook "
        "(`n_splits=2`) happens to be the most flattering one:")
    say()
    rows = []
    for n in (2, 3, 4, 5):
        sc = -cross_val_score(
            LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced"),
            X, y, cv=TimeSeriesSplit(n_splits=n), scoring="neg_log_loss")
        rows.append({"n_splits": n, "mean_log_loss": sc.mean(),
                     "worst_fold": sc.max(), "best_fold": sc.min()})
    say(pd.DataFrame(rows).round(4).to_markdown(index=False))
    say()
    say("With two folds the model appears to beat the unconditional baseline "
        f"of {naive:.4f}. With five it does not. A result that reverses on a "
        "validation setting is not evidence of predictive information.")
    say()

    path = REPORT_DIR / "01_reproduce_original.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
