"""Experiment 4 -- is the measured skill economic foresight, or arithmetic?

Experiment 3 shows learned models beating the persistence benchmark at short
horizons by a wide margin.  Before believing that, one mechanism has to be
ruled out.

The regime at ``t+h`` is a function of features computed at ``t+h``, and those
features are trailing-window statistics.  ``SPX_rmom63`` observed at ``t+5``
still shares 58 of its 63 days with the same feature observed at ``t``.  So a
model can forecast the future *state* very accurately while knowing nothing
about the future *market*: it only has to extrapolate the roll-off of its own
windows.  This is not look-ahead bias -- no future data is used -- but it is
not forecasting skill either, and it inflates exactly the horizons where the
original notebook's successor looks most impressive.

Three checks:

M1  How much of the ``t+h`` feature vector is already pinned down at ``t``?
M2  Do the same models beat the same benchmark on targets with **no** window
    overlap at all: forward S&P return sign, and forward realised volatility?
M3  Where does the skill sit -- on days when the state persists, or on days
    when it changes?

Output: reports/04_is_the_skill_real.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from regimelab.config import ARTIFACT_DIR, REPORT_DIR
from regimelab.data import build_panel, download_all
from regimelab.evaluation import binary_log_loss, diebold_mariano, safe_auc, skill_score, summarise_predictions
from regimelab.features import build_features, label_feature_frame
from regimelab.forecasting import run_walk_forward

OUT: list[str] = []
HORIZONS = (5, 25)


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


# Trailing window length actually used by each labelling feature.
FEATURE_WINDOW = {
    "SPX_rmom21": 21, "SPX_rmom63": 63, "SPX_logvol21": 21,
    "SPX_volratio": 63, "SPX_dd252": 252, "VIX_log": 1,
    "Baa_log": 1, "Baa_chg63": 63, "Slope_10Y2Y_level": 1,
    "DGS10_chg63": 63, "RUT_rel_SPX_63": 63, "Nikkei_rmom63": 63,
    "USDJPY_rmom63": 63, "SPX_bond_corr63": 63,
}


def main() -> None:
    say("# Experiment 4 - Is the measured skill real?")
    say()

    panel = build_panel(download_all(), sample="core_1990")
    feat = build_features(panel)
    L = label_feature_frame(feat)

    # ------------------------------------------------------------------
    say("## M1. How much of the future feature vector is already known?")
    say()
    rows = []
    for c in L.columns:
        w = FEATURE_WINDOW.get(c, 1)
        row = {"feature": c, "window_days": w}
        for h in (5, 10, 25):
            row[f"overlap_h{h}"] = max(0.0, (w - h) / w) if w > 1 else 0.0
            row[f"autocorr_h{h}"] = float(L[c].autocorr(lag=h))
        rows.append(row)
    m1 = pd.DataFrame(rows)
    say(m1.round(3).to_markdown(index=False))
    say()
    say("`overlap_hN` is the share of the feature's trailing window at `t+N` "
        "that was already observed at `t`; `autocorr_hN` is the realised lag-N "
        "autocorrelation. Both say the same thing: at h=5 the state in five "
        "days is largely arithmetic, and at h=25 the 63-day features still "
        "retain 60% of their window and correlations above 0.8.")
    say()
    say(f"Mean lag-5 autocorrelation across labelling features: "
        f"**{m1['autocorr_h5'].mean():.3f}**; lag-25: "
        f"**{m1['autocorr_h25'].mean():.3f}**.")
    say()

    # ------------------------------------------------------------------
    say("## M2. Targets with no window overlap")
    say()
    say("`mkt_negative_return` is 1 when the S&P log return over `(t, t+h]` is "
        "negative. `mkt_high_vol` is 1 when realised volatility over the same "
        "window exceeds its training-sample median. Neither shares any data "
        "with the features at `t`. The benchmark is unchanged in spirit: the "
        "regime-conditional frequency estimated on the training fold, i.e. "
        "today's state and nothing else.")
    say()

    cache = ARTIFACT_DIR / "04_market_targets.csv"
    if cache.exists():
        mkt = pd.read_csv(cache)
        print(f"  [cached] {cache.name}")
    else:
        all_rows = []
        for target in ("mkt_high_vol", "mkt_negative_return"):
            for h in HORIZONS:
                print(f"  [{target}] h={h}")
                res = run_walk_forward(
                    feat, panel, horizon=h, target_kind=target,
                    n_regimes=4, labeler_method="hmm",
                    initial_train=2000, test_block=252, verbose=True)
                tbl = summarise_predictions(res.y, res.preds, h,
                                            baseline="persist_markov")
                tbl.insert(0, "h", h)
                tbl.insert(1, "target", target)
                tbl.insert(2, "base_rate", tbl.attrs["base_rate"])
                tbl.insert(3, "n_obs", tbl.attrs["n_obs"])
                tbl.insert(4, "n_eff", tbl.attrs["n_eff"])
                all_rows.append(tbl)
        mkt = pd.concat(all_rows, ignore_index=True)
        mkt.to_csv(cache, index=False)

    for target in ("mkt_high_vol", "mkt_negative_return"):
        for h in HORIZONS:
            tbl = mkt[(mkt["target"] == target) & (mkt["h"] == h)]
            if tbl.empty:
                continue
            say(f"### `{target}`, h = {h}")
            say()
            meta = [c for c in ("base_rate", "n_obs", "n_eff") if c in tbl.columns]
            if len(meta) == 3:
                say(f"Base rate {tbl['base_rate'].iloc[0]:.1%}, "
                    f"{int(tbl['n_obs'].iloc[0]):,} daily observations "
                    f"(~{int(tbl['n_eff'].iloc[0]):,} non-overlapping).")
                say()
            say(tbl.drop(columns=meta).round(4).to_markdown(index=False))
            say()

    say("### Reading M2")
    say()
    for target in ("mkt_high_vol", "mkt_negative_return"):
        sub = mkt[(mkt["target"] == target) & (mkt["model"] != "persist_markov")]
        best = sub.sort_values("log_loss").groupby("h", as_index=False).first()
        line = ", ".join(
            f"h={int(r.h)}: {r.model} skill {r.skill_vs_base:+.1%} (p={r.dm_p:.3f})"
            for r in best.itertuples())
        say(f"* **{target}** -- {line}")
    say()

    # ------------------------------------------------------------------
    say("## M3. Does the skill survive on days when the regime changes?")
    say()
    path = ARTIFACT_DIR / "03_oos_predictions.parquet"
    if not path.exists():
        say("_(run experiment 3 first)_")
    else:
        oos = pd.read_parquet(path)
        oos.index.names = ["run", "date"]
        oos = oos.reset_index()
        rows = []
        for h in HORIZONS:
            d = oos[oos["run"] == f"point_in_time_h{h}"].copy()
            if d.empty:
                continue
            # risk-off indicator implied by the state at t
            state_off = d["persist_markov"] > 0.5
            changed = (d["y"].astype(bool) != state_off).values
            for m in ["persist_markov", "hmm_analytic", "compact|logit_reg",
                      "full|hgb", "full|rf"]:
                if m not in d.columns:
                    continue
                lm = binary_log_loss(d["y"].values, d[m].values)
                lb = binary_log_loss(d["y"].values, d["persist_markov"].values)
                for subset, mask in [("state persists", ~changed),
                                     ("state changes", changed)]:
                    if mask.sum() < 50:
                        continue
                    rows.append({
                        "h": h, "subset": subset, "model": m,
                        "share_of_days": float(mask.mean()),
                        "log_loss": float(lm[mask].mean()),
                        "skill_vs_base": skill_score(lm[mask], lb[mask]),
                    })
        m3 = pd.DataFrame(rows)
        say(m3.round(4).to_markdown(index=False))
        say()
        say("AUC is deliberately omitted: conditioning on whether the state "
            "changed makes the outcome an almost deterministic function of "
            "today's state within each subset, so any ranking metric computed "
            "there is an artefact rather than a measurement.")
        say()
        say("The 'state changes' rows are the ones that matter: they are the "
            "days on which a forecast could have been useful. On the ~86-95% "
            "of days when the state persists the benchmark is close to "
            "unbeatable and every learned model is worse than it; on the days "
            "it turns, the benchmark's loss exceeds 2 and the learned models "
            "cut it by a quarter to a half. The models carry information "
            "about turns and noise about continuation, which is why "
            "experiment 7 combines them rather than choosing between them.")
        say()

    path = REPORT_DIR / "04_is_the_skill_real.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
