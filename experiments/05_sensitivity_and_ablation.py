"""Experiment 5 -- how fragile is the conclusion, and what carries it?

A result that only holds at one K, one labeller and one sample is not a
result.  This runs the same purged walk-forward while varying, one at a time:

A  the number of regimes K;
B  the labelling estimator (HMM vs the original notebook's Gaussian mixture);
C  the sample (1990 start vs the original's 2000 start);
D  class weighting, which the original used and which should hurt a
   probability forecast even though it helps a classification decision;
E  which feature blocks the forecaster is allowed to see.

Everything is measured at h=25 on the `any_in_window` target, which is the
horizon and definition most relevant to the application.

Output: reports/05_sensitivity_and_ablation.md
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
from regimelab.evaluation import summarise_predictions
from regimelab.features import build_features
from regimelab.forecasting import run_walk_forward

OUT: list[str] = []
H = 25
TARGET = "any_in_window"
COMBOS = (("state", "logit"), ("compact", "logit_reg"), ("compact", "hgb"))
COMBOS_CW = COMBOS + (("compact", "logit_balanced"),)


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "volatility": ("VIX_log", "VIX_vrp", "SPX_logvol21", "SPX_volratio"),
    "credit": ("Baa_log", "Baa_chg21", "Baa_chg63",
               "Quality_level", "Quality_chg21"),
    "momentum": ("SPX_rmom5", "SPX_rmom21", "SPX_rmom63", "SPX_dd252",
                 "RUT_rel_SPX_63", "NDX_rel_SPX_63"),
    "rates": ("DGS10_chg21", "DGS10_chg63", "DGS2_chg21", "DGS2_chg63",
              "Slope_10Y2Y_level", "Slope_10Y2Y_chg63",
              "Slope_10Y3M_level", "Slope_10Y3M_chg63", "SPX_bond_corr63"),
}


def run(feat, panel, label: str, **kw) -> pd.DataFrame:
    print(f"  {label}")
    res = run_walk_forward(
        feat, panel, horizon=H, target_kind=TARGET,
        initial_train=2000, test_block=252, verbose=True,
        **kw)
    tbl = summarise_predictions(res.y, res.preds, H, baseline="persist_markov")
    tbl.insert(0, "arm", label)
    tbl.insert(1, "base_rate", tbl.attrs["base_rate"])
    return tbl


def skill_table(tbl: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    d = tbl[tbl["model"].isin(models)]
    out = d.pivot_table(index="arm", columns="model",
                        values="skill_vs_base", sort=False)
    rates = d.groupby("arm", sort=False)["base_rate"].first()
    out.insert(0, "base_rate", rates)
    return out


def main() -> None:
    say("# Experiment 5 - Sensitivity and ablation")
    say()
    say(f"All arms: horizon {H} days, target `{TARGET}`, expanding "
        "walk-forward purged by the horizon, benchmark `persist_markov`. "
        "Cells are **skill versus that benchmark** -- positive means the "
        "method removed some of the benchmark's log loss.")
    say()

    panel90 = build_panel(download_all(), sample="core_1990")
    feat90 = build_features(panel90)

    models = ["state|logit", "compact|logit_reg", "compact|hgb",
              "hmm_analytic", "uncond"]
    arms: list[pd.DataFrame] = []

    # ---- A: number of regimes ----------------------------------------
    say("## A. Number of regimes")
    say()
    for k in (2, 3, 4, 5):
        arms.append(run(feat90, panel90, f"K={k}", n_regimes=k,
                        labeler_method="hmm", combos=COMBOS))
    a = pd.concat(arms, ignore_index=True)
    say(skill_table(a, models).round(4).to_markdown())
    say()
    say("Base rates differ by K, so compare skill columns, not log losses.")
    say()

    # ---- B: labelling estimator --------------------------------------
    say("## B. Labelling estimator")
    say()
    barms = [a[a["arm"] == "K=4"].assign(arm="hmm (K=4)")]
    barms.append(run(feat90, panel90, "gmm (K=4)", n_regimes=4,
                     labeler_method="gmm", combos=COMBOS))
    b = pd.concat(barms, ignore_index=True)
    say(skill_table(b, [m for m in models if m != "hmm_analytic"]).round(4).to_markdown())
    say()
    say("The Gaussian mixture has no time dimension, so its state estimate "
        "flickers and its transition matrix is estimated from a much noisier "
        "label series. `hmm_analytic` has no mixture counterpart and is "
        "omitted from this table.")
    say()

    # ---- C: sample ---------------------------------------------------
    say("## C. Sample period")
    say()
    panel00 = build_panel(download_all(), sample="full_2000")
    feat00 = build_features(panel00)
    carms = [a[a["arm"] == "K=4"].assign(arm="core_1990 (1990-)")]
    carms.append(run(feat00, panel00, "full_2000 (2000-, incl. gold)",
                     n_regimes=4, labeler_method="hmm", combos=COMBOS))
    c = pd.concat(carms, ignore_index=True)
    say(skill_table(c, models).round(4).to_markdown())
    say()

    # ---- D: class weighting ------------------------------------------
    say("## D. Class weighting")
    say()
    d = run(feat90, panel90, "class-weight comparison", n_regimes=4,
            labeler_method="hmm", combos=COMBOS_CW)
    show = d[d["model"].isin(["compact|logit_reg", "compact|logit_balanced",
                              "persist_markov"])]
    say(show[["model", "log_loss", "brier", "auc", "skill_vs_base"]]
        .round(4).to_markdown(index=False))
    say()
    say("`class_weight='balanced'` is what the original notebook used. It "
        "leaves ranking (AUC) roughly intact while badly damaging log loss "
        "and Brier score, because re-weighting the classes deliberately "
        "biases the predicted probabilities away from the base rate. If the "
        "output is a probability rather than a decision, it should not be on.")
    say()

    # ---- E: feature blocks -------------------------------------------
    say("## E. Which features carry the signal?")
    say()
    earms = []
    for name, cols in FEATURE_BLOCKS.items():
        keep = [c for c in cols if c in feat90.columns]
        earms.append(run(feat90, panel90, f"only {name}", n_regimes=4,
                         labeler_method="hmm", combos=COMBOS,
                         design_features=feat90[keep]))
    for name, cols in FEATURE_BLOCKS.items():
        drop = set(cols)
        keep = [c for c in feat90.columns if c not in drop]
        earms.append(run(feat90, panel90, f"all except {name}", n_regimes=4,
                         labeler_method="hmm", combos=COMBOS,
                         design_features=feat90[keep]))
    e = pd.concat([a[a["arm"] == "K=4"].assign(arm="all features")] + earms,
                  ignore_index=True)
    say(skill_table(e, ["compact|logit_reg", "compact|hgb"]).round(4).to_markdown())
    say()
    say("The labeller keeps its full feature set in every arm, so the regime "
        "definition is identical across rows and only the forecaster's inputs "
        "change.")
    say()

    everything = pd.concat([a, b, c, d, e], ignore_index=True)
    everything.to_csv(ARTIFACT_DIR / "05_sensitivity.csv", index=False)

    path = REPORT_DIR / "05_sensitivity_and_ablation.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
