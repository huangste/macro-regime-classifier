"""Experiment 3 -- can anything beat persistence out of sample?

Everything is refitted inside each expanding walk-forward fold: the scaler,
the HMM that defines the regimes, the transition matrix, and the classifier.
Training is purged by the forecast horizon so no training label is drawn from
the block being scored.

The benchmark is deliberately hard.  ``persist_markov`` uses the training
transition matrix and today's state and nothing else; every learned model is
given the same state posterior as an input, so it *nests* that benchmark and
any gain it shows is incremental information rather than rediscovered
persistence.

Significance uses a Diebold-Mariano statistic on daily log-loss differentials
with a Newey-West variance at bandwidth 2h, because daily sampling of an
h-day horizon makes those differentials heavily autocorrelated.

Output: reports/03_forecast_benchmark.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from regimelab.config import ARTIFACT_DIR, FORECAST_HORIZONS, REPORT_DIR
from regimelab.data import build_panel, download_all
from regimelab.evaluation import reliability_table, summarise_predictions
from regimelab.features import build_features
from regimelab.forecasting import run_walk_forward

OUT: list[str] = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def main() -> None:
    say("# Experiment 3 - Out-of-sample forecasting benchmark")
    say()

    panel = build_panel(download_all(), sample="core_1990")
    feat = build_features(panel)

    say(f"Sample {feat.index[0].date()} to {feat.index[-1].date()}, "
        f"{len(feat):,} trading days, {feat.shape[1]} features. "
        "Expanding walk-forward, 8-year initial training window, 1-year test "
        "blocks, labeller and models refitted in every fold.")
    say()

    all_rows: list[pd.DataFrame] = []
    store: dict[str, pd.DataFrame] = {}

    for target_kind, blurb in [
        ("point_in_time",
         "P(risk-off **on day t+h**). This decays towards the base rate as h "
         "grows: a single day far in the future is close to a draw from the "
         "stationary distribution."),
        ("any_in_window",
         "P(**at least one** risk-off day in (t, t+h]). This rises with h and "
         "is the quantity a risk manager actually wants."),
    ]:
        say(f"## Target: `{target_kind}`")
        say()
        say(blurb)
        say()

        for h in FORECAST_HORIZONS:
            print(f"  [{target_kind}] h={h}")
            res = run_walk_forward(
                feat, panel, horizon=h, target_kind=target_kind,
                n_regimes=4, labeler_method="hmm",
                initial_train=2000, test_block=252, verbose=True,
            )
            tbl = summarise_predictions(res.y, res.preds, h,
                                        baseline="persist_markov")
            tbl.insert(0, "h", h)
            tbl.insert(1, "target", target_kind)
            all_rows.append(tbl)
            store[f"{target_kind}_h{h}"] = res.to_frame()

            say(f"### h = {h} trading days")
            say()
            say(f"Pooled out-of-sample: {tbl.attrs['n_obs']:,} daily "
                f"observations, about {tbl.attrs['n_eff']:,} non-overlapping, "
                f"base rate {tbl.attrs['base_rate']:.1%}. "
                f"{res.n_folds} folds from "
                f"{res.dates[0].date()} to {res.dates[-1].date()}.")
            say()
            say(tbl.round(4).to_markdown(index=False))
            say()

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(ARTIFACT_DIR / "03_benchmark_full.csv", index=False)
    pd.concat(store.values(), keys=store.keys()).to_parquet(
        ARTIFACT_DIR / "03_oos_predictions.parquet")

    # ------------------------------------------------------------------
    say("## Does anything beat the persistence benchmark?")
    say()
    wins = summary[(summary["model"] != "persist_markov")].copy()
    best = (wins.sort_values("log_loss")
            .groupby(["target", "h"], as_index=False).first())
    say(best[["target", "h", "model", "log_loss", "skill_vs_base", "dm_t", "dm_p"]]
        .round(4).to_markdown(index=False))
    say()
    say("`skill_vs_base` is the fraction of the persistence benchmark's log "
        "loss removed; `dm_t` is negative when the model has lower loss and "
        "`dm_p` is the one-sided p-value. A result is only interesting if the "
        "skill is positive **and** `dm_p` is small.")
    say()

    sig = best[(best["skill_vs_base"] > 0) & (best["dm_p"] < 0.05)]
    if len(sig):
        say(f"{len(sig)} of {len(best)} horizon/target combinations show a "
            "statistically distinguishable improvement over persistence.")
    else:
        say("**No horizon/target combination shows a statistically "
            "distinguishable improvement over persistence.**")
    say()

    # ------------------------------------------------------------------
    say("## Calibration")
    say()
    key = "any_in_window_h25"
    if key in store:
        df = store[key]
        for m in ["persist_markov", "compact|logit"]:
            if m not in df.columns:
                continue
            say(f"`{m}` at h=25, any-in-window:")
            say()
            say(reliability_table(df["y"].values, df[m].values, bins=8)
                .round(3).to_markdown(index=False))
            say()

    path = REPORT_DIR / "03_forecast_benchmark.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
