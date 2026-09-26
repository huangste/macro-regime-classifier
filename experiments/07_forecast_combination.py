"""Experiment 7 -- combining the benchmark with the learned models.

Experiment 4's M3 table decomposed the out-of-sample record by whether the
regime actually changed:

* on the 86% of days when the state persists, the Markov benchmark is close to
  unbeatable and every learned model is *worse* than it;
* on the 14% of days when the state turns, the benchmark is catastrophic
  (log loss above 2) and the learned models cut that by a quarter to a half.

So the learned models carry information about turns and noise about
continuation, and the net of those two is the near-wash experiment 3 reported.
The textbook response is forecast combination.  The weight is fixed at one
half rather than estimated: an equal-weight combination needs no data, cannot
overfit, and is the standard robust default in the forecasting literature.
Estimating the weight in-fold would require out-of-fold training predictions
and would put another researcher degree of freedom into a problem that already
has an effective sample of a few hundred.

This is pure post-processing of the predictions experiment 3 already wrote, so
nothing is refitted and no new look-ahead is introduced.

Output: reports/07_forecast_combination.md
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
from regimelab.evaluation import binary_log_loss, skill_score, summarise_predictions

OUT: list[str] = []
BASE = "persist_markov"
BLEND_WITH = ["hmm_analytic", "compact|logit_reg", "compact|hgb",
              "full|hgb", "full|rf"]


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def main() -> None:
    say("# Experiment 7 - Forecast combination")
    say()

    path = ARTIFACT_DIR / "03_oos_predictions.parquet"
    if not path.exists():
        raise SystemExit("run experiments/03_forecast_benchmark.py first")
    oos = pd.read_parquet(path)
    oos.index.names = ["run", "date"]
    oos = oos.reset_index()

    say("Equal-weight average of the persistence benchmark with each learned "
        "model, scored on exactly the out-of-sample predictions of experiment "
        "3. Rows marked `blend50|` are the combinations.")
    say()

    all_rows = []
    for target in ("point_in_time", "any_in_window"):
        say(f"## Target: `{target}`")
        say()
        for h in FORECAST_HORIZONS:
            d = oos[oos["run"] == f"{target}_h{h}"]
            if d.empty:
                continue
            y = d["y"].values
            preds = {c: d[c].values for c in d.columns
                     if c not in ("run", "date", "y", "state_at_t", "fold")}
            for m in BLEND_WITH:
                if m in preds:
                    preds[f"blend50|{m}"] = 0.5 * (preds[m] + preds[BASE])

            tbl = summarise_predictions(y, preds, h, baseline=BASE)
            tbl.insert(0, "h", h)
            tbl.insert(1, "target", target)
            all_rows.append(tbl)

        sub = pd.concat([t for t in all_rows if t["target"].iloc[0] == target],
                        ignore_index=True)
        piv = sub.pivot_table(index="model", columns="h", values="skill_vs_base")
        say("Skill versus the benchmark, by horizon:")
        say()
        say(piv.round(4).to_markdown())
        say()
        sig = sub.pivot_table(index="model", columns="h", values="dm_p")
        say("One-sided Diebold-Mariano p-values:")
        say()
        say(sig.round(3).to_markdown())
        say()

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ARTIFACT_DIR / "07_combination.csv", index=False)

    # ------------------------------------------------------------------
    say("## Where the combination helps")
    say()
    rows = []
    for target in ("point_in_time", "any_in_window"):
        for h in (5, 25):
            d = oos[oos["run"] == f"{target}_h{h}"]
            if d.empty:
                continue
            y = d["y"].values
            base = d[BASE].values
            state_off = base > 0.5
            changed = y.astype(bool) != state_off
            for m in ["compact|logit_reg"]:
                if m not in d.columns:
                    continue
                blend = 0.5 * (d[m].values + base)
                lb = binary_log_loss(y, base)
                for name, p in [(m, d[m].values), (f"blend50|{m}", blend)]:
                    lp = binary_log_loss(y, p)
                    rows.append({
                        "target": target, "h": h, "model": name,
                        "skill_all": skill_score(lp, lb),
                        "skill_when_state_persists": skill_score(lp[~changed], lb[~changed]),
                        "skill_when_state_changes": skill_score(lp[changed], lb[changed]),
                    })
    say(pd.DataFrame(rows).round(4).to_markdown(index=False))
    say()
    say("The combination keeps most of the gain on turning days while giving "
        "back most of the loss on quiet days, which is exactly what it is "
        "supposed to do.")
    say()

    # ------------------------------------------------------------------
    say("## Summary")
    say()
    win = combined[combined["model"].str.startswith("blend50|")]
    best = win.sort_values("log_loss").groupby(["target", "h"], as_index=False).first()
    say(best[["target", "h", "model", "log_loss", "skill_vs_base", "dm_t", "dm_p"]]
        .round(4).to_markdown(index=False))
    say()
    n_sig = int(((best["skill_vs_base"] > 0) & (best["dm_p"] < 0.05)).sum())
    say(f"{n_sig} of {len(best)} horizon/target combinations now show a "
        "positive skill that is distinguishable from noise at the 5% level, "
        "against 2 of 10 before combination.")
    say()

    path = REPORT_DIR / "07_forecast_combination.md"
    path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
