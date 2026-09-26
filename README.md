# macro-regime-classifier
Cross-asset machine learning model for identifying macro market regimes

Features include:

- Equity market indicators
- Volatility metrics
- FX signals
- Credit spreads
- Yield curve structure

The goal is to classify market states such as:

- Steady / Risk-On
- Transition / Walk-On-Ice
- Crisis
- Inflation

This project is part of the Imperial College AI & Machine Learning Certificate capstone work.

[Open the notebook in Colab](https://colab.research.google.com/github/huangste/macro-regime-classifier/blob/main/Regime_Model_v2_feature_engineering.ipynb)

---

## Market regime research

Unsupervised classification of historical market regimes, plus an honest
assessment of how far those regimes can be forecast, with a Streamlit
research application on top.

## Layout

```
Regime_Model_v2_feature_engineering.ipynb   the working notebook (latest Colab version)
original/     the notebook exactly as used for the research below, preserved and runnable
regimelab/    the research library
experiments/  numbered research scripts; each writes a report
reports/      generated findings, plus METHODOLOGY.md
app/          the Streamlit application
tests/        look-ahead, label-stability and data-update tests
artifacts/    evaluation outputs used by the app
data/cache/   stored price and FRED history

decision_flow.md, HyperparameterTuning,
BBO_stage2_exploration_week2.ipynb           earlier capstone notes and exploration
```

`original/` holds the exact notebook version the research reproduced and
diagnosed. The root-level notebook is the later Colab version; the two differ
in their hand-mapped regime numbering (regimes 0 and 1 swap names, and the
risk-off mask moves from `[0, 2]` to `[3, 4]`), which is the label-instability
problem the new labelling removes. `experiments/01_reproduce_original.py`
re-runs the `original/` pipeline end-to-end and prints both its published
numbers and the diagnostics.

## Running it

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app/streamlit_app.py
```

The app fits its model on first load and caches it. To fit and save the model
from the command line instead (writes `artifacts/model_state.joblib`):

```bash
python -m regimelab.pipeline
```

The diagnostics tab reads the out-of-sample skill tables that experiments 3,
6 and 7 produce; they are committed, so this is only needed to regenerate
them:

```bash
python experiments/03_forecast_benchmark.py
```

## Experiments

| script | question |
|---|---|
| `01_reproduce_original.py` | what did the original do, and what was wrong with it |
| `02_label_stability.py` | do the regime definitions survive a reseed and a refit |
| `03_forecast_benchmark.py` | does anything beat a persistence forecast out of sample |
| `04_is_the_skill_real.py` | is the measured skill foresight, or overlapping windows |
| `05_sensitivity_and_ablation.py` | does the conclusion hold across K, estimator and sample |
| `06_target_reformulation.py` | is the target, rather than the model, the binding constraint |
| `07_forecast_combination.py` | does averaging with the benchmark recover the signal |

Each writes a markdown report into `reports/`. Run them in order: 04 and 07
both consume the predictions 03 writes, and the app's diagnostics tab reads
the tables produced by 03, 06 and 07.

`reports/METHODOLOGY.md` is the summary of what all of this concluded, and is
also what the app's **Method** tab displays.

## Tests

```bash
python -m pytest tests/ -q
```

The suite is mostly methodological rather than functional: it perturbs the
future and asserts that features, filtered state estimates and walk-forward
folds do not move, it asserts that regime labels are invariant to the random
seed, and it checks that data updates only fetch what is new.

## Data

Prices from Yahoo Finance; rates, credit spreads, oil and VIX from FRED. The
full history is stored as CSV under `data/cache/` and committed, so the app
and every experiment run offline and reproducibly from a fresh clone.

Updates are incremental: only the days after the last stored date are
downloaded (plus a two-week overlap to pick up revisions) and merged in. The
app does this automatically at most every six hours, and **Update market
data now** in the sidebar does it on demand. If a source is unreachable the
stored history is kept and the sidebar says which series are stale. Yahoo
bars are stored only once the New York session has closed, so a pre-open or
intraday price never stands in for a close. From Python:

```python
from regimelab.data import update_cache
update_cache()
```

Data sources remain subject to their providers' terms of use.
