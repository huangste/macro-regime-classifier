# Market regime research

Unsupervised classification of historical market regimes, plus an honest
assessment of how far those regimes can be forecast.

## Layout

```
original/     the notebook exactly as received, preserved and runnable
regimelab/    the research library
experiments/  numbered research scripts; each writes a report
reports/      generated findings, plus METHODOLOGY.md
app/          the Streamlit application
tests/        look-ahead and label-stability tests
artifacts/    cached fitted state and evaluation outputs
data/cache/   cached price and FRED downloads
```

The original notebook is untouched in `original/` and is the first commit in
the repository, so the historical classification it produced can always be
reproduced. `experiments/01_reproduce_original.py` re-runs its pipeline
end-to-end and prints both its published numbers and the diagnostics.

## Running it

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Fetch data and fit the production model (writes `artifacts/model_state.joblib`):

```bash
python -m regimelab.pipeline
```

Start the application:

```bash
python -m streamlit run app/streamlit_app.py
```

The app fits its own model on first load and caches it, so `regimelab.pipeline`
is optional. What is *not* optional for the diagnostics tab is experiment 3,
which produces the out-of-sample skill tables the app displays next to every
forecast:

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
folds do not move, and it asserts that regime labels are invariant to the
random seed.

## Data

Prices from Yahoo Finance; rates, credit spreads, oil and VIX from FRED. All
downloads are cached as CSV under `data/cache/`, so runs are reproducible and
work offline once populated. Delete the cache or press **Refresh market data**
in the app to re-download.
