"""Out-of-sample evaluation for overlapping-horizon forecasts.

Three things are done differently from the original notebook.

*Purging.*  With a horizon of ``h`` days sampled daily, the last ``h`` training
rows have targets that fall inside the test block.  ``walk_forward_folds``
removes them, so no training label is ever drawn from the period being scored.

*Baselines that already know about persistence.*  Regimes are very persistent,
so beating the unconditional base rate is close to free.  The benchmark that
matters is a Markov forecast built from the training transition matrix, which
uses today's state and nothing else.  ``skill_score`` is reported against it.

*Honest standard errors.*  Overlapping windows make daily loss differentials
strongly autocorrelated.  ``diebold_mariano`` uses a Newey-West HAC variance
with a bandwidth tied to the horizon, which is the difference between "this
looks better" and "this is distinguishable from noise".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-6


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    train_start: int
    train_end: int      # exclusive, already purged
    test_start: int
    test_end: int       # exclusive

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


def walk_forward_folds(
    n: int,
    horizon: int,
    initial_train: int,
    test_block: int,
    expanding: bool = True,
) -> list[Fold]:
    """Expanding-window folds with the training tail purged by ``horizon``.

    A row at index ``i`` carries the label observed at ``i + horizon``, so
    training must stop ``horizon`` rows before the test block begins.
    """
    folds: list[Fold] = []
    test_start = initial_train
    while test_start + test_block <= n - horizon:
        train_end = test_start - horizon
        train_start = 0 if expanding else max(0, train_end - initial_train)
        if train_end - train_start < 250:
            break
        folds.append(Fold(train_start, train_end, test_start,
                          min(test_start + test_block, n - horizon)))
        test_start += test_block
    return folds


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def binary_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation log loss (kept per-observation for the DM test)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def skill_score(loss_model: np.ndarray, loss_base: np.ndarray) -> float:
    """Fraction of the baseline's loss removed.  Zero means no value added."""
    lb = float(np.mean(loss_base))
    return float(1.0 - np.mean(loss_model) / lb) if lb > 0 else np.nan


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """HAC standard error of the mean of a serially correlated series."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dm = d - d.mean()
    gamma0 = float(dm @ dm) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        gk = float(dm[k:] @ dm[:-k]) / n
        var += 2.0 * w * gk
    var = max(var, 1e-18)
    return float(np.sqrt(var / n))


def diebold_mariano(loss_model: np.ndarray, loss_base: np.ndarray,
                    horizon: int) -> tuple[float, float]:
    """One-sided DM statistic for "model beats baseline".

    Returns ``(t_stat, p_value)``.  A negative ``t_stat`` means the model has
    lower loss.  The HAC bandwidth is ``2*horizon`` because daily sampling of
    an ``h``-day horizon induces autocorrelation out to at least lag ``h``.
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_base, dtype=float)
    if len(d) < 30:
        return np.nan, np.nan
    se = newey_west_se(d, lag=max(1, 2 * horizon))
    t = float(d.mean() / se)
    from scipy.stats import norm
    return t, float(norm.cdf(t))       # P(model at least this much better)


def effective_sample_size(n: int, horizon: int) -> int:
    return int(np.floor(n / max(horizon, 1)))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise_predictions(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    horizon: int,
    baseline: str = "persist_markov",
) -> pd.DataFrame:
    """Pooled out-of-sample table for one horizon."""
    losses = {k: binary_log_loss(y, v) for k, v in preds.items()}
    base = losses.get(baseline)

    rows = []
    for name, p in preds.items():
        row = {
            "model": name,
            "log_loss": float(np.mean(losses[name])),
            "brier": float(np.mean(brier(y, p))),
            "auc": safe_auc(y, p),
        }
        if base is not None:
            row["skill_vs_base"] = skill_score(losses[name], base)
            if name != baseline:
                t, pv = diebold_mariano(losses[name], base, horizon)
                row["dm_t"] = t
                row["dm_p"] = pv
            else:
                row["dm_t"] = np.nan
                row["dm_p"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    df.attrs["n_obs"] = len(y)
    df.attrs["n_eff"] = effective_sample_size(len(y), horizon)
    df.attrs["base_rate"] = float(np.mean(y))
    return df


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration: predicted probability versus realised frequency."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": float(p[m].mean()),
            "observed_freq": float(y[m].mean()),
        })
    return pd.DataFrame(rows)
