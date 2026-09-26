"""Feature construction.

Design notes
------------
The original notebook used raw multi-horizon percentage returns plus rolling
standard deviations.  Two problems follow from that choice:

*   Scale non-stationarity.  A 21-day S&P return of -8% meant something very
    different in 2017 than in 2008.  Clustering on raw returns therefore mixes
    "how bad is this" with "how volatile is this era", and tends to produce
    clusters that track calendar epochs rather than economic states.

*   No risk-appetite variables.  The panel contained no volatility, credit or
    breadth measure, so a "crisis" could only be recognised through the sign
    and size of trailing returns.

Here returns are scaled by trailing realised volatility (risk-adjusted
momentum), volatility enters in logs (far closer to Gaussian, which matters
for Gaussian mixtures and HMMs), and explicit stress variables -- VIX, the
Moody's Baa spread, drawdown, the stock/bond correlation and small-cap
relative strength -- are included.

Every feature at row ``t`` uses only information available at the close of
``t``.  ``feature_lookback()`` reports the longest window used, which is what
the walk-forward embargo must respect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Longest rolling window used anywhere in build_features().
MAX_LOOKBACK = 252


# --------------------------------------------------------------------------
# Original notebook feature set (kept verbatim for reproduction)
# --------------------------------------------------------------------------
def original_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Exact reproduction of the feature block in the original notebook."""
    price_cols = [c for c in ["SPX", "Nikkei", "NDX", "DXY", "USDJPY", "Gold", "Oil"] if c in raw]
    rate_cols = [c for c in ["DGS10", "DGS2", "US_10Y_2Y_Slope"] if c in raw]

    features = pd.DataFrame(index=raw.index)
    for col in price_cols:
        for h in (1, 10, 21, 63):
            features[f"{col}_{h}d_ret"] = raw[col].pct_change(h)
    for col in rate_cols:
        for h in (1, 10, 21, 63):
            features[f"{col}_{h}d_chg"] = raw[col].diff(h)

    daily_ret = raw[price_cols].pct_change(1)
    for col in price_cols:
        features[f"{col}_21d_vol"] = daily_ret[col].rolling(21).std()
        features[f"{col}_63d_vol"] = daily_ret[col].rolling(63).std()
    return features.dropna()


# --------------------------------------------------------------------------
# v2 feature set
# --------------------------------------------------------------------------
def _risk_adjusted_momentum(px: pd.Series, h: int, vol: pd.Series) -> pd.Series:
    """Log return over ``h`` days divided by its expected size under vol."""
    r = np.log(px).diff(h)
    return r / (vol * np.sqrt(h))


def _realised_vol(px: pd.Series, window: int) -> pd.Series:
    return np.log(px).diff().rolling(window).std()


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Full v2 feature matrix (used for forecasting)."""
    f = pd.DataFrame(index=panel.index)

    price_cols = [
        c for c in ["SPX", "NDX", "Nikkei", "RUT", "Oil", "USDJPY", "DXY", "Gold"]
        if c in panel.columns
    ]

    for col in price_cols:
        px = panel[col]
        v21 = _realised_vol(px, 21)
        v63 = _realised_vol(px, 63)
        for h in (5, 21, 63):
            f[f"{col}_rmom{h}"] = _risk_adjusted_momentum(px, h, v21)
        f[f"{col}_logvol21"] = np.log(v21 * np.sqrt(TRADING_DAYS))
        f[f"{col}_volratio"] = np.log(v21 / v63)
        f[f"{col}_dd252"] = np.log(px / px.rolling(TRADING_DAYS).max())

    # ---- Volatility index -------------------------------------------------
    if "VIX" in panel.columns:
        lv = np.log(panel["VIX"])
        f["VIX_log"] = lv
        f["VIX_chg21"] = lv.diff(21)
        # VIX relative to trailing realised vol: the variance risk premium
        # proxy, high when options are pricing more fear than has been realised
        if "SPX" in panel.columns:
            rv = _realised_vol(panel["SPX"], 21) * np.sqrt(TRADING_DAYS) * 100.0
            f["VIX_vrp"] = np.log(panel["VIX"] / rv.replace(0, np.nan))

    # ---- Credit -----------------------------------------------------------
    if "BaaSpread" in panel.columns:
        # Spread can be near zero or slightly negative; shift before logging.
        s = panel["BaaSpread"]
        f["Baa_log"] = np.log(s - s.min() + 0.25)
        f["Baa_chg21"] = s.diff(21)
        f["Baa_chg63"] = s.diff(63)
    if "QualitySpread" in panel.columns:
        f["Quality_level"] = panel["QualitySpread"]
        f["Quality_chg21"] = panel["QualitySpread"].diff(21)

    # ---- Rates and curve --------------------------------------------------
    for col in ["DGS10", "DGS2"]:
        if col in panel.columns:
            f[f"{col}_chg21"] = panel[col].diff(21)
            f[f"{col}_chg63"] = panel[col].diff(63)
    for col in ["Slope_10Y2Y", "Slope_10Y3M"]:
        if col in panel.columns:
            f[f"{col}_level"] = panel[col]
            f[f"{col}_chg63"] = panel[col].diff(63)

    # ---- Cross-asset structure -------------------------------------------
    if {"SPX", "DGS10"} <= set(panel.columns):
        eq = np.log(panel["SPX"]).diff()
        dy = panel["DGS10"].diff()
        f["SPX_bond_corr63"] = eq.rolling(63).corr(dy)
    if {"RUT", "SPX"} <= set(panel.columns):
        f["RUT_rel_SPX_63"] = np.log(panel["RUT"] / panel["SPX"]).diff(63)
    if {"NDX", "SPX"} <= set(panel.columns):
        f["NDX_rel_SPX_63"] = np.log(panel["NDX"] / panel["SPX"]).diff(63)

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    f.index.name = "Date"
    return f


# --------------------------------------------------------------------------
# Compact labelling feature set
# --------------------------------------------------------------------------
# Deliberately small and economically interpretable.  A low-dimensional space
# makes the mixture/HMM fit far more stable across seeds and data vintages,
# which is the root cause of the label-permutation problem.
LABEL_FEATURES: tuple[str, ...] = (
    "SPX_rmom21",
    "SPX_rmom63",
    "SPX_logvol21",
    "SPX_volratio",
    "SPX_dd252",
    "VIX_log",
    "Baa_log",
    "Baa_chg63",
    "Slope_10Y2Y_level",
    "DGS10_chg63",
    "RUT_rel_SPX_63",
    "Nikkei_rmom63",
    "USDJPY_rmom63",
    "SPX_bond_corr63",
)


def label_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LABEL_FEATURES if c in features.columns]
    missing = set(LABEL_FEATURES) - set(cols)
    if missing:
        print(f"  [warn] labelling features unavailable: {sorted(missing)}")
    return features[cols]


def feature_lookback() -> int:
    return MAX_LOOKBACK
