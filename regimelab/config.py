"""Central configuration: paths, data universe, and modelling constants.

Everything that the research depends on is declared here so that a run is
reproducible from a single object rather than from notebook cell order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data universe
# --------------------------------------------------------------------------
# Yahoo tickers -> internal name.  The master trading calendar is SPX.
YAHOO_SERIES: dict[str, str] = {
    "^GSPC": "SPX",
    "^NDX": "NDX",
    "^N225": "Nikkei",
    "^RUT": "RUT",
    "GC=F": "Gold",
    "DX-Y.NYB": "DXY",
}

# FRED series -> internal name.  Preferred over Yahoo where a longer or
# cleaner history exists (oil, VIX, USDJPY, rates, credit).
FRED_SERIES: dict[str, str] = {
    "VIXCLS": "VIX",
    "DCOILWTICO": "Oil",
    "DEXJPUS": "USDJPY",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "DGS3MO": "DGS3MO",
    "BAA10Y": "BaaSpread",   # Moody's Baa yield less 10y Treasury, daily 1986+
    "DBAA": "BaaYield",
    "DAAA": "AaaYield",
}

# Series carried as *levels in percent* -> differenced rather than pct-changed.
RATE_LIKE: tuple[str, ...] = (
    "DGS10", "DGS2", "DGS3MO", "BaaSpread", "BaaYield", "AaaYield",
    "Slope_10Y2Y", "Slope_10Y3M", "QualitySpread",
)

# Price series -> pct-changed.
PRICE_LIKE: tuple[str, ...] = (
    "SPX", "NDX", "Nikkei", "RUT", "Gold", "DXY", "Oil", "USDJPY",
)

# VIX is a level in vol points; treated separately (log level + change).
VOL_LIKE: tuple[str, ...] = ("VIX",)


@dataclass(frozen=True)
class SampleSpec:
    """A named choice of start date and column subset.

    ``core_long`` buys ~10 extra years of history by dropping gold and the
    Yahoo dollar index (both start in 2000 and 2000s respectively).
    """

    name: str
    start: str
    drop: tuple[str, ...] = ()


# Gold (Yahoo continuous future) only starts 2000-08, so it is the binding
# constraint on sample length; dropping it buys ten extra years of history.
SAMPLES: dict[str, SampleSpec] = {
    "core_1990": SampleSpec("core_1990", "1990-01-01", drop=("Gold",)),
    "full_2000": SampleSpec("full_2000", "2000-01-01", drop=()),
}


# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------
FEATURE_HORIZONS: tuple[int, ...] = (5, 21, 63)
VOL_WINDOWS: tuple[int, ...] = (21, 63)
FORECAST_HORIZONS: tuple[int, ...] = (5, 10, 15, 20, 25)

DEFAULT_N_REGIMES: int = 4
RANDOM_STATE: int = 42


@dataclass(frozen=True)
class RunConfig:
    sample: str = "core_1990"
    n_regimes: int = DEFAULT_N_REGIMES
    labeler: str = "hmm"                 # "hmm" | "gmm"
    n_pca_components: float | int = 0.90
    horizons: tuple[int, ...] = FORECAST_HORIZONS
    random_state: int = RANDOM_STATE
    # Walk-forward validation
    initial_train_years: float = 8.0
    test_block_days: int = 126           # ~6 months per out-of-sample block
    extra: dict = field(default_factory=dict)
