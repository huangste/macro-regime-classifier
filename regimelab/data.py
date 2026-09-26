"""Data acquisition and calendar alignment.

Two fixes relative to the original notebook:

1.  The original concatenated series with different trading calendars and then
    called ``dropna()``.  Because the Nikkei trades on US holidays (and vice
    versa) this silently deleted rows.  Here every series is reindexed onto the
    S&P 500 trading calendar and forward-filled with a short limit.

2.  The original downloaded HYG (credit) and then never included it in the
    panel, so the model had no credit-risk variable at all.  Credit is
    supplied here via Moody's Baa-over-10y spread, which is daily from 1986
    rather than 2007.

All downloads are cached as CSV under ``data/cache`` so that research runs and
the Streamlit app are reproducible and offline-capable.
"""

from __future__ import annotations

import io
import urllib.request
import warnings
from datetime import date

import pandas as pd

from .config import CACHE_DIR, FRED_SERIES, SAMPLES, YAHOO_SERIES

warnings.filterwarnings("ignore", category=FutureWarning)

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1900-01-01&coed=2100-01-01"


def _cache_file(kind: str, key: str) -> "object":
    safe = key.replace("^", "").replace("=", "_").replace(".", "_").replace("-", "_")
    return CACHE_DIR / f"{kind}__{safe}.csv"


def _read_cache(path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def _write_cache(path, s: pd.Series) -> None:
    s.to_frame(name=s.name or "value").to_csv(path)


def fetch_fred(series_id: str, refresh: bool = False) -> pd.Series:
    """Daily FRED series as a float Series indexed by date."""
    path = _cache_file("fred", series_id)
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached.rename(series_id)

    raw = urllib.request.urlopen(_FRED_URL.format(sid=series_id), timeout=90).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "value"]
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=pd.to_datetime(df["date"]),
        name=series_id,
    ).dropna()
    s.index.name = "Date"
    _write_cache(path, s)
    return s


def fetch_yahoo(ticker: str, start: str = "1980-01-01", refresh: bool = False) -> pd.Series:
    """Adjusted close for a Yahoo ticker as a float Series."""
    path = _cache_file("yahoo", ticker)
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            return cached.rename(ticker)

    import yfinance as yf

    df = yf.download(
        ticker, start=start, auto_adjust=True, progress=False, threads=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna().astype(float).rename(ticker)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.index.name = "Date"
    _write_cache(path, s)
    return s


def download_all(refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch every configured series (cached).  Missing sources are skipped."""
    out: dict[str, pd.Series] = {}
    for sid, name in FRED_SERIES.items():
        try:
            out[name] = fetch_fred(sid, refresh=refresh).rename(name)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  [warn] FRED {sid} failed: {type(exc).__name__}: {exc}")
    for tkr, name in YAHOO_SERIES.items():
        try:
            out[name] = fetch_yahoo(tkr, refresh=refresh).rename(name)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  [warn] Yahoo {tkr} failed: {type(exc).__name__}: {exc}")
    return out


def build_panel(
    series: dict[str, pd.Series] | None = None,
    sample: str = "core_1990",
    refresh: bool = False,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Aligned panel of levels on the S&P 500 trading calendar.

    Non-US series are forward-filled across their own holidays with a short
    limit; a gap longer than ``ffill_limit`` sessions is left missing so that
    genuinely absent data is not fabricated.
    """
    spec = SAMPLES[sample]
    if series is None:
        series = download_all(refresh=refresh)

    if "SPX" not in series:
        raise RuntimeError("SPX is required as the master trading calendar")

    calendar = series["SPX"].loc[spec.start:].index

    cols: dict[str, pd.Series] = {}
    for name, s in series.items():
        if name in spec.drop:
            continue
        cols[name] = s.reindex(s.index.union(calendar)).ffill(limit=ffill_limit).reindex(calendar)

    panel = pd.DataFrame(cols).sort_index()

    # Derived level series
    if {"DGS10", "DGS2"} <= set(panel.columns):
        panel["Slope_10Y2Y"] = panel["DGS10"] - panel["DGS2"]
    if {"DGS10", "DGS3MO"} <= set(panel.columns):
        panel["Slope_10Y3M"] = panel["DGS10"] - panel["DGS3MO"]
    if {"BaaYield", "AaaYield"} <= set(panel.columns):
        panel["QualitySpread"] = panel["BaaYield"] - panel["AaaYield"]
    panel = panel.drop(columns=[c for c in ("BaaYield", "AaaYield") if c in panel.columns])

    panel = panel.dropna()
    panel.index.name = "Date"
    return panel


def coverage_report(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in sorted(series.items()):
        rows.append(
            {
                "series": name,
                "start": s.index[0].date(),
                "end": s.index[-1].date(),
                "n": len(s),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    s = download_all()
    print(coverage_report(s).to_string(index=False))
    for key in SAMPLES:
        p = build_panel(s, sample=key)
        print(f"\n{key}: {p.shape} {p.index[0].date()} -> {p.index[-1].date()}")
        print("  cols:", list(p.columns))
