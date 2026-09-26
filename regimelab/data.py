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
import time
import urllib.request
import warnings

import pandas as pd

from .config import CACHE_DIR, FRED_SERIES, SAMPLES, YAHOO_SERIES

warnings.filterwarnings("ignore", category=FutureWarning)

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed=2100-01-01"

# An incremental update re-downloads this many calendar days before the last
# cached date, so late revisions to recent observations are picked up.
UPDATE_OVERLAP_DAYS = 14

# If a re-downloaded overlap differs from the cache by more than this
# (relative), history has been restated -- e.g. a dividend adjustment -- and
# the whole series is re-downloaded instead of appended to.
RESTATEMENT_TOL = 1e-3


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


def merge_update(old: pd.Series, new: pd.Series) -> pd.Series:
    """Append ``new`` to ``old``; where they overlap, the new value wins."""
    s = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    return s[~s.index.duplicated(keep="last")]


def history_restated(old: pd.Series, new: pd.Series, tol: float = RESTATEMENT_TOL) -> bool:
    """True if ``new`` disagrees with ``old`` on the dates they share."""
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return False
    a, b = old.loc[common].astype(float), new.loc[common].astype(float)
    rel = (a - b).abs() / a.abs().clip(lower=1e-9)
    return bool((rel > tol).any())


def _download_fred(series_id: str, start: str) -> pd.Series:
    url = _FRED_URL.format(sid=series_id, start=start)
    raw = urllib.request.urlopen(url, timeout=90).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "value"]
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=pd.to_datetime(df["date"]),
        name=series_id,
    ).dropna()
    s.index.name = "Date"
    return s


def _download_yahoo(ticker: str, start: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna().astype(float).rename(ticker)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.index.name = "Date"
    return completed_sessions(s)


def completed_sessions(s: pd.Series, now: pd.Timestamp | None = None) -> pd.Series:
    """Drop today's bar until the New York close has passed.

    Yahoo returns a row for the current date before and during the session,
    priced at the last trade.  Storing it would let a pre-open or intraday
    price stand in for a close, so only finished sessions are kept.
    """
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    today = now.tz_localize(None).normalize() if now.tzinfo else now.normalize()
    if now.hour * 60 + now.minute < 16 * 60 + 30:
        return s[s.index < today]
    return s[s.index <= today]


def _fetch(kind: str, key: str, downloader, full_start: str,
           refresh: bool, update: bool, market_prices: bool) -> pd.Series:
    """Shared cache logic.

    ``refresh``  re-download the whole history.
    ``update``   download only from shortly before the last cached date and
                 merge; falls back to a full download if history was restated.
    neither      return the cache, downloading in full only if it is missing.
    """
    path = _cache_file(kind, key)
    cached = None if refresh else _read_cache(path)

    if cached is not None and not update:
        return cached.rename(key)

    if cached is not None and update:
        if market_prices:
            cached = completed_sessions(cached)
        start = (cached.index[-1] - pd.Timedelta(days=UPDATE_OVERLAP_DAYS)).date().isoformat()
        new = downloader(key, start)
        # The last cached day is excluded from the comparison: it may be a
        # bar stored before this safeguard existed, and a corrected close is
        # an update, not a restatement of history.
        if market_prices and history_restated(cached.iloc[:-1], new):
            s = downloader(key, full_start)
        else:
            s = merge_update(cached, new)
    else:
        s = downloader(key, full_start)

    s = s.rename(key)
    s.index.name = "Date"
    _write_cache(path, s)
    return s


def fetch_fred(series_id: str, refresh: bool = False, update: bool = False) -> pd.Series:
    """Daily FRED series as a float Series indexed by date.

    FRED revises recent observations, so an update keeps the new values on the
    overlap rather than treating a revision as a restatement of history.
    """
    return _fetch("fred", series_id, _download_fred, "1900-01-01",
                  refresh, update, market_prices=False)


def fetch_yahoo(ticker: str, start: str = "1980-01-01", refresh: bool = False,
                update: bool = False) -> pd.Series:
    """Adjusted close for a Yahoo ticker as a float Series."""
    return _fetch("yahoo", ticker, _download_yahoo, start,
                  refresh, update, market_prices=True)


def update_cache() -> dict[str, dict]:
    """Bring every cached series up to date, downloading only what is new.

    Never raises: a series that fails to update keeps its cached history, and
    the failure is reported so the caller can say what the data is as of.
    """
    report: dict[str, dict] = {}
    jobs = [("fred", sid, name) for sid, name in FRED_SERIES.items()] + \
           [("yahoo", tkr, name) for tkr, name in YAHOO_SERIES.items()]
    for kind, key, name in jobs:
        before = _read_cache(_cache_file(kind, key))
        last_before = before.index[-1].date() if before is not None else None
        fn = fetch_fred if kind == "fred" else fetch_yahoo
        try:
            try:
                s = fn(key, update=True)
            except Exception:
                time.sleep(2)            # one retry for transient timeouts
                s = fn(key, update=True)
            report[name] = {"ok": True, "before": last_before,
                            "after": s.index[-1].date(),
                            "new_rows": int(len(s) - (len(before) if before is not None else 0))}
        except Exception as exc:  # network dependent
            report[name] = {"ok": False, "before": last_before, "after": last_before,
                            "error": f"{type(exc).__name__}: {exc}"[:200]}
    return report


def _last_cached_date(path) -> str | None:
    """Date on the final line of a cache file, without parsing the whole file."""
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 256))
        tail = fh.read().decode("utf-8", errors="ignore").strip().splitlines()
    return tail[-1].split(",")[0] if tail else None


def cache_signature() -> str:
    """Changes exactly when any cached series gains or loses its last date."""
    parts = []
    for sid in FRED_SERIES:
        parts.append(f"{sid}:{_last_cached_date(_cache_file('fred', sid))}")
    for tkr in YAHOO_SERIES:
        parts.append(f"{tkr}:{_last_cached_date(_cache_file('yahoo', tkr))}")
    return "|".join(parts)


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
