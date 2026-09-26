"""Incremental data updates: only new rows are fetched, history is kept."""

from __future__ import annotations

import pandas as pd
import pytest

import regimelab.data as data


def _series(start: str, n: int, value0: float = 100.0) -> pd.Series:
    idx = pd.bdate_range(start, periods=n, name="Date")
    return pd.Series([value0 + i for i in range(n)], index=idx, dtype=float)


def test_merge_keeps_history_and_prefers_new_values():
    old = _series("2026-01-01", 10)
    new = _series("2026-01-08", 10, value0=500.0)
    out = data.merge_update(old, new)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out.loc[old.index[0]] == old.iloc[0]          # history kept
    assert out.loc[new.index[0]] == new.iloc[0]          # overlap overwritten
    assert out.index[-1] == new.index[-1]                # new rows appended


def test_restatement_detected_only_beyond_tolerance():
    old = _series("2026-01-01", 5)
    assert not data.history_restated(old, old * (1 + 1e-5))
    assert data.history_restated(old, old * 1.01)
    assert not data.history_restated(old, _series("2027-01-01", 5))


def test_todays_bar_is_dropped_until_the_new_york_close():
    s = _series("2026-09-21", 5)                          # Mon 21 .. Fri 25
    before_close = pd.Timestamp("2026-09-25 11:00", tz="America/New_York")
    after_close = pd.Timestamp("2026-09-25 17:00", tz="America/New_York")
    assert data.completed_sessions(s, before_close).index[-1] == pd.Timestamp("2026-09-24")
    assert data.completed_sessions(s, after_close).index[-1] == pd.Timestamp("2026-09-25")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    return tmp_path


def test_update_downloads_only_from_near_the_last_cached_date(tmp_cache):
    history = _series("2020-01-01", 1500)
    calls = []

    def fake(key, start):
        calls.append(start)
        full = _series("2020-01-01", 1510)                # ten new sessions
        return full[full.index >= pd.Timestamp(start)]

    data._write_cache(data._cache_file("fred", "X"), history.rename("X"))
    out = data._fetch("fred", "X", fake, "1900-01-01", refresh=False,
                      update=True, market_prices=False)

    assert len(calls) == 1
    assert pd.Timestamp(calls[0]) > history.index[-30]    # not a full download
    assert len(out) == 1510
    reread = data._read_cache(data._cache_file("fred", "X"))
    assert len(reread) == 1510                            # persisted


def test_restated_history_triggers_one_full_redownload(tmp_cache):
    history = _series("2020-01-01", 300)
    calls = []

    def fake(key, start):
        calls.append(start)
        full = _series("2020-01-01", 305) * 1.05          # everything rescaled
        return full[full.index >= pd.Timestamp(start)]

    data._write_cache(data._cache_file("yahoo", "Y"), history.rename("Y"))
    out = data._fetch("yahoo", "Y", fake, "1980-01-01", refresh=False,
                      update=True, market_prices=True)
    assert calls[-1] == "1980-01-01"
    assert out.iloc[0] == pytest.approx(history.iloc[0] * 1.05)


def test_plain_read_never_touches_the_network(tmp_cache):
    data._write_cache(data._cache_file("fred", "Z"), _series("2020-01-01", 50).rename("Z"))

    def boom(key, start):
        raise AssertionError("network called on a cached read")

    out = data._fetch("fred", "Z", boom, "1900-01-01", refresh=False,
                      update=False, market_prices=False)
    assert len(out) == 50


def test_cache_signature_moves_only_when_the_last_date_moves(tmp_cache, monkeypatch):
    monkeypatch.setattr(data, "FRED_SERIES", {"A": "A"})
    monkeypatch.setattr(data, "YAHOO_SERIES", {})
    data._write_cache(data._cache_file("fred", "A"), _series("2020-01-01", 10).rename("A"))
    sig1 = data.cache_signature()
    data._write_cache(data._cache_file("fred", "A"), _series("2020-01-01", 10).rename("A"))
    assert data.cache_signature() == sig1
    data._write_cache(data._cache_file("fred", "A"), _series("2020-01-01", 11).rename("A"))
    assert data.cache_signature() != sig1
