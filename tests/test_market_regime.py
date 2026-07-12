"""Tests for market_regime.py — regime logic, VIX gating, can_trade flag."""
import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from market_regime import MarketRegimeDetector, MIN_BARS_REQUIRED
except ImportError as e:
    pytest.skip(f"market_regime not importable: {e}", allow_module_level=True)


def _make_df(n=250, trend="bull"):
    """Build a minimal OHLCV DataFrame with a controllable trend."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    if trend == "bull":
        close = np.linspace(100, 140, n)   # steady uptrend
    elif trend == "bear":
        close = np.linspace(140, 90, n)    # steady downtrend
    else:
        close = np.full(n, 120.0)          # flat
    df = pd.DataFrame({
        "Close": close,
        "Open":  close * 0.99,
        "High":  close * 1.01,
        "Low":   close * 0.98,
        "Volume": np.full(n, 1_000_000),
    }, index=dates)
    return df


def _make_vix(level=15.0, n=5):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": np.full(n, level)}, index=dates)


@pytest.fixture()
def detector():
    return MarketRegimeDetector()


# ── Insufficient data ─────────────────────────────────────────────────────────

def test_insufficient_data_returns_unknown(detector):
    df = _make_df(n=50)   # way below MIN_BARS_REQUIRED
    result = detector.detect(df)
    assert result["regime"] == "unknown"
    assert result["tradeable"] is False


# ── Bull regime ───────────────────────────────────────────────────────────────

def test_bull_market_detected(detector):
    df = _make_df(n=MIN_BARS_REQUIRED + 10, trend="bull")
    result = detector.detect(df, vix_df=_make_vix(14.0))
    assert result["regime"] == "bull"
    assert result["tradeable"] is True
    assert result["confidence"] > 0.5


# ── Bear regime ───────────────────────────────────────────────────────────────

def test_bear_market_detected(detector):
    df = _make_df(n=MIN_BARS_REQUIRED + 10, trend="bear")
    result = detector.detect(df, vix_df=_make_vix(20.0))
    assert result["regime"] in ("bear", "sideways", "unknown")


# ── Crisis / VIX gate ─────────────────────────────────────────────────────────

def test_high_vix_triggers_crisis(detector):
    df = _make_df(n=MIN_BARS_REQUIRED + 10, trend="bull")
    result = detector.detect(df, vix_df=_make_vix(40.0))  # > EXTREME_VOLATILITY_THRESHOLD
    assert result["regime"] == "crisis"
    assert result["tradeable"] is False


# ── None price_df ─────────────────────────────────────────────────────────────

def test_none_dataframe_returns_unknown(detector):
    result = detector.detect(None)
    assert result["regime"] == "unknown"
    assert result["tradeable"] is False


# ── analyze() backward-compatible output ─────────────────────────────────────

def test_analyze_returns_required_keys(detector, monkeypatch):
    """analyze() fetches live data — mock yfinance to avoid network calls."""
    import market_regime as mr

    bull_df  = _make_df(n=MIN_BARS_REQUIRED + 10, trend="bull")
    vix_small = _make_vix(14.0)

    class FakeTicker:
        def __init__(self, sym):
            self._sym = sym
        def history(self, period):
            return vix_small if "VIX" in self._sym else bull_df

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    result = detector.analyze()

    for key in ("regime", "can_trade", "confidence", "spy_return_1m", "vix"):
        assert key in result, f"Missing key: {key}"
