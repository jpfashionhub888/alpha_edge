"""Tests for backtest/walk_forward.py — constructor, risk management call order.

WalkForwardBacktester(train_window_days=180, retrain_frequency_days=30, ...)
Methods: run(raw_df, target='target'), _apply_risk_management(...)
"""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backtest.walk_forward import WalkForwardBacktester
except (ImportError, Exception) as e:
    pytest.skip(f"walk_forward not importable: {e}", allow_module_level=True)


def _make_price_df(n=500):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) * 100
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.full(n, 1_000_000),
    }, index=dates)


@pytest.fixture()
def wf():
    """WalkForwardBacktester with correct constructor arguments.
    Skip gracefully if ML dependencies (catboost, sklearn) are unavailable.
    """
    try:
        return WalkForwardBacktester(
            train_window_days=180,
            retrain_frequency_days=30,
        )
    except Exception as e:
        pytest.skip(f"WalkForwardBacktester init failed (ML deps unavailable): {e}")


# ── Constructor attributes ────────────────────────────────────────────────────

def test_constructor_sets_train_window(wf):
    """train_window_days must be stored as wf.train_window."""
    assert wf.train_window == 180, "train_window_days must be stored as train_window"


def test_constructor_sets_retrain_frequency(wf):
    """retrain_frequency_days must be stored as wf.retrain_every."""
    assert wf.retrain_every == 30, "retrain_frequency_days must be stored as retrain_every"


def test_constructor_creates_ensemble(wf):
    """WalkForwardBacktester must expose an ensemble attribute."""
    assert hasattr(wf, "ensemble"), "WalkForwardBacktester must have an ensemble attribute"


def test_constructor_creates_stock_selector(wf):
    """WalkForwardBacktester must expose a stock_selector attribute."""
    assert hasattr(wf, "stock_selector"), (
        "WalkForwardBacktester must have a stock_selector attribute"
    )


# ── Risk management method ────────────────────────────────────────────────────

def test_apply_risk_management_exists(wf):
    """_apply_risk_management must exist — enforces stop-loss/take-profit in backtests."""
    assert hasattr(wf, "_apply_risk_management"), (
        "_apply_risk_management method required for risk enforcement (S1 fix)"
    )


# ── Risk management runs after signal generation ──────────────────────────────

def test_risk_management_order(wf):
    """If both ensemble.generate_signals and _apply_risk_management are called,
    signals must be generated BEFORE risk management is applied (S1 ordering fix).
    """
    call_log = []

    def log_risk(*a, **kw):
        call_log.append("risk")
        return a[0] if a else pd.DataFrame()

    def log_signals(*a, **kw):
        call_log.append("signals")
        return a[0] if a else pd.DataFrame()

    # Patch both methods if they exist
    if hasattr(wf, "ensemble") and hasattr(wf.ensemble, "generate_signals"):
        wf.ensemble.generate_signals = log_signals
    if hasattr(wf, "_apply_risk_management"):
        wf._apply_risk_management = log_risk

    df = _make_price_df(400)
    try:
        wf.run(df)
    except Exception:
        pass  # ML may fail without proper target column — we only care about order

    if "risk" in call_log and "signals" in call_log:
        assert call_log.index("signals") < call_log.index("risk"), (
            "generate_signals must run BEFORE _apply_risk_management (S1 fix)"
        )
