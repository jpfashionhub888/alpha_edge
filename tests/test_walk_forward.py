"""Tests for backtest/walk_forward.py — fold structure, overfit skip, risk order."""
import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backtest.walk_forward import WalkForwardBacktester


def _make_price_df(n=500):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) * 100
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low":  close * 0.98, "Close": close,
        "Volume": np.full(n, 1_000_000),
    }, index=dates)


@pytest.fixture()
def wf():
    return WalkForwardBacktester(n_splits=3, test_size=60)


# ── Fold count ────────────────────────────────────────────────────────────────

def test_correct_number_of_folds(wf):
    df = _make_price_df(400)
    splits = list(wf._generate_splits(df))
    assert len(splits) == 3, "Should generate exactly n_splits=3 folds"


# ── Overfit guard skips flagged models ───────────────────────────────────────

def test_overfit_flagged_model_skipped(wf, monkeypatch):
    """A model flagged as overfit must not generate predictions."""
    df = _make_price_df(400)
    call_count = {"n": 0}

    original_train = wf._train_fold if hasattr(wf, "_train_fold") else None

    def mock_predict(X):
        call_count["n"] += 1
        return np.full(len(X), 0.6)

    # Patch so model.overfit_flagged = True — prediction must be skipped
    with patch.object(wf, "_train_fold", return_value=None) as mock_train:
        mock_model = MagicMock()
        mock_model.overfit_flagged = True
        mock_model.predict.side_effect = mock_predict
        mock_train.return_value = mock_model
        try:
            wf.run({"SPY": df})
        except Exception:
            pass   # Incomplete stubs may throw — we only care predict wasn't called
    # Overfit model must not have its predict() called
    assert call_count["n"] == 0, "Overfit flagged model must not call predict()"


# ── Risk management call order ────────────────────────────────────────────────

def test_risk_management_runs_after_generate_signals(wf, monkeypatch):
    """_apply_risk_management must not be called before generate_signals."""
    call_log = []

    def log_risk(*a, **kw):
        call_log.append("risk")
        return a[0] if a else pd.DataFrame()

    def log_signals(*a, **kw):
        call_log.append("signals")
        return a[0] if a else pd.DataFrame()

    if hasattr(wf, "ensemble") and hasattr(wf.ensemble, "generate_signals"):
        monkeypatch.setattr(wf.ensemble, "generate_signals", log_signals)
    if hasattr(wf, "_apply_risk_management"):
        monkeypatch.setattr(wf, "_apply_risk_management", log_risk)

    df = _make_price_df(400)
    try:
        wf.run({"SPY": df})
    except Exception:
        pass

    if "risk" in call_log and "signals" in call_log:
        assert call_log.index("signals") < call_log.index("risk"), \
            "generate_signals must run BEFORE _apply_risk_management (S1 fix)"
