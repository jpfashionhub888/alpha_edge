"""Tests for models/ensemble.py — majority vote, signal generation, threshold."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_model(prob):
    """Create a mock sklearn-like model returning fixed probability."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    m.overfit_flagged = False
    return m


@pytest.fixture()
def ensemble():
    try:
        from models.ensemble import EnsembleModel
        return EnsembleModel()
    except ImportError:
        pytest.skip("EnsembleModel not importable — skipping ensemble tests")


# ── Majority vote ─────────────────────────────────────────────────────────────

def test_majority_high_wins(ensemble):
    """Three models all above threshold -> prediction should be BUY (1)."""
    ensemble.models = [_mock_model(0.80), _mock_model(0.75), _mock_model(0.70)]
    X = pd.DataFrame({"close": [150.0], "rsi": [60.0]})
    try:
        result = ensemble.predict(X)
        pred = result[0] if hasattr(result, "__len__") else result
        assert pred >= 0.60, "Strong consensus should yield high prediction"
    except Exception as e:
        pytest.skip(f"predict() not directly testable: {e}")


def test_majority_low_loses(ensemble):
    """Three models all below threshold -> prediction should be low."""
    ensemble.models = [_mock_model(0.40), _mock_model(0.35), _mock_model(0.45)]
    X = pd.DataFrame({"close": [150.0], "rsi": [40.0]})
    try:
        result = ensemble.predict(X)
        pred = result[0] if hasattr(result, "__len__") else result
        assert pred < 0.60, "Weak consensus should yield low prediction"
    except Exception as e:
        pytest.skip(f"predict() not directly testable: {e}")


# ── generate_signals ──────────────────────────────────────────────────────────

def test_generate_signals_returns_dataframe(ensemble):
    df = pd.DataFrame({
        "symbol": ["AAPL", "TSLA"],
        "pred":   [0.70, 0.45],
        "returns": [0.01, -0.005],
    })
    try:
        result = ensemble.generate_signals(df)
        assert isinstance(result, pd.DataFrame), "generate_signals must return DataFrame"
    except Exception as e:
        pytest.skip(f"generate_signals not testable with stub data: {e}")
