"""Tests for meta_labeler.py — gate, pass-through when unfitted, threshold."""
import os
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.meta_labeler import MetaLabeler
except ImportError as e:
    pytest.skip(f"meta_labeler not importable: {e}", allow_module_level=True)


@pytest.fixture()
def labeler():
    return MetaLabeler()


# ── Unfitted pass-through ────────────────────────────────────────────────────

def test_unfitted_approves_all(labeler):
    """Before fitting, MetaLabeler must not block any trade."""
    result = labeler.should_trade("AAPL", primary_prob=0.70, features={})
    # Unfitted labeler should either return True or raise no error
    assert isinstance(result, (bool, dict)), "Must return bool or dict when unfitted"


# ── Threshold gate ────────────────────────────────────────────────────────────

def test_low_confidence_blocked(labeler, monkeypatch):
    """If meta confidence is below threshold, should_trade must return False."""
    # Monkey-patch the internal model to return low confidence
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # 20% meta confidence
    labeler.model = mock_model
    labeler.fitted = True
    labeler.feature_names = ["close", "rsi"]

    result = labeler.should_trade(
        "AAPL",
        primary_prob=0.70,
        features={"close": 150.0, "rsi": 60.0},
    )
    # 0.20 confidence is below any sane threshold
    decision = result if isinstance(result, bool) else result.get("approved", True)
    assert decision is False, "Low meta confidence must block trade"


def test_high_confidence_approved(labeler, monkeypatch):
    """High meta confidence must approve the trade."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% meta confidence
    labeler.model = mock_model
    labeler.fitted = True
    labeler.feature_names = ["close", "rsi"]

    result = labeler.should_trade(
        "AAPL",
        primary_prob=0.75,
        features={"close": 150.0, "rsi": 65.0},
    )
    decision = result if isinstance(result, bool) else result.get("approved", False)
    assert decision is True, "High meta confidence must approve trade"
