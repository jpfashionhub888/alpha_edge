"""Tests for models/meta_labeler.py — unfitted pass-through, confidence threshold gate.

MetaLabeler.should_trade(X: pd.DataFrame, primary_prob: float) -> Tuple[bool, float]
Private attributes: _is_fitted, _model, _feature_names, threshold
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.meta_labeler import MetaLabeler
except ImportError as e:
    pytest.skip(f"meta_labeler not importable: {e}", allow_module_level=True)


def _make_features(n=1):
    """Minimal feature DataFrame with two columns."""
    return pd.DataFrame({"close": [150.0] * n, "rsi": [60.0] * n})


@pytest.fixture()
def labeler():
    return MetaLabeler()


# ── Unfitted pass-through ─────────────────────────────────────────────────────

def test_unfitted_approves_all(labeler):
    """Before fitting, MetaLabeler must pass-through (not block) any trade."""
    approved, conf = labeler.should_trade(_make_features(), primary_prob=0.70)
    # Unfitted labeler returns (True, 1.0) — never blocks when no model exists
    assert approved is True, "Unfitted MetaLabeler must approve all trades"
    assert isinstance(conf, float), "Confidence must be a float"


# ── Threshold gate ─────────────────────────────────────────────────────────────

def test_low_confidence_blocked(labeler):
    """If meta confidence is below threshold, should_trade must return (False, ...)."""
    # Patch the internal model to return low-confidence probabilities
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # 20% confidence
    labeler._model     = mock_model
    labeler._is_fitted = True
    labeler._feature_names = ["close", "rsi"]

    approved, conf = labeler.should_trade(_make_features(), primary_prob=0.70)
    assert approved is False, f"Low meta confidence (0.20) must block trade, got approved={approved}"
    assert conf < labeler.threshold, "Confidence must be below threshold when blocked"


def test_high_confidence_approved(labeler):
    """High meta confidence must approve the trade."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% confidence
    labeler._model     = mock_model
    labeler._is_fitted = True
    labeler._feature_names = ["close", "rsi"]

    approved, conf = labeler.should_trade(_make_features(), primary_prob=0.75)
    assert approved is True, f"High meta confidence (0.90) must approve trade, got approved={approved}"
    assert conf >= labeler.threshold, "Confidence must meet threshold when approved"


# ── Return type contract ───────────────────────────────────────────────────────

def test_should_trade_returns_tuple(labeler):
    """should_trade() must always return a (bool, float) tuple."""
    result = labeler.should_trade(_make_features(), primary_prob=0.65)
    assert isinstance(result, tuple) and len(result) == 2, (
        "should_trade must return a (bool, float) 2-tuple"
    )
    approved, conf = result
    assert isinstance(approved, bool), "First element must be bool"
    assert isinstance(conf, float), "Second element must be float"
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} must be in [0, 1]"
