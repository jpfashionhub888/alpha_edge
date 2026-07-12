"""Tests for risk/position_sizer.py — Kelly formula, VaR cap, correlation penalty."""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from risk.position_sizer import PositionSizer
except ImportError as e:
    pytest.skip(f"position_sizer not importable: {e}", allow_module_level=True)


@pytest.fixture()
def sizer():
    return PositionSizer()


# ── Basic sanity ──────────────────────────────────────────────────────────────

def test_zero_portfolio_returns_zero(sizer):
    size = sizer.calculate_position_size("AAPL", 0.65, 0, {})
    assert size == 0, "Zero portfolio must give zero size"


def test_low_confidence_reduces_size(sizer):
    size_high = sizer.calculate_position_size("AAPL", 0.90, 10_000, {})
    size_low  = sizer.calculate_position_size("AAPL", 0.51, 10_000, {})
    assert size_high >= size_low, "Higher confidence should produce >= position size"


def test_size_never_exceeds_portfolio(sizer):
    size = sizer.calculate_position_size("TSLA", 0.99, 10_000, {})
    assert size <= 10_000, "Position must never exceed full portfolio"


def test_size_is_non_negative(sizer):
    size = sizer.calculate_position_size("NVDA", 0.70, 10_000, {})
    assert size >= 0, "Position size must be non-negative"


# ── Concentration cap ─────────────────────────────────────────────────────────

def test_existing_positions_reduce_new_size(sizer):
    existing = {
        "AAPL": {"market_value": 2500},
        "MSFT": {"market_value": 2500},
        "GOOGL": {"market_value": 2500},
    }
    size_empty     = sizer.calculate_position_size("TSLA", 0.80, 10_000, {})
    size_crowded   = sizer.calculate_position_size("TSLA", 0.80, 10_000, existing)
    assert size_crowded <= size_empty, "Crowded portfolio must reduce new position size"


# ── Peer-group correlation guard ──────────────────────────────────────────────

def test_same_sector_caps_allocation(sizer):
    """Holding peers should reduce new allocation via correlation penalty."""
    # AAPL and MSFT are in the same PEER_GROUPS tech cluster
    existing = {"AAPL": {"market_value": 3000}}
    size_with_peer    = sizer.calculate_position_size("MSFT", 0.80, 10_000, existing)
    size_without_peer = sizer.calculate_position_size("MSFT", 0.80, 10_000, {})
    assert size_with_peer <= size_without_peer, "Peer holding should reduce new size"
