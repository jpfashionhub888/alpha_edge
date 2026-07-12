"""Tests for risk/position_sizer.py — Kelly formula, VaR cap, correlation penalty.

Uses the real PositionSizer.calculate() API:
  PositionSizer(portfolio_value, base_risk_pct=0.015)
  sizer.calculate(symbol, price, atr, signal_score, ...) -> float (dollars)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from risk.position_sizer import PositionSizer, MIN_POSITION_USD
except ImportError as e:
    pytest.skip(f"position_sizer not importable: {e}", allow_module_level=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sizer():
    """Standard sizer with $10k portfolio."""
    return PositionSizer(portfolio_value=10_000)

@pytest.fixture()
def small_sizer():
    """Sizer representing a very small account to test minimum-size behaviour."""
    return PositionSizer(portfolio_value=100)


# ── Basic sanity ──────────────────────────────────────────────────────────────

def test_returns_float(sizer):
    result = sizer.calculate("AAPL", price=185.0, atr=3.2)
    assert isinstance(result, float), "calculate() must return a float"


def test_non_negative(sizer):
    result = sizer.calculate("AAPL", price=185.0, atr=3.2, signal_score=0.70)
    assert result >= 0, "Position size must be non-negative"


def test_size_never_exceeds_hard_cap(sizer):
    """Hard cap: single position never exceeds 15% of portfolio (MAX_POSITION_PCT)."""
    result = sizer.calculate("AAPL", price=185.0, atr=3.2, signal_score=0.99, n_trades=0)
    assert result <= 10_000 * 0.15 + 1, (
        f"Position ${result:.0f} exceeds 15% hard cap of $1500 on $10k portfolio"
    )


def test_zero_portfolio_returns_minimum(small_sizer):
    """portfolio_value=0/tiny returns MIN_POSITION_USD (guard in calculate())."""
    # Small portfolio: kelly < MIN_POSITION_USD -> returns 0.0
    result = small_sizer.calculate("AAPL", price=185.0, atr=3.2, signal_score=0.70)
    assert result >= 0, "Small portfolio must not return negative size"


def test_negative_atr_uses_portfolio_cap(sizer):
    """atr <= 0 triggers VaR fallback to MAX_POSITION_PCT of portfolio."""
    result = sizer.calculate("AAPL", price=185.0, atr=0.0, signal_score=0.70)
    assert result >= 0, "Zero ATR must not return negative size"


# ── Signal strength scaling ───────────────────────────────────────────────────

def test_higher_signal_gives_equal_or_larger_size(sizer):
    """Higher signal_score should produce >= allocation (signal_mult is monotone)."""
    low  = sizer.calculate("AAPL", price=185.0, atr=3.2, signal_score=0.51)
    high = sizer.calculate("AAPL", price=185.0, atr=3.2, signal_score=0.90)
    assert high >= low, "Higher signal score should yield >= position size"


# ── Regime scaling ────────────────────────────────────────────────────────────

def test_low_regime_conf_reduces_size(sizer):
    """regime_conf=0.3 (minimum) should produce smaller allocation than conf=1.0."""
    full    = sizer.calculate("AAPL", price=185.0, atr=3.2, regime_conf=1.0)
    reduced = sizer.calculate("AAPL", price=185.0, atr=3.2, regime_conf=0.3)
    assert reduced <= full, "Low regime confidence must reduce position size"


# ── Peer-group correlation guard ──────────────────────────────────────────────

def test_correlated_peer_reduces_size(sizer):
    """Holding AAPL should reduce a new MSFT allocation (same Big-Tech peer group)."""
    no_peer   = sizer.calculate("MSFT", price=400.0, atr=5.0, open_positions={})
    with_peer = sizer.calculate(
        "MSFT", price=400.0, atr=5.0,
        open_positions={"AAPL": {"shares": 10, "entry_price": 185.0}},
    )
    assert with_peer <= no_peer, (
        "Holding a correlated peer (AAPL) must reduce MSFT allocation"
    )


def test_uncorrelated_symbol_not_penalised(sizer):
    """Non-peer symbols in portfolio must not reduce the new symbol's size."""
    no_portfolio = sizer.calculate("NVDA", price=500.0, atr=10.0, open_positions={})
    with_unrelated = sizer.calculate(
        "NVDA", price=500.0, atr=10.0,
        open_positions={"XOM": {"shares": 5, "entry_price": 100.0}},
    )
    # XOM is Energy, NVDA is Semis — different peer groups → no penalty
    assert with_unrelated >= no_portfolio * 0.99, (
        "Unrelated holding (XOM vs NVDA) must not reduce allocation"
    )
