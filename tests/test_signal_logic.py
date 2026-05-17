# tests/test_signal_logic.py
"""
Tests for compute_signal() — the function that turns model output
into BUY/HOLD/AVOID. These are pure-function tests with no I/O.

Run: pytest tests/test_signal_logic.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scanner import compute_signal, SIGNAL_WEIGHTS, SENTIMENT_DAMPENER


# ----- Basic signal generation ------------------------------------------ #

def test_uptrend_high_prediction_yields_buy():
    sig, combined = compute_signal(
        pred=0.70, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'BUY'
    assert 0.0 <= combined <= 1.0


def test_uptrend_low_prediction_yields_hold():
    sig, _ = compute_signal(
        pred=0.45, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'HOLD'


def test_downtrend_always_avoid():
    """Downtrend regime: AVOID regardless of how strong the prediction is."""
    for pred in [0.1, 0.5, 0.99]:
        sig, _ = compute_signal(
            pred=pred, regime='downtrend', sent_score=0.0,
            sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
        )
        assert sig == 'AVOID', f"pred={pred} should be AVOID in downtrend"


def test_volatile_regime_is_caution():
    sig, _ = compute_signal(
        pred=0.80, regime='volatile', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'CAUTION'


def test_sideways_regime_is_hold_even_with_high_prediction():
    """Per scanner V4 audit: sideways NEVER generates BUY."""
    sig, _ = compute_signal(
        pred=0.99, regime='sideways', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'HOLD'


# ----- Sector and earnings overrides ------------------------------------ #

def test_weak_sector_demotes_buy_to_hold():
    """A would-be BUY gets demoted if sector multiplier < 0.8."""
    sig, _ = compute_signal(
        pred=0.80, regime='uptrend', sent_score=0.0,
        sect_mult=0.7, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'HOLD'


def test_earnings_this_week_blocks_buy():
    sig, _ = compute_signal(
        pred=0.80, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=['AAPL']
    )
    assert sig == 'EARNINGS_HOLD'


def test_earnings_does_not_affect_avoid():
    """Earnings hold only applies to BUY — AVOID should still be AVOID."""
    sig, _ = compute_signal(
        pred=0.99, regime='downtrend', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=['AAPL']
    )
    assert sig == 'AVOID'


# ----- Combined score math ---------------------------------------------- #

def test_combined_clipped_to_zero_one():
    """Combined must always be in [0, 1] regardless of inputs."""
    for pred, sent, sect in [
        (-1.0, -2.0, 0.0),   # all bearish
        (2.0, 2.0, 5.0),     # all bullish
        (0.5, 0.0, 1.0),     # neutral
    ]:
        _, c = compute_signal(
            pred=pred, regime='uptrend', sent_score=sent,
            sect_mult=sect, symbol='X', earnings_symbols=[]
        )
        assert 0.0 <= c <= 1.0, f"combined {c} out of range"


def test_sentiment_does_not_dominate(monkeypatch=None):
    """
    H4 regression test: sentiment contribution must be small relative
    to prediction. With dampener=0.5 and weight=0.2, a max sentiment
    of ±1.0 contributes ±0.10 to combined.
    """
    _, c_with_max_pos_sent = compute_signal(
        pred=0.50, regime='uptrend', sent_score=1.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    _, c_with_max_neg_sent = compute_signal(
        pred=0.50, regime='uptrend', sent_score=-1.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    sentiment_swing = c_with_max_pos_sent - c_with_max_neg_sent

    # Max possible swing should be 2 * w_sent * SENTIMENT_DAMPENER
    max_swing = 2 * SIGNAL_WEIGHTS['sentiment'] * SENTIMENT_DAMPENER
    assert sentiment_swing <= max_swing + 1e-9, (
        f"Sentiment swung combined by {sentiment_swing:.3f}, "
        f"max should be {max_swing:.3f}"
    )

    # Should be smaller than the prediction contribution
    # (going from pred=0.5 to pred=1.0 changes combined by 0.5*0.6 = 0.30)
    pred_contribution_max = SIGNAL_WEIGHTS['prediction'] * 0.5
    assert sentiment_swing < pred_contribution_max, (
        f"Sentiment swing {sentiment_swing} >= max prediction "
        f"contribution {pred_contribution_max}"
    )


def test_buy_threshold_exact_boundary():
    """At exactly the BUY threshold (0.55), prediction >= triggers BUY."""
    sig_at, _ = compute_signal(
        pred=0.55, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    sig_below, _ = compute_signal(
        pred=0.549, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    assert sig_at == 'BUY'
    assert sig_below == 'HOLD'


# ----- Edge cases ------------------------------------------------------- #

def test_unknown_regime_yields_hold():
    """Unknown regime string should default to HOLD (not crash)."""
    sig, _ = compute_signal(
        pred=0.99, regime='moon_phase', sent_score=0.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    assert sig == 'HOLD'


def test_zero_prediction():
    """pred=0 must not break combined or signal generation."""
    sig, c = compute_signal(
        pred=0.0, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='X', earnings_symbols=[]
    )
    assert sig == 'HOLD'
    assert 0.0 <= c <= 1.0


def test_empty_earnings_list():
    """Empty earnings list is the common case; must not affect signal."""
    sig, _ = compute_signal(
        pred=0.80, regime='uptrend', sent_score=0.0,
        sect_mult=1.0, symbol='AAPL', earnings_symbols=[]
    )
    assert sig == 'BUY'
