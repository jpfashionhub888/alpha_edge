# tests/test_paper_trader.py
"""
Invariant tests for PaperTrader.

These tests cover the *accounting* layer — not strategy.
If any of these break, the system has produced wrong P&L
numbers and should not be trusted to trade.

Run: pytest tests/test_paper_trader.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable for the test
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution.paper_trader import PaperTrader


@pytest.fixture
def trader(tmp_path):
    """Fresh trader pointing at a temp log file."""
    log_file = tmp_path / "paper_trades.json"
    return PaperTrader(
        starting_capital=10_000.0,
        log_file=str(log_file),
    )


# ----- Reconciliation invariants ----------------------------------------- #

def test_reconcile_passes_on_fresh_trader(trader):
    """Fresh trader: cash=$10k, no positions, no trades → must reconcile."""
    trader.reconcile()  # no exception


def test_reconcile_after_single_buy(trader):
    """One BUY → cash drops by cost, position has cost, sums match."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)

    assert "AAPL" in trader.positions
    # capital + position cost should still equal $10,000
    # because nothing is realized yet
    pos_cost = trader.positions["AAPL"]["cost"]
    assert trader.capital + pos_cost == pytest.approx(10_000.0, abs=0.01)
    trader.reconcile()


def test_reconcile_after_buy_then_sell(trader):
    """Round-trip: invariant holds and PnL matches cash delta."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    initial_capital = trader.capital

    # Sell at 5% gain (round trip)
    trader.close_position("AAPL", price=105.0, reason="test")

    assert "AAPL" not in trader.positions
    # capital should now equal starting_capital + realized_pnl
    realized = sum(t.get("pnl", 0.0) for t in trader.trade_history
                   if t.get("action") in ("SELL", "PARTIAL_SELL"))
    assert trader.capital == pytest.approx(
        trader.starting_capital + realized, abs=0.01
    )
    trader.reconcile()


def test_reconcile_after_partial_exit(trader):
    """Partial exit must not break the invariant — this is the H5 bug check."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    # Trigger partial exit at +5%
    trader.update_position("AAPL", current_price=106.0)

    pos = trader.positions.get("AAPL")
    assert pos is not None, "Position should still exist after partial exit"
    assert pos["partial_exit_done"] is True
    assert pos["shares"] > 0

    # ── The critical assertion ────────────────────────────────────────
    # cost_per_share of remaining half should equal the original
    # cost_per_share. If H5 is regressed, this will fail.
    original_buy = next(
        t for t in trader.trade_history if t["action"] == "BUY"
    )
    original_cost_per_share = original_buy["cost"] / original_buy["shares"]
    remaining_cost_per_share = pos["cost"] / pos["shares"]
    assert remaining_cost_per_share == pytest.approx(
        original_cost_per_share, rel=1e-6
    ), (
        f"Cost-per-share drifted after partial exit "
        f"({remaining_cost_per_share} vs {original_cost_per_share}) — "
        f"H5 bug has regressed"
    )

    trader.reconcile()


def test_reconcile_fails_on_corrupted_state(trader):
    """If we manually corrupt state, reconcile() must crash."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)

    # Corrupt: spend $500 from cash that didn't come from a trade
    trader.capital -= 500.0

    with pytest.raises(AssertionError, match="reconciliation FAILED"):
        trader.reconcile()


# ----- Open-position guards ---------------------------------------------- #

def test_max_positions_enforced(trader):
    """Can't open more than max_positions."""
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        trader.open_position(sym, price=100.0, signal=0.85, atr=2.0)

    assert len(trader.positions) <= trader.max_positions
    # F should have been rejected
    assert "F" not in trader.positions


def test_duplicate_symbol_rejected(trader):
    """Opening the same symbol twice does nothing the second time."""
    ok1 = trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    shares_before = trader.positions["AAPL"]["shares"]

    ok2 = trader.open_position("AAPL", price=105.0, signal=0.85, atr=2.0)
    shares_after = trader.positions["AAPL"]["shares"]

    assert ok1 is True
    assert ok2 is False
    assert shares_before == shares_after


def test_zero_shares_rejected(trader):
    """If sizing produces zero shares, no position is opened."""
    # Trade a super-high-priced asset that can't fit in max_position_pct
    ok = trader.open_position(
        "BRK_A", price=500_000.0, signal=0.85, atr=10_000.0
    )
    assert ok is False
    assert "BRK_A" not in trader.positions


# ----- Stop-loss / take-profit ordering ---------------------------------- #

def test_stop_loss_fires_before_take_profit(trader):
    """If price has dropped past stop, stop fires; we don't see take-profit."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    # Price gaps down to -5%
    trader.update_position("AAPL", current_price=95.0)
    assert "AAPL" not in trader.positions
    last = trader.trade_history[-1]
    assert last["action"] == "SELL"
    assert last["reason"] == "stop_loss"


def test_take_profit_fires_at_8_pct(trader):
    """Take profit at +8%."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    trader.update_position("AAPL", current_price=108.5)
    assert "AAPL" not in trader.positions
    assert trader.trade_history[-1]["reason"] == "take_profit"


def test_trailing_stop_does_not_fire_below_activation(trader):
    """Trailing stop should not fire if max_gain < 2% activation."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    # Peak at 101 (below 2% activation), then drop 5%
    trader.update_position("AAPL", current_price=101.0)
    trader.update_position("AAPL", current_price=95.95)  # 5% drop from 101

    # 95.95 from 100 = -4% pnl which IS past the ATR stop (~5%),
    # so stop_loss may fire. That's fine; we're testing trailing did NOT.
    if "AAPL" not in trader.positions:
        # Whatever exit happened, it must not be a trailing stop
        assert trader.trade_history[-1]["reason"] != "trailing_stop"


# ----- State persistence ------------------------------------------------- #

def test_save_and_load_roundtrip(trader, tmp_path):
    """Save state, create a new trader, load state — positions match."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    trader.open_position("MSFT", price=200.0, signal=0.85, atr=4.0)

    log_file = trader.log_file

    # New trader pointing at same log
    other = PaperTrader(starting_capital=10_000.0, log_file=log_file)
    other.load_state()

    assert set(other.positions.keys()) == {"AAPL", "MSFT"}
    assert other.capital == pytest.approx(trader.capital, abs=0.01)
    other.reconcile()


def test_load_corrupted_json_does_not_crash(trader, tmp_path):
    """Corrupted JSON file: backup + start fresh, no exception."""
    log_file = trader.log_file
    with open(log_file, "w") as f:
        f.write("{not valid json")

    new_trader = PaperTrader(starting_capital=10_000.0, log_file=log_file)
    new_trader.load_state()

    assert new_trader.capital == 10_000.0
    assert new_trader.positions == {}
    assert os.path.exists(log_file + ".corrupted")


def test_load_state_with_failed_reconcile_resets_to_fresh(trader):
    """
    If saved state has bad accounting (e.g. someone edited the JSON),
    load_state should detect and start fresh rather than crash.
    """
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    # Don't go through save_state (which reconciles); write directly.
    state = {
        "capital": 5_000.0,             # wrong cash level
        "starting_capital": 10_000.0,
        "positions": trader.positions,  # only $98x in costs
        "trade_history": trader.trade_history,
        "daily_realized_pnl": 0.0,
        "daily_loss_limit": 500.0,
        "last_reset_date": "2026-01-01",
        "_halt_trading": False,
        "saved_at": "2026-01-01T00:00:00",
    }
    with open(trader.log_file, "w") as f:
        json.dump(state, f)

    new_trader = PaperTrader(
        starting_capital=10_000.0, log_file=trader.log_file
    )
    new_trader.load_state()
    # Should have reset to fresh because reconcile failed on load
    assert new_trader.capital == 10_000.0
    assert new_trader.positions == {}


def test_save_state_refuses_to_write_corrupt_state(trader):
    """If reconcile fails, save_state must raise — not silently persist bad state."""
    trader.open_position("AAPL", price=100.0, signal=0.85, atr=2.0)
    trader.capital -= 500.0  # corrupt
    with pytest.raises(AssertionError):
        trader.save_state()
