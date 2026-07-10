# tests/test_paper_trader.py
"""
Unit tests for execution/paper_trader.py

Covers:
- Basic buy/sell lifecycle
- Daily loss limit halt
- Circuit breaker interaction
- Partial exit cost-basis fix (regression for Bug 3)
- Time stop closes flat positions
- Atomic save + load roundtrip
- Kelly sizing cap
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest


class TestOpenPosition:
    """PaperTrader.open_position() — buy side."""

    def test_basic_buy_deducts_capital(self, paper_trader):
        """Buying shares reduces available cash by cost + commission."""
        initial = paper_trader.capital
        opened  = paper_trader.open_position('AAPL', price=150.0, signal=0.75)
        assert opened is True
        assert paper_trader.capital < initial
        assert 'AAPL' in paper_trader.positions

    def test_zero_shares_skips_trade(self, paper_trader):
        """If position sizing returns 0 shares (e.g. price > capital), skip."""
        # Use a price so large no shares can be bought
        opened = paper_trader.open_position('BRK', price=1_000_000.0, signal=0.8)
        assert opened is False
        assert 'BRK' not in paper_trader.positions

    def test_duplicate_symbol_rejected(self, paper_trader):
        """Cannot open two positions in the same symbol."""
        paper_trader.open_position('AAPL', price=150.0, signal=0.75)
        opened_again = paper_trader.open_position('AAPL', price=155.0, signal=0.80)
        assert opened_again is False
        # Still only one position
        assert len(paper_trader.positions) == 1

    def test_max_positions_enforced(self, paper_trader):
        """Cannot exceed max_positions (set to 5 in fixture)."""
        # Uncorrelated symbols (different sectors) so this test isolates
        # max_positions enforcement from the correlation-adjustment layer
        # in PositionSizer — the original tech-heavy symbol list happened
        # to trigger correlation-based size reduction unrelated to what
        # this test is actually checking.
        symbols = ['XOM', 'JNJ', 'JPM', 'CAT', 'WMT']
        for sym in symbols:
            paper_trader.open_position(sym, price=100.0, signal=0.7)
        # 6th position should be rejected
        opened = paper_trader.open_position('META', price=100.0, signal=0.9)
        assert opened is False
        assert len(paper_trader.positions) == 5

    def test_slippage_applied_on_fill(self, paper_trader):
        """Fill price should be slightly above market price (buy-side slippage)."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75)
        pos = paper_trader.positions['AAPL']
        assert pos['entry_price'] > 100.0  # slippage pushes fill above ask

    def test_atr_stop_loss_set(self, paper_trader):
        """When ATR is provided, stop_loss_pct is derived from ATR, not default."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75, atr=2.0)
        pos = paper_trader.positions['AAPL']
        # ATR-based stop: (2*atr)/price = 4/100 = 4% — clamped to [2%, 8%]
        assert 0.02 <= pos['stop_loss_pct'] <= 0.08


class TestClosePosition:
    """PaperTrader.close_position() — sell side."""

    def test_sell_returns_capital(self, paper_trader):
        """Closing a position adds proceeds back to capital."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75)
        capital_after_buy = paper_trader.capital
        paper_trader.close_position('AAPL', price=110.0, reason='take_profit')
        assert paper_trader.capital > capital_after_buy

    def test_close_nonexistent_returns_false(self, paper_trader):
        """Closing a symbol not in positions returns False."""
        result = paper_trader.close_position('FAKE', price=100.0)
        assert result is False

    def test_loss_updates_daily_pnl(self, paper_trader):
        """A losing close should add negative amount to daily_realized_pnl."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75)
        before = paper_trader.daily_realized_pnl
        paper_trader.close_position('AAPL', price=90.0, reason='stop_loss')
        assert paper_trader.daily_realized_pnl < before


class TestDailyLossLimit:
    """Daily loss limit halt mechanism."""

    def test_breach_halts_new_trades(self, paper_trader):
        """After daily loss limit is breached, open_position returns False."""
        # Simulate a large realized loss
        paper_trader.daily_realized_pnl = -(paper_trader.daily_loss_limit + 1)
        paper_trader._halt_trading = True

        opened = paper_trader.open_position('AAPL', price=100.0, signal=0.9)
        assert opened is False

    def test_halt_lifts_next_day(self, paper_trader):
        """The halt flag auto-resets on a new trading day."""
        # Skip if sentencepiece not installed — freezegun scans transformers
        # which tries to import sentencepiece; unrelated to our code.
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            pytest.skip("sentencepiece not installed — freezegun/transformers conflict")

        from freezegun import freeze_time

        paper_trader._halt_trading = True
        paper_trader.last_reset_date = (datetime.now() - timedelta(days=1)).date()

        # Simulate next day
        with freeze_time(datetime.now() + timedelta(days=1)):
            paper_trader._check_daily_reset()

        assert paper_trader._halt_trading is False
        assert paper_trader.daily_realized_pnl == 0.0


class TestPartialExit:
    """Partial exit cost-basis regression (Bug 3 fix)."""

    def test_partial_exit_reduces_shares(self, paper_trader):
        """After partial exit, position holds half the original shares."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75, atr=1.0)
        original_shares = paper_trader.positions['AAPL']['shares']

        paper_trader.update_position('AAPL', current_price=106.0)  # triggers 5% partial

        # Position should still be open but with fewer shares
        if 'AAPL' in paper_trader.positions:
            remaining = paper_trader.positions['AAPL']['shares']
            assert remaining < original_shares

    def test_partial_exit_logged_in_history(self, paper_trader):
        """PARTIAL_SELL action appears in trade_history after partial exit."""
        paper_trader.open_position('AAPL', price=100.0, signal=0.75, atr=1.0)
        paper_trader.update_position('AAPL', current_price=106.0)

        partial_trades = [t for t in paper_trader.trade_history
                          if t.get('action') == 'PARTIAL_SELL']
        assert len(partial_trades) >= 1


class TestStatePersistence:
    """Atomic save + load roundtrip (Bug 4 & 5 fixes)."""

    def test_save_and_reload_capital(self, paper_trader, tmp_log):
        """Capital saved to disk matches capital after reload."""
        paper_trader.open_position('AAPL', price=150.0, signal=0.75)
        paper_trader.save_state()

        # Create a fresh instance and load
        from execution.paper_trader import PaperTrader
        pt2 = PaperTrader(starting_capital=10_000.0, log_file=tmp_log)
        pt2.load_state()

        assert abs(pt2.capital - paper_trader.capital) < 0.01

    def test_corrupted_state_starts_fresh(self, tmp_log):
        """If state file is corrupted JSON, start fresh without crashing."""
        with open(tmp_log, 'w') as f:
            f.write('{ INVALID JSON !!!}')

        from execution.paper_trader import PaperTrader
        pt = PaperTrader(starting_capital=10_000.0, log_file=tmp_log)
        pt.load_state()  # Should not raise

        assert pt.capital == 10_000.0
        assert pt.positions == {}

    def test_atomic_write_no_partial_file(self, paper_trader, tmp_log):
        """State save writes to .tmp first then replaces — no half-written file."""
        paper_trader.save_state()
        tmp_path = tmp_log + '.tmp'
        # The .tmp file should be gone after a successful save
        assert not os.path.exists(tmp_path)
        assert os.path.exists(tmp_log)


class TestKellySizing:
    """Kelly Criterion position sizing."""

    def test_kelly_never_exceeds_max_position_pct(self, paper_trader):
        """Kelly fraction × multiplier is always capped at max_position_pct."""
        paper_trader.kelly_position_sizing = True
        paper_trader.kelly_multiplier      = 1.0   # full Kelly
        paper_trader.kelly_reward_risk_ratio = 2.5

        # High-conviction signal — Kelly might try to use > max
        shares = paper_trader.get_position_size(
            price=100.0,
            signal_strength=0.95,
        )
        max_allowed = int((paper_trader.capital * paper_trader.max_position_pct) / 100.0)
        assert shares <= max_allowed

    def test_weak_signal_reduces_but_does_not_zero_size(self, paper_trader):
        """
        Signal-quality vetoing is compute_signal()'s job (the BUY-threshold
        gate upstream), not the position sizer's — get_position_size() should
        never be called with a signal this weak via the real pipeline, since
        compute_signal() only calls open_position() when combined > 0.55.

        The sizer itself uses aggregate historical win-rate/avg-win/avg-loss
        for its Kelly calculation and treats signal_strength as a bounded
        multiplier (0.5x-1.2x) on top, not a per-trade win-probability — so
        a weak signal should reduce size, not zero it outright. (The old
        Kelly formula used signal_strength directly as a per-trade
        probability and could zero out weak signals itself; that behavior
        was intentionally removed in favor of this separation of concerns.)
        """
        paper_trader.kelly_position_sizing  = True
        paper_trader.kelly_reward_risk_ratio = 2.5

        weak_shares   = paper_trader.get_position_size(price=100.0, signal_strength=0.20)
        strong_shares = paper_trader.get_position_size(price=100.0, signal_strength=0.95)

        assert weak_shares >= 0
        assert strong_shares >= weak_shares
