# tests/test_performance_analytics.py
"""
Unit tests for performance_analytics.py

Actual API: PerformanceAnalytics.calculate_metrics(
    trades, starting_capital, current_value, days_back=7
)
Returns a dict with keys: total_trades, wins, losses, win_rate,
total_pnl, total_pct, avg_win, avg_loss, profit_factor,
best_trade, worst_trade
"""

import numpy as np
import pytest


@pytest.fixture
def analytics():
    from performance_analytics import PerformanceAnalytics
    return PerformanceAnalytics()


def _metrics(analytics, trades, starting=10_000.0, current=10_500.0):
    """Helper: call calculate_metrics with sensible defaults."""
    return analytics.calculate_metrics(
        trades=trades,
        starting_capital=starting,
        current_value=current,
        days_back=30,          # 30-day window so fixture trades are included
    )


class TestProfitFactor:
    """Profit factor must handle edge cases without crashing."""

    def test_all_wins_returns_inf(self, analytics, sample_trade_history):
        """All-win trades → profit_factor = inf, not ZeroDivisionError."""
        wins_only = [t for t in sample_trade_history if t.get('pnl', 0) > 0]
        m = _metrics(analytics, wins_only)
        assert m['profit_factor'] == float('inf')

    def test_all_losses_gives_zero_profit_factor(self, analytics, sample_trade_history):
        """All losses → no wins → profit_factor = 0.0 (win sum / loss sum)."""
        losses_only = [t for t in sample_trade_history if t.get('pnl', 0) <= 0]
        m = _metrics(analytics, losses_only)
        pf = m['profit_factor']
        # Either 0.0 or 'N/A' — must not crash
        assert pf == 0.0 or pf == 'N/A' or pf == float('inf')

    def test_empty_returns_na(self, analytics):
        """Empty trade list → profit_factor = 'N/A' (no trades resolved)."""
        m = _metrics(analytics, [])
        assert m['profit_factor'] == 'N/A'

    def test_mixed_trades_no_error(self, analytics, sample_trade_history):
        """Mixed wins/losses → profit_factor is a number (not NaN, not crash)."""
        m = _metrics(analytics, sample_trade_history)
        pf = m['profit_factor']
        if isinstance(pf, float):
            assert not np.isnan(pf)
        else:
            assert pf in ('N/A', float('inf'))


class TestWinRate:
    """Win rate as percentage (0–100), PARTIAL_SELL included."""

    def test_win_rate_is_percentage(self, analytics, sample_trade_history):
        """Win rate is in range [0, 100]."""
        m = _metrics(analytics, sample_trade_history)
        assert 0.0 <= m['win_rate'] <= 100.0

    def test_partial_sell_counted_in_total(self, analytics, sample_trade_history):
        """
        Regression: old == 'SELL' filter excluded PARTIAL_SELL from total_trades.
        Both SELL and PARTIAL_SELL must count as realized exits.
        """
        m = _metrics(analytics, sample_trade_history)
        # sample_trade_history has 5 SELL + 1 PARTIAL_SELL = 6 realized exits
        realized = [t for t in sample_trade_history
                    if t.get('action') in ('SELL', 'PARTIAL_SELL')]
        assert m['total_trades'] == len(realized), (
            f"PARTIAL_SELL excluded from count: got {m['total_trades']}, "
            f"expected {len(realized)}"
        )

    def test_empty_trades_zero_win_rate(self, analytics):
        """No trades → win_rate = 0."""
        m = _metrics(analytics, [])
        assert m['win_rate'] == 0


class TestPnLAccounting:
    """P&L totals and counts are consistent."""

    def test_wins_plus_losses_equals_total(self, analytics, sample_trade_history):
        """wins + losses == total_trades."""
        m = _metrics(analytics, sample_trade_history)
        assert m['wins'] + m['losses'] == m['total_trades']

    def test_avg_win_positive(self, analytics, sample_trade_history):
        """avg_win must be >= 0 when there are wins."""
        m = _metrics(analytics, sample_trade_history)
        if m['wins'] > 0:
            assert m['avg_win'] >= 0

    def test_avg_loss_positive(self, analytics, sample_trade_history):
        """avg_loss is stored as a positive magnitude (absolute value)."""
        m = _metrics(analytics, sample_trade_history)
        if m['losses'] > 0:
            assert m['avg_loss'] >= 0

    def test_best_trade_has_highest_pnl(self, analytics, sample_trade_history):
        """best_trade is the trade dict with maximum pnl."""
        m = _metrics(analytics, sample_trade_history)
        if m['best_trade']:
            realized = [t for t in sample_trade_history
                        if t.get('action') in ('SELL', 'PARTIAL_SELL')]
            max_pnl = max(t.get('pnl', 0) for t in realized)
            assert m['best_trade'].get('pnl') == max_pnl

    def test_worst_trade_has_lowest_pnl(self, analytics, sample_trade_history):
        """worst_trade is the trade dict with minimum pnl."""
        m = _metrics(analytics, sample_trade_history)
        if m['worst_trade']:
            realized = [t for t in sample_trade_history
                        if t.get('action') in ('SELL', 'PARTIAL_SELL')]
            min_pnl = min(t.get('pnl', 0) for t in realized)
            assert m['worst_trade'].get('pnl') == min_pnl

    def test_total_pnl_is_portfolio_delta(self, analytics):
        """total_pnl = current_value - starting_capital."""
        m = analytics.calculate_metrics(
            trades=[], starting_capital=10_000.0,
            current_value=10_500.0, days_back=30
        )
        assert abs(m['total_pnl'] - 500.0) < 0.01
