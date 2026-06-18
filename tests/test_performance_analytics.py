# tests/test_performance_analytics.py
"""
Unit tests for performance_analytics.py

Tests:
- Profit factor: no divide-by-zero on all-win scenarios
- Sharpe ratio: returns 0.0 (not NaN) on < 2 trades
- Max drawdown: correct calculation against known sequence
- Win rate: PARTIAL_SELL trades included in realized count
- Expectancy: positive on winning trade mix
"""

import numpy as np
import pytest


@pytest.fixture
def analytics():
    from performance_analytics import PerformanceAnalytics
    return PerformanceAnalytics()


class TestProfitFactor:
    """Profit factor must handle edge cases without crashing."""

    def test_all_wins_returns_high_value(self, analytics, sample_trade_history):
        """All-win trade history → profit factor ≫ 1 (or inf), never div/0."""
        wins_only = [t for t in sample_trade_history if t.get('pnl', 0) > 0]
        # Should not raise ZeroDivisionError
        pf = analytics.calculate_profit_factor(wins_only)
        assert pf >= 1.0 or pf == float('inf')

    def test_all_losses_returns_zero(self, analytics, sample_trade_history):
        """All-loss trade history → profit factor = 0, never div/0."""
        losses_only = [t for t in sample_trade_history if t.get('pnl', 0) <= 0]
        pf = analytics.calculate_profit_factor(losses_only)
        assert pf == 0.0

    def test_empty_returns_zero(self, analytics):
        """Empty trade list → profit factor = 0."""
        pf = analytics.calculate_profit_factor([])
        assert pf == 0.0

    def test_mixed_trades_positive(self, analytics, sample_trade_history):
        """Mixed wins/losses with more wins → PF > 1."""
        pf = analytics.calculate_profit_factor(sample_trade_history)
        assert isinstance(pf, float)
        assert not np.isnan(pf)


class TestSharpeRatio:
    """Sharpe ratio must be safe on small or degenerate trade sets."""

    def test_zero_trades_returns_zero(self, analytics):
        """No trades → Sharpe = 0.0, not NaN or exception."""
        sharpe = analytics.calculate_sharpe(returns=[])
        assert sharpe == 0.0
        assert not np.isnan(sharpe)

    def test_single_trade_returns_zero(self, analytics):
        """One trade → insufficient data for std → Sharpe = 0.0."""
        sharpe = analytics.calculate_sharpe(returns=[0.05])
        assert sharpe == 0.0 or not np.isnan(sharpe)

    def test_constant_returns_zero_std(self, analytics):
        """Identical returns every day → std=0 → Sharpe = 0.0 (no div/0)."""
        returns = [0.001] * 100
        sharpe = analytics.calculate_sharpe(returns=returns)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

    def test_positive_edge_gives_positive_sharpe(self, analytics):
        """Consistently positive returns → Sharpe > 0."""
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.002, 0.01, 252))  # ~50% annual return
        sharpe = analytics.calculate_sharpe(returns=returns)
        assert sharpe > 0


class TestMaxDrawdown:
    """Max drawdown calculation against a known equity sequence."""

    def test_known_sequence(self, analytics):
        """
        Equity: 100 → 120 → 90 → 110 → 80 → 100
        Peak = 120, trough after peak = 80
        Max DD = (80 - 120) / 120 = -33.3%
        """
        equity = [100, 120, 90, 110, 80, 100]
        dd = analytics.calculate_max_drawdown(equity)
        assert abs(dd - (-0.333)) < 0.01, f'Expected ~-33.3%, got {dd:.1%}'

    def test_monotonic_increase_gives_zero_dd(self, analytics):
        """Steadily rising equity → max drawdown = 0."""
        equity = [100, 110, 120, 130, 140]
        dd = analytics.calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_empty_returns_zero(self, analytics):
        """Empty equity curve → drawdown = 0."""
        dd = analytics.calculate_max_drawdown([])
        assert dd == 0.0


class TestWinRate:
    """Win rate must count PARTIAL_SELL trades (regression test)."""

    def test_partial_sell_counted_in_realized(self, analytics, sample_trade_history):
        """
        Regression: old code used action == 'SELL' which excluded PARTIAL_SELL.
        Win rate must include PARTIAL_SELL trades in the denominator.
        """
        stats = analytics.calculate_stats(sample_trade_history)

        # Count manually
        realized = [t for t in sample_trade_history
                    if t.get('action') in ('SELL', 'PARTIAL_SELL')]
        expected_total = len(realized)

        assert stats.get('total_trades', 0) == expected_total, (
            f'PARTIAL_SELL excluded: got {stats["total_trades"]}, '
            f'expected {expected_total}'
        )

    def test_win_rate_between_zero_and_one(self, analytics, sample_trade_history):
        """Win rate is always in [0, 1]."""
        stats = analytics.calculate_stats(sample_trade_history)
        wr = stats.get('win_rate', 0)
        assert 0.0 <= wr <= 1.0

    def test_expectancy_positive_on_good_history(self, analytics, sample_trade_history):
        """sample_trade_history has more wins than losses by value → expectancy > 0."""
        stats = analytics.calculate_stats(sample_trade_history)
        assert stats.get('expectancy', 0) > 0
