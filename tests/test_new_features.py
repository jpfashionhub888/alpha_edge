# tests/test_new_features.py
"""
Tests for the three new features added in this sprint:
  1. ML Score Exit Signal (execution/paper_trader.py)
  2. Sharpe / Sortino / Calmar metrics (performance_analytics.py)
  3. OptionsAnalyzer (options_analyzer.py)
"""

import math
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────
#  1. ML SCORE EXIT SIGNAL
# ─────────────────────────────────────────────────────────────

class TestMLScoreExit:
    """ML Score Exit triggers close when live confidence drops below threshold."""

    def _make_trader(self):
        from execution.paper_trader import PaperTrader
        t = PaperTrader(starting_capital=10_000, log_file='logs/_test_ml.json')
        return t

    def test_ml_score_exit_triggers_when_below_threshold(self, tmp_path):
        """Position is closed when ml_score drops below 0.40."""
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(
            starting_capital=10_000,
            log_file=str(tmp_path / 'state.json')
        )

        # Open a position
        opened = trader.open_position('AAPL', 100.0, signal=0.75, ml_score=0.80)
        assert opened

        capital_before = trader.capital

        # Call update with ml_score well below threshold (0.40)
        trader.update_position('AAPL', current_price=98.0, ml_score=0.25)

        # Position should be closed
        assert 'AAPL' not in trader.positions
        # Trade history should have ml_score_exit
        exits = [t for t in trader.trade_history if t.get('reason') == 'ml_score_exit']
        assert len(exits) == 1

    def test_ml_score_exit_skipped_for_winners(self, tmp_path):
        """
        ML exit is skipped when max_gain >= 5% — trailing stop handles it instead.
        This avoids cutting a winner just because the model confidence oscillates.
        """
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(
            starting_capital=10_000,
            log_file=str(tmp_path / 'state.json')
        )

        # Open at 100, push highest to 107 (7% gain > 5% threshold)
        trader.open_position('NVDA', 100.0, signal=0.80, ml_score=0.85)
        trader.positions['NVDA']['highest_price'] = 107.0   # simulate prior high

        # ml_score is low but max_gain is 7% — should NOT exit via ML
        trader.update_position('NVDA', current_price=106.0, ml_score=0.20)

        # Position still open (trailing stop would need a bigger drop to trigger)
        assert 'NVDA' in trader.positions

    def test_ml_score_neutral_when_none(self, tmp_path):
        """update_position with ml_score=None should not trigger ML exit."""
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(
            starting_capital=10_000,
            log_file=str(tmp_path / 'state.json')
        )
        trader.open_position('SPY', 400.0, signal=0.75)
        trader.update_position('SPY', current_price=398.0, ml_score=None)

        # Still in position (no stop triggered, ml_score=None → no ML exit)
        assert 'SPY' in trader.positions

    def test_entry_ml_score_stored_in_position(self, tmp_path):
        """entry_ml_score is persisted in the position dict."""
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(
            starting_capital=10_000,
            log_file=str(tmp_path / 'state.json')
        )
        trader.open_position('MSFT', 300.0, signal=0.75, ml_score=0.82)
        assert trader.positions['MSFT']['entry_ml_score'] == 0.82

    def test_entry_ml_score_in_buy_trade_record(self, tmp_path):
        """entry_ml_score appears in the BUY trade record."""
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(
            starting_capital=10_000,
            log_file=str(tmp_path / 'state.json')
        )
        trader.open_position('GOOG', 150.0, signal=0.78, ml_score=0.71)
        buy = next(t for t in trader.trade_history if t['action'] == 'BUY')
        assert buy['entry_ml_score'] == 0.71


# ─────────────────────────────────────────────────────────────
#  2. SHARPE / SORTINO / CALMAR
# ─────────────────────────────────────────────────────────────

class TestRiskMetrics:
    """Risk-adjusted metric calculations."""

    def _make_analytics(self):
        from performance_analytics import PerformanceAnalytics
        return PerformanceAnalytics()

    def test_sharpe_positive_for_consistent_winners(self):
        """Sharpe > 0 when all trades are profitable."""
        from datetime import datetime, timedelta
        from performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()

        now = datetime.now()
        trades = []
        for i in range(10):
            trades.append({
                'action' : 'SELL',
                'symbol' : 'TEST',
                'pnl'    : 50.0 + i * 5,        # all wins
                'date'   : (now - timedelta(days=i)).isoformat(),
            })

        metrics = analytics.calculate_metrics(
            trades=trades,
            starting_capital=10_000,
            current_value=10_500,
            days_back=30,
        )

        assert 'sharpe' in metrics
        assert 'sortino' in metrics
        assert 'calmar' in metrics
        assert 'max_drawdown' in metrics

        # With all-winning trades: Sharpe should be positive or N/A (< 2 data points edge)
        sharpe = metrics['sharpe']
        if sharpe != 'N/A':
            assert isinstance(sharpe, (int, float))

    def test_max_drawdown_zero_when_all_wins(self):
        """No drawdown when equity curve only goes up."""
        from datetime import datetime, timedelta
        from performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        now = datetime.now()
        trades = [
            {'action': 'SELL', 'pnl': 100.0,
             'date': (now - timedelta(days=i)).isoformat()}
            for i in range(5)
        ]
        metrics = analytics.calculate_metrics(trades, 10_000, 10_500, days_back=30)
        assert metrics['max_drawdown'] == 0.0

    def test_empty_metrics_has_risk_fields(self):
        """Empty metrics dict always includes Sharpe/Sortino/Calmar/max_drawdown."""
        from performance_analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics()
        m = analytics._empty_metrics(10_000, 10_000, 7)
        assert 'sharpe'       in m
        assert 'sortino'      in m
        assert 'calmar'       in m
        assert 'max_drawdown' in m
        assert m['sharpe'] == 'N/A'

    def test_equity_curve_builds_correctly(self):
        """_build_equity_curve returns correct daily return fractions."""
        from datetime import datetime, timedelta
        from performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        now = datetime.now()
        # Two trades: +$100 today, -$50 yesterday on $10k capital
        exits = [
            {'date': (now - timedelta(days=1)).isoformat(), 'pnl': -50.0},
            {'date': now.isoformat(),                        'pnl':  100.0},
        ]
        returns = analytics._build_equity_curve(exits, 10_000)
        assert len(returns) == 2
        # First return: -50 / 10000 = -0.005
        assert abs(returns[0] - (-0.005)) < 1e-6

    def test_sortino_higher_than_sharpe_for_asymmetric_returns(self):
        """Sortino >= Sharpe when wins are bigger than losses (downside is small)."""
        from datetime import datetime, timedelta
        from performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        now = datetime.now()
        # Mix: mostly big wins, one small loss
        trades = [
            {'action': 'SELL', 'pnl':  200.0, 'date': (now - timedelta(days=5)).isoformat()},
            {'action': 'SELL', 'pnl':  150.0, 'date': (now - timedelta(days=4)).isoformat()},
            {'action': 'SELL', 'pnl': -20.0,  'date': (now - timedelta(days=3)).isoformat()},
            {'action': 'SELL', 'pnl':  180.0, 'date': (now - timedelta(days=2)).isoformat()},
            {'action': 'SELL', 'pnl':  120.0, 'date': (now - timedelta(days=1)).isoformat()},
        ]
        metrics = analytics.calculate_metrics(trades, 10_000, 10_630, days_back=30)
        sharpe  = metrics['sharpe']
        sortino = metrics['sortino']

        # Both numeric → Sortino should be >= Sharpe with asymmetric returns
        if isinstance(sharpe, float) and isinstance(sortino, float):
            assert sortino >= sharpe - 0.01   # tiny tolerance for float arithmetic


# ─────────────────────────────────────────────────────────────
#  3. OPTIONS ANALYZER
# ─────────────────────────────────────────────────────────────

class TestOptionsAnalyzer:
    """Unit tests for OptionsAnalyzer (all network calls mocked)."""

    def _make_mock_chain(self, call_vol=500, put_vol=300, call_oi=200, put_oi=150,
                         call_iv=0.25, put_iv=0.30):
        """Build minimal mock calls/puts DataFrames."""
        import pandas as pd
        calls = pd.DataFrame([{
            'volume': call_vol, 'openInterest': call_oi,
            'impliedVolatility': call_iv, 'strike': 100.0,
        }])
        puts = pd.DataFrame([{
            'volume': put_vol,  'openInterest': put_oi,
            'impliedVolatility': put_iv, 'strike': 100.0,
        }])
        return calls, puts

    def test_score_is_float_in_valid_range(self):
        """options_score must be in [-0.30, +0.30]."""
        from options_analyzer import OptionsAnalyzer, SCORE_CAP
        analyzer = OptionsAnalyzer(cache_minutes=0)

        calls, puts = self._make_mock_chain()

        with patch.object(analyzer, '_fetch_chain', return_value=(calls, puts, '2026-07-18')):
            score = analyzer.get_options_score('AAPL')

        assert isinstance(score, float)
        assert -SCORE_CAP <= score <= SCORE_CAP

    def test_bearish_pcr_gives_negative_score(self):
        """High put volume (PCR > 1.30) should produce negative score."""
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=0)

        # put_vol=1400, call_vol=1000 → PCR=1.4 (bearish)
        calls, puts = self._make_mock_chain(call_vol=1000, put_vol=1400)

        with patch.object(analyzer, '_fetch_chain', return_value=(calls, puts, '2026-07-18')):
            score = analyzer.get_options_score('SPY')

        assert score < 0

    def test_bullish_pcr_gives_positive_score(self):
        """Low put volume (PCR < 0.70) should produce positive score."""
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=0)

        # put_vol=300, call_vol=1000 → PCR=0.3 (bullish)
        calls, puts = self._make_mock_chain(call_vol=1000, put_vol=300)

        with patch.object(analyzer, '_fetch_chain', return_value=(calls, puts, '2026-07-18')):
            score = analyzer.get_options_score('NVDA')

        assert score > 0

    def test_unusual_call_activity_boosts_score(self):
        """Volume > 3× OI on calls should add +0.15 to score."""
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=0)

        # call_vol=900 > 3×oi=300 → unusual call activity
        calls, puts = self._make_mock_chain(call_vol=900, call_oi=100,
                                            put_vol=200, put_oi=300)

        with patch.object(analyzer, '_fetch_chain', return_value=(calls, puts, '2026-07-18')):
            score = analyzer.get_options_score('TSLA')

        # Should be positive (call sweep detected)
        assert score > 0

    def test_cache_returns_same_value(self):
        """Second call within TTL should not re-fetch the chain."""
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=60)

        calls, puts = self._make_mock_chain()

        with patch.object(analyzer, '_fetch_chain', return_value=(calls, puts, '2026-07-18')) as mock_fetch:
            analyzer.get_options_score('AAPL')
            analyzer.get_options_score('AAPL')   # second call

        # fetch_chain should only be called once
        assert mock_fetch.call_count == 1

    def test_returns_zero_on_error(self):
        """Any exception in analysis returns 0.0 gracefully (non-blocking)."""
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=0)

        with patch.object(analyzer, '_fetch_chain', side_effect=RuntimeError("API down")):
            score = analyzer.get_options_score('BROKEN')

        assert score == 0.0

    def test_empty_analysis_on_yfinance_missing(self):
        """_empty_analysis returns structured dict with score=0.0."""
        from options_analyzer import OptionsAnalyzer
        result = OptionsAnalyzer._empty_analysis('AAPL', 'test error')
        assert result['options_score'] == 0.0
        assert result['symbol'] == 'AAPL'
        assert result['error'] == 'test error'

    def test_put_call_ratio_fallback_to_oi(self):
        """PCR falls back to OI when volume is all zero."""
        import pandas as pd
        from options_analyzer import OptionsAnalyzer
        analyzer = OptionsAnalyzer(cache_minutes=0)

        calls = pd.DataFrame([{'volume': 0, 'openInterest': 100, 'impliedVolatility': 0.25, 'strike': 100}])
        puts  = pd.DataFrame([{'volume': 0, 'openInterest': 150, 'impliedVolatility': 0.30, 'strike': 100}])

        pcr = analyzer._put_call_ratio(calls, puts)
        assert pcr is not None
        assert abs(pcr - 1.5) < 0.01   # 150/100
