# backtesting/signals/library/price_momentum.py
"""
Price Momentum Signal (12-1 Momentum / Jegadeesh-Titman 1993)

Academic basis:
  Jegadeesh & Titman (1993), Asness (1994), Fama & French (1996).
  Stocks that outperformed over the past 12 months (excluding the most
  recent month) continue to outperform over the next 3-6 months.
  The 1-month exclusion avoids the short-term reversal effect.

  Expected IC: 0.04-0.07 (monthly resolution)
  Holding period: 20-60 trading days
  Rebalance: monthly

Formula:
  Momentum_score = Return(t-252, t-21) - market_return(same window)
  (market-adjusted to remove beta from the ranking)

  Cross-sectionally z-scored so top-N selection is scale-invariant.

Lookahead rules:
  - Only uses price data strictly before the scoring date
  - Rebalances monthly (not on every bar) to reduce turnover

Usage:
    from backtesting.signals.library.price_momentum import MomentumSignal
    signal = MomentumSignal(lookback_days=252, skip_days=21)
    scores = signal.compute(date, price_data)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class MomentumSignal(BaseSignal):
    """
    12-1 cross-sectional price momentum signal.

    Ranks stocks by their 12-month return (skipping the last month) relative
    to the cross-section. Top-N stocks are held for one month then re-ranked.

    Parameters
    ----------
    lookback_days : total lookback window in trading days (default 252 = 1 year)
    skip_days     : recent days to exclude to avoid reversal (default 21 = 1 month)
    min_history   : minimum days of price history required to score a stock
    market_adjust : if True, subtract equal-weight market return before ranking
                    (removes systematic beta from momentum signal)
    """

    name = 'price_momentum_12_1'

    def __init__(
        self,
        lookback_days : int  = 252,
        skip_days     : int  = 21,
        min_history   : int  = 280,
        market_adjust : bool = True,
    ):
        self.lookback_days  = lookback_days
        self.skip_days      = skip_days
        self.min_history    = min_history
        self.market_adjust  = market_adjust

    @property
    def holding_period_days(self) -> int:
        return 21   # 1-month hold

    @property
    def rebalance_freq(self) -> str:
        return 'M'  # monthly rebalance

    @property
    def min_history_days(self) -> int:
        return self.min_history

    def compute(
        self,
        date    : pd.Timestamp,
        data    : Dict[str, pd.DataFrame],
        earnings: Optional[Dict] = None,   # unused, kept for API compatibility
    ) -> Dict[str, float]:
        """
        Compute momentum scores for all symbols.

        Parameters
        ----------
        date : current backtesting date (only use prices strictly before this)
        data : {symbol: ohlcv_df}

        Returns
        -------
        {symbol: momentum_score} -- z-scored, higher = more bullish
        """
        raw_scores = {}

        for sym, df in data.items():
            past = df[df.index < date]
            if len(past) < self.min_history:
                continue

            close = past['close']
            n = len(close)
            if n < self.lookback_days + self.skip_days:
                continue

            # Formation period: t-252 to t-21 (skip last month)
            start_idx = n - self.lookback_days
            end_idx   = n - self.skip_days

            if start_idx < 0 or end_idx <= start_idx:
                continue

            price_start = float(close.iloc[start_idx])
            price_end   = float(close.iloc[end_idx])

            if price_start <= 0:
                continue

            raw_scores[sym] = (price_end / price_start) - 1.0

        if not raw_scores:
            return {}

        # Market-adjust: subtract equal-weight cross-sectional mean return
        if self.market_adjust and len(raw_scores) > 1:
            mkt_ret = float(np.mean(list(raw_scores.values())))
            raw_scores = {k: v - mkt_ret for k, v in raw_scores.items()}

        return self._z_score(raw_scores)

    @staticmethod
    def _z_score(scores: Dict[str, float]) -> Dict[str, float]:
        if len(scores) < 2:
            return scores
        vals = np.array(list(scores.values()))
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        if std < 1e-8:
            return {k: 0.0 for k in scores}
        return {k: float((v - mean) / std) for k, v in scores.items()}


class MomentumBacktest:
    """
    Convenience wrapper to run a full momentum backtest.

    Usage:
        from backtesting.signals.library.price_momentum import MomentumBacktest

        bt = MomentumBacktest()
        result = bt.run(
            symbols    = [...],
            start_date = '2020-01-01',
            end_date   = '2026-07-01',
        )
        result.print_summary()
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_positions  : int   = 10,
        lookback_days  : int   = 252,
        skip_days      : int   = 21,
        market_adjust  : bool  = True,
    ):
        from backtesting.data.loader         import DataLoader
        from backtesting.engine.event_driven import EventDrivenBacktester
        from backtesting.engine.fill_model   import FillModel
        from backtesting.engine.cost_model   import TransactionCostModel

        self.loader  = DataLoader()
        self.signal  = MomentumSignal(
            lookback_days = lookback_days,
            skip_days     = skip_days,
            market_adjust = market_adjust,
        )
        self.engine  = EventDrivenBacktester(
            initial_capital = initial_capital,
            fill_model      = FillModel(spread_bps=5.0, market_impact_factor=0.1),
            cost_model      = TransactionCostModel(),
        )
        self.max_positions = max_positions

    def run(
        self,
        symbols   : list,
        start_date: str,
        end_date  : str,
        spy_regime_filter: bool = True,
    ):
        from backtesting.analysis.metrics import performance_summary, print_summary

        logger.info('Loading OHLCV for %d symbols...', len(symbols))
        fetch_syms = list(set(symbols + ['SPY']))
        all_data   = self.loader.get_ohlcv(fetch_syms, '2017-01-01', end_date)
        price_data = {k: v for k, v in all_data.items() if k != 'SPY'}
        spy_prices = all_data.get('SPY')
        logger.info('Loaded %d symbols', len(price_data))

        # Pre-compute SPY 200d MA for regime filter
        spy_ma200 = None
        if spy_regime_filter and spy_prices is not None:
            spy_close = spy_prices['close']
            spy_ma200 = spy_close.rolling(200).mean()

        signal = self.signal

        def signal_fn(date, data):
            # Regime filter: skip when SPY below 200d MA
            if spy_ma200 is not None:
                avail = spy_ma200[spy_ma200.index < date]
                if len(avail) > 0:
                    ma_val = float(avail.iloc[-1])
                    spy_avail = spy_prices['close'][spy_prices.index < date]
                    if len(spy_avail) > 0 and not np.isnan(ma_val):
                        spy_val = float(spy_avail.iloc[-1])
                        if spy_val < ma_val:
                            return {}   # go to cash in bear market

            scores = signal.compute(date, data)
            if not scores:
                return {}
            n   = min(self.max_positions, len(scores))
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return {sym: 1.0 / n for sym, _ in top}

        logger.info('Running backtest %s to %s...', start_date, end_date)
        result = self.engine.run(
            price_data    = price_data,
            signal_fn     = signal_fn,
            start_date    = start_date,
            end_date      = end_date,
            max_positions = self.max_positions,
            rebalance_freq= self.signal.rebalance_freq,
        )

        eq      = result.equity_curve['equity']
        trades  = result.trades
        summary = performance_summary(eq, trades)
        print_summary(summary)
        return result
