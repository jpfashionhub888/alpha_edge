# backtesting/signals/library/sector_momentum.py
"""
Sector Rotation Momentum Signal

Academic basis:
  Moskowitz & Grinblatt (1999) -- "Do Industries Explain Momentum?"
  Sector momentum is stronger and more persistent than individual stock
  momentum because institutional rotation between sectors is slow.

  Expected IC: 0.05-0.12 monthly (much stronger than individual stock IC)
  Holding period: 1-3 months
  Rebalance: monthly

Universe: 11 SPDR Select Sector ETFs
  XLK  Technology
  XLF  Financials
  XLE  Energy
  XLV  Health Care
  XLI  Industrials
  XLY  Consumer Discretionary
  XLC  Communication Services
  XLP  Consumer Staples
  XLU  Utilities
  XLRE Real Estate
  XLB  Materials

Signal formula (12-1 momentum):
  score = Return(t-252, t-21) / volatility(t-63, t)
  Volatility-adjusted to prevent high-vol sectors dominating.
  Top-N sectors held equal-weight. SPY 200d MA regime gate.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.signals.base import BaseSignal

logger = logging.getLogger(__name__)

SECTOR_ETFS = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
    'XLY', 'XLC', 'XLP', 'XLU', 'XLRE', 'XLB',
]

SECTOR_NAMES = {
    'XLK' : 'Technology',
    'XLF' : 'Financials',
    'XLE' : 'Energy',
    'XLV' : 'Health Care',
    'XLI' : 'Industrials',
    'XLY' : 'Consumer Disc.',
    'XLC' : 'Communication',
    'XLP' : 'Consumer Staples',
    'XLU' : 'Utilities',
    'XLRE': 'Real Estate',
    'XLB' : 'Materials',
}


class SectorMomentumSignal(BaseSignal):
    """
    Volatility-adjusted 12-1 momentum on 11 SPDR sector ETFs.

    Parameters
    ----------
    lookback_days : return lookback window (default 252 = 1 year)
    skip_days     : skip most recent period to avoid reversal (default 21 = 1 month)
    vol_window    : window for volatility normalisation (default 63 = 1 quarter)
    vol_adjust    : if True, divide raw return by trailing volatility
    top_n         : number of sectors to hold (default 3)
    """

    name = 'sector_momentum_12_1'

    def __init__(
        self,
        lookback_days: int  = 252,
        skip_days    : int  = 21,
        vol_window   : int  = 63,
        vol_adjust   : bool = True,
        top_n        : int  = 3,
    ):
        self.lookback_days = lookback_days
        self.skip_days     = skip_days
        self.vol_window    = vol_window
        self.vol_adjust    = vol_adjust
        self.top_n         = top_n

    @property
    def holding_period_days(self) -> int:
        return 21

    @property
    def rebalance_freq(self) -> str:
        return 'M'

    @property
    def min_history_days(self) -> int:
        return self.lookback_days + self.skip_days + 20

    def compute(
        self,
        date    : pd.Timestamp,
        data    : Dict[str, pd.DataFrame],
        earnings: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Compute sector momentum scores.

        Parameters
        ----------
        date : scoring date (only use data strictly before this)
        data : {etf_ticker: ohlcv_df}

        Returns
        -------
        {ticker: score} z-scored, higher = more bullish
        """
        scores = {}

        for ticker, df in data.items():
            past = df[df.index < date]['close']
            n    = len(past)
            if n < self.lookback_days + self.skip_days + self.vol_window:
                continue

            # Formation period: t-252 to t-21
            p_start = float(past.iloc[n - self.lookback_days])
            p_end   = float(past.iloc[n - self.skip_days])
            if p_start <= 0:
                continue
            raw_ret = (p_end / p_start) - 1.0

            if self.vol_adjust:
                # Daily returns over last vol_window days
                recent = past.iloc[-self.vol_window:]
                daily  = recent.pct_change().dropna()
                vol    = float(daily.std()) * np.sqrt(252)
                if vol < 1e-6:
                    vol = 0.01
                scores[ticker] = raw_ret / vol
            else:
                scores[ticker] = raw_ret

        if len(scores) < 2:
            return {}

        return self._z_score(scores)

    def generate_weights(
        self,
        date    : pd.Timestamp,
        data    : Dict[str, pd.DataFrame],
        top_n   : Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Return equal weights for top-N sectors by momentum score."""
        scores = self.compute(date, data)
        if not scores:
            return {}
        n   = top_n or self.top_n
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return {sym: 1.0 / n for sym, _ in top}

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


class SectorMomentumBacktest:
    """
    Full backtest runner for sector rotation momentum.

    Usage:
        from backtesting.signals.library.sector_momentum import SectorMomentumBacktest

        bt = SectorMomentumBacktest(top_n=3)
        result = bt.run(start_date='2010-01-01', end_date='2026-07-01')
    """

    def __init__(
        self,
        initial_capital : float = 100_000,
        top_n           : int   = 3,
        lookback_days   : int   = 252,
        skip_days       : int   = 21,
        vol_adjust      : bool  = True,
    ):
        from backtesting.data.loader         import DataLoader
        from backtesting.engine.event_driven import EventDrivenBacktester
        from backtesting.engine.fill_model   import FillModel
        from backtesting.engine.cost_model   import TransactionCostModel

        self.loader  = DataLoader()
        self.signal  = SectorMomentumSignal(
            lookback_days = lookback_days,
            skip_days     = skip_days,
            vol_adjust    = vol_adjust,
            top_n         = top_n,
        )
        self.engine  = EventDrivenBacktester(
            initial_capital = initial_capital,
            fill_model      = FillModel(spread_bps=3.0, market_impact_factor=0.05),
            cost_model      = TransactionCostModel(commission_per_share=0.0),
        )
        self.top_n = top_n

    def run(
        self,
        start_date       : str,
        end_date         : str,
        spy_regime_filter: bool = True,
        print_holdings   : bool = True,
    ):
        from backtesting.analysis.metrics import performance_summary, print_summary

        fetch_syms = SECTOR_ETFS + ['SPY']
        logger.info('Loading sector ETFs + SPY...')
        all_data   = self.loader.get_ohlcv(fetch_syms, '2007-01-01', end_date)
        sector_data = {k: v for k, v in all_data.items() if k in SECTOR_ETFS}
        spy_prices  = all_data.get('SPY')
        logger.info('Loaded: %s', sorted(sector_data.keys()))

        spy_ma200 = None
        if spy_regime_filter and spy_prices is not None:
            spy_close = spy_prices['close']
            spy_ma200 = spy_close.rolling(200).mean()

        signal = self.signal
        top_n  = self.top_n

        def signal_fn(date, data):
            if spy_ma200 is not None:
                avail_ma  = spy_ma200[spy_ma200.index < date]
                avail_spy = spy_prices['close'][spy_prices.index < date]
                if len(avail_ma) > 0 and len(avail_spy) > 0:
                    ma_val  = float(avail_ma.iloc[-1])
                    spy_val = float(avail_spy.iloc[-1])
                    if not np.isnan(ma_val) and spy_val < ma_val:
                        return {}

            scores = signal.compute(date, data)
            if not scores:
                return {}
            n   = min(top_n, len(scores))
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            if print_holdings:
                names = ', '.join(
                    '%s(%s)' % (s, SECTOR_NAMES.get(s, s)) for s, _ in top
                )
                logger.debug('%s holdings: %s', date.date(), names)
            return {sym: 1.0 / n for sym, _ in top}

        logger.info('Running backtest %s to %s...', start_date, end_date)
        result = self.engine.run(
            price_data    = sector_data,
            signal_fn     = signal_fn,
            start_date    = start_date,
            end_date      = end_date,
            max_positions = top_n,
            rebalance_freq= self.signal.rebalance_freq,
        )

        eq     = result.equity_curve['equity']
        trades = result.trades
        summary = performance_summary(eq, trades)
        print_summary(summary)
        return result
