# backtesting/signals/library/earnings_revision.py
"""
Earnings Revision Momentum Signal (SUE — Standardised Unexpected Earnings)

Academic basis:
  Ball & Brown (1968), Foster, Olsen & Shevlin (1984),
  Bernard & Thomas (1989) — post-earnings announcement drift (PEAD).

  Stocks with positive earnings surprises continue to outperform for
  4-8 weeks after the announcement. The market under-reacts to earnings
  news, creating a systematic drift.

  Expected IC: 0.04–0.08 (strong for daily-resolution signals)
  Holding period: 1–4 weeks
  Rebalance: weekly

Formula:
  SUE = (Actual_EPS - Consensus_EPS) / σ(EPS_revisions)

  Where σ(EPS_revisions) = standard deviation of analyst estimate revisions
  over the trailing 8 quarters (normalises for scale).

  When point-in-time consensus data is unavailable (e.g. yfinance), we
  use a simplified version:
    SUE_simplified = (Actual_EPS - Mean_Trailing_EPS) / σ(Trailing_EPS)

Lookahead rules enforced:
  - Only earnings announcements with date < current backtesting date are used
  - Forward returns computed at T+1 (one day after signal)
  - No future EPS actuals used to estimate consensus

Usage:
    from backtesting.signals.library.earnings_revision import EarningsRevisionSignal

    signal = EarningsRevisionSignal(lookback_quarters=8, decay_days=20)
    scores = signal.compute(date, data_with_earnings)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class EarningsRevisionSignal(BaseSignal):
    """
    SUE-based post-earnings announcement drift signal.

    Ranks stocks by their most recent earnings surprise relative to
    historical EPS volatility. Trades on continuation of the surprise
    (positive surprise → long, negative surprise → short).

    Parameters
    ----------
    lookback_quarters : number of past quarters used to compute EPS std dev
    decay_days        : signal decays to zero after this many days post-announcement
                        (prevents trading stale earnings)
    min_quarters      : minimum quarters of history required to score a stock
    """

    name = 'earnings_revision_momentum'

    def __init__(
        self,
        lookback_quarters: int  = 8,
        decay_days       : int  = 20,    # trade within 20 days of announcement
        min_quarters     : int  = 4,
    ):
        self.lookback_quarters = lookback_quarters
        self.decay_days        = decay_days
        self.min_quarters      = min_quarters

    @property
    def holding_period_days(self) -> int:
        return 10   # 2-week average hold

    @property
    def rebalance_freq(self) -> str:
        return 'W'  # weekly re-rank

    @property
    def min_history_days(self) -> int:
        return 90   # need ~2 years of quarterly earnings

    def compute(
        self,
        date    : pd.Timestamp,
        data    : Dict[str, pd.DataFrame],
        earnings: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """
        Compute SUE scores for all symbols.

        Parameters
        ----------
        date     : current backtesting date (only use data BEFORE this)
        data     : {symbol: ohlcv_df} (standard engine format)
        earnings : {symbol: earnings_df} with columns [actual_eps, estimated_eps]
                   If None, falls back to price-based momentum proxy

        Returns
        -------
        {symbol: sue_score} — z-scored, higher = more bullish
        """
        scores = {}

        if earnings is None:
            # Fallback: price-based momentum as SUE proxy
            # (earnings data not provided — use 20-day momentum residual)
            logger.debug('EarningsRevisionSignal: no earnings data — using momentum proxy')
            return self._compute_momentum_proxy(date, data)

        for sym, earn_df in earnings.items():
            if sym not in data:
                continue

            try:
                score = self._compute_sue_score(sym, earn_df, date)
                if score is not None:
                    scores[sym] = score
            except Exception as e:
                logger.debug(f'{sym}: SUE computation failed ({e})')

        if not scores:
            logger.debug(f'{date}: no SUE scores computed')
            return {}

        # Cross-sectional z-score (rank normalisation)
        scores = self._z_score(scores)
        return scores

    def _compute_sue_score(
        self,
        symbol  : str,
        earn_df : pd.DataFrame,
        date    : pd.Timestamp,
    ) -> Optional[float]:
        """
        Compute SUE for a single symbol as of `date`.

        Only use earnings announced strictly BEFORE `date`.
        """
        # Filter to past announcements only (enforce no lookahead)
        past = earn_df[earn_df.index < date].copy()

        if len(past) < self.min_quarters:
            return None

        # Most recent announcement
        latest = past.iloc[-1]
        days_since = (date - past.index[-1]).days

        # Only trade within decay_days of the announcement
        if days_since > self.decay_days:
            return None

        actual    = float(latest.get('actual_eps', np.nan))
        estimated = float(latest.get('estimated_eps', np.nan))

        if np.isnan(actual) or np.isnan(estimated):
            # Use simplified SUE: actual vs trailing mean
            actual_series = past['actual_eps'].dropna()
            if len(actual_series) < self.min_quarters:
                return None
            trailing      = actual_series.iloc[:-1]
            if len(trailing) < 2:
                return None
            surprise      = actual - trailing.mean()
            eps_std       = trailing.std()
        else:
            surprise  = actual - estimated
            # Use trailing EPS standard deviation to normalise
            lookback  = past.iloc[-self.lookback_quarters:]
            actual_ser = lookback['actual_eps'].dropna()
            if len(actual_ser) < 2:
                eps_std = abs(estimated) * 0.1 if abs(estimated) > 0.01 else 0.1
            else:
                eps_std = float(actual_ser.std())

        if eps_std < 1e-6:
            eps_std = 0.1   # avoid division by zero for very stable earners

        sue = surprise / eps_std

        # Time-decay weight: signal is strongest right after announcement
        decay_weight = max(0, 1 - days_since / self.decay_days)

        return float(sue * decay_weight)

    def _compute_momentum_proxy(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Price-momentum proxy for when earnings data is unavailable.

        Uses 60-day price return minus 5-day return (avoids short-term reversal).
        This is NOT a substitute for earnings data but allows the framework
        to be tested before earnings data is connected.
        """
        scores = {}
        for sym, df in data.items():
            past = df[df.index < date]
            if len(past) < 65:
                continue
            close = past['close']
            ret_60 = float(close.iloc[-1] / close.iloc[-60] - 1)
            ret_5  = float(close.iloc[-1] / close.iloc[-5] - 1)
            scores[sym] = ret_60 - ret_5   # medium-term momentum, skip reversal

        return self._z_score(scores)

    @staticmethod
    def _z_score(scores: Dict[str, float]) -> Dict[str, float]:
        """Cross-sectional z-score normalisation."""
        if len(scores) < 2:
            return scores
        vals  = np.array(list(scores.values()))
        mean  = float(np.mean(vals))
        std   = float(np.std(vals))
        if std < 1e-8:
            return {k: 0.0 for k in scores}
        return {k: float((v - mean) / std) for k, v in scores.items()}


class EarningsRevisionBacktest:
    """
    Convenience wrapper to run a full earnings revision backtest.

    Usage:
        from backtesting.signals.library.earnings_revision import EarningsRevisionBacktest

        bt = EarningsRevisionBacktest()
        result = bt.run(
            symbols    = ['AAPL', 'MSFT', 'GOOGL', ...],
            start_date = '2020-01-01',
            end_date   = '2024-12-31',
        )
        result.print_summary()
    """

    def __init__(
        self,
        initial_capital  : float = 100_000,
        max_positions    : int   = 10,
        decay_days       : int   = 20,
        lookback_quarters: int   = 8,
    ):
        from backtesting.data.loader     import DataLoader
        from backtesting.engine.event_driven import EventDrivenBacktester
        from backtesting.engine.fill_model   import FillModel
        from backtesting.engine.cost_model   import TransactionCostModel

        self.loader    = DataLoader()
        self.signal    = EarningsRevisionSignal(
            lookback_quarters = lookback_quarters,
            decay_days        = decay_days,
        )
        self.engine    = EventDrivenBacktester(
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
    ):
        """
        Fetch data, run backtest, return BacktestResult.
        """
        from backtesting.analysis.metrics import performance_summary, print_summary

        logger.info(f'Loading OHLCV for {len(symbols)} symbols...')
        price_data = self.loader.get_ohlcv(symbols, start_date, end_date)
        logger.info(f'Loaded {len(price_data)} symbols with data')

        logger.info('Loading earnings history...')
        earnings_data = self.loader.get_earnings_history(list(price_data.keys()))
        logger.info(f'Earnings data for {len(earnings_data)} symbols')

        # Build signal function that closes over earnings_data
        signal = self.signal
        earn   = earnings_data

        def signal_fn(date, data):
            return signal.generate_weights(
                date      = date,
                data      = data,
                top_n     = self.max_positions,
                long_only = True,
            ) if not earn else _signal_with_earnings(date, data)

        def _signal_with_earnings(date, data):
            scores = signal.compute(date, data, earn)
            if not scores:
                return {}
            n = min(self.max_positions, len(scores))
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return {sym: 1.0 / n for sym, _ in top}

        logger.info(f'Running backtest {start_date} → {end_date}...')
        result = self.engine.run(
            price_data    = price_data,
            signal_fn     = _signal_with_earnings if earn else signal_fn,
            start_date    = start_date,
            end_date      = end_date,
            max_positions = self.max_positions,
            rebalance_freq= self.signal.rebalance_freq,
        )

        # Print summary
        eq     = result.equity_curve['equity']
        trades = result.trades
        summary = performance_summary(eq, trades)
        print_summary(summary)

        return result
