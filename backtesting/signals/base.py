# backtesting/signals/base.py
"""
Signal base class.

Every signal in the library must inherit from BaseSignal and implement:
  - compute(date, data) → {symbol: score}
  - name: str

Scores are continuous values where higher = more bullish.
The backtesting engine converts scores to target weights via position sizing.
Scores are NOT probabilities and do NOT need to sum to 1.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseSignal(ABC):
    """
    Abstract base for all AlphaEdge signals.

    Contract:
      - compute() receives only data strictly BEFORE `date` (enforced by engine)
      - compute() returns scores in [-1, +1] range where possible
        (+1 = maximum long conviction, -1 = maximum short conviction)
      - compute() must never raise — return {} on any failure
      - Signal must declare its expected holding period and rebalance frequency
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique signal identifier."""
        ...

    @property
    def holding_period_days(self) -> int:
        """Expected holding period in trading days."""
        return 5

    @property
    def rebalance_freq(self) -> str:
        """Rebalance frequency: 'D', 'W', or 'M'."""
        return 'W'

    @property
    def min_history_days(self) -> int:
        """Minimum history required before signal produces scores."""
        return 60

    @abstractmethod
    def compute(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Compute signal scores for all symbols as of `date`.

        Parameters
        ----------
        date : the current backtesting date (signal is computed BEFORE this date's data)
        data : {symbol: DataFrame} containing data strictly before `date`
               Each DataFrame has columns: open, high, low, close, volume

        Returns
        -------
        {symbol: score} — only include symbols with a valid score
        Missing symbols are treated as "no view" (not traded)
        """
        ...

    def generate_weights(
        self,
        date    : pd.Timestamp,
        data    : Dict[str, pd.DataFrame],
        top_n   : int   = 10,
        long_only: bool = True,
    ) -> Dict[str, float]:
        """
        Convert scores to target portfolio weights.

        Default: equal-weight top-N long positions.
        Override for custom position sizing.

        Parameters
        ----------
        top_n     : number of positions to hold
        long_only : if True, only go long top-N; if False, long top-N, short bottom-N
        """
        try:
            scores = self.compute(date, data)
        except Exception as e:
            logger.warning(f'{self.name}: compute() failed ({e})')
            return {}

        if not scores:
            return {}

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if long_only:
            top    = sorted_scores[:top_n]
            n      = len(top)
            return {sym: 1.0 / n for sym, _ in top} if n > 0 else {}
        else:
            top    = sorted_scores[:top_n]
            bottom = sorted_scores[-top_n:]
            n      = len(top)
            weights = {}
            for sym, _ in top:
                weights[sym] = weights.get(sym, 0) + 0.5 / n
            for sym, _ in bottom:
                weights[sym] = weights.get(sym, 0) - 0.5 / n
            return weights

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r})'
