# backtesting/portfolio/constructor.py
"""
Portfolio construction.

Two approaches implemented:
  1. HierarchicalRiskParity (HRP) — Lopez de Prado's method.
     More robust than mean-variance; doesn't require inverting the
     covariance matrix (numerically unstable for large universes).

  2. PositionSizer — signal-strength-aware fractional Kelly sizing.

For Phase 1 with a single signal, equal-weight top-N is sufficient.
HRP becomes important in Phase 3 when combining multiple signals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """
    Lopez de Prado's HRP portfolio construction.

    Advantages over mean-variance optimisation:
    - No matrix inversion (numerically stable for large universes)
    - Produces diversified portfolios without requiring return forecasts
    - Naturally handles correlated assets without concentration

    Typical use: convert signal scores → HRP-weighted positions
    1. Select top-N stocks by signal score
    2. Use HRP on their return history to determine weights
    3. Scale weights by signal conviction (optional)
    """

    def get_weights(
        self,
        returns         : pd.DataFrame,
        signal_scores   : Optional[Dict[str, float]] = None,
        signal_weight   : float = 0.3,
    ) -> pd.Series:
        """
        Compute HRP weights.

        Parameters
        ----------
        returns       : DataFrame of daily returns (index=date, columns=symbols)
        signal_scores : optional {symbol: score} to blend with HRP weights
                        (higher score → higher weight adjustment)
        signal_weight : fraction of weight determined by signal vs HRP (0–1)
                        0 = pure HRP, 1 = pure signal rank weighting

        Returns
        -------
        pd.Series of portfolio weights (sum to 1.0, all positive for long-only)
        """
        symbols = list(returns.columns)
        n       = len(symbols)

        if n < 2:
            return pd.Series({sym: 1.0 / n for sym in symbols})

        # Covariance + correlation
        cov  = returns.cov()
        corr = returns.corr()

        # Clip correlation to [-1, 1] for numerical stability
        corr = corr.clip(-1 + 1e-8, 1 - 1e-8)

        # Distance matrix (correlation distance)
        dist = np.sqrt(((1 - corr) / 2).values)
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        try:
            dist_condensed = squareform(dist, checks=False)
            link           = linkage(dist_condensed, method='single')
            sort_ix        = self._get_quasi_diag(link, n)
        except Exception as e:
            logger.warning(f'HRP clustering failed ({e}) — using equal weight')
            hrp_weights = pd.Series(1.0 / n, index=symbols)
        else:
            hrp_weights = self._recursive_bisection(cov, sort_ix)
            # Reindex to original symbol order
            hrp_weights = hrp_weights.reindex(symbols).fillna(0)
            total = hrp_weights.sum()
            if total > 0:
                hrp_weights /= total

        # Blend with signal scores if provided
        if signal_scores and signal_weight > 0:
            scores  = pd.Series(signal_scores).reindex(symbols).fillna(0)
            # Normalise scores to [0, 1] range
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                norm_scores = (scores - s_min) / (s_max - s_min)
            else:
                norm_scores = pd.Series(1.0 / n, index=symbols)
            norm_scores /= norm_scores.sum()

            hrp_weights = (
                (1 - signal_weight) * hrp_weights
                + signal_weight     * norm_scores
            )
            total = hrp_weights.sum()
            if total > 0:
                hrp_weights /= total

        return hrp_weights

    def _get_quasi_diag(self, link: np.ndarray, n: int) -> list:
        """Sort clustered items by distance (quasi-diagonalisation)."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= n:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= n]
            i   = df0.index
            j   = df0.values - n
            sort_ix[i]  = link[j, 0]
            df0_new = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0_new]).sort_index().reset_index(drop=True)
        return sort_ix.tolist()

    def _recursive_bisection(self, cov: pd.DataFrame, sort_ix: list) -> pd.Series:
        """Allocate weights via recursive bisection of sorted clusters."""
        symbols = list(cov.columns)
        w       = pd.Series(1.0, index=[symbols[i] for i in sort_ix if i < len(symbols)])
        c_items = [sort_ix]
        while c_items:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                if i + 1 >= len(c_items):
                    break
                c0, c1 = c_items[i], c_items[i + 1]
                v0     = self._cluster_var(cov, c0, symbols)
                v1     = self._cluster_var(cov, c1, symbols)
                alpha  = 1 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
                idx0   = [symbols[k] for k in c0 if k < len(symbols)]
                idx1   = [symbols[k] for k in c1 if k < len(symbols)]
                w[idx0] *= alpha
                w[idx1] *= (1 - alpha)
        return w

    def _cluster_var(
        self,
        cov    : pd.DataFrame,
        cluster: list,
        symbols: list,
    ) -> float:
        """Portfolio variance for a cluster of assets."""
        idx  = [symbols[k] for k in cluster if k < len(symbols)]
        sub  = cov.loc[idx, idx]
        ivp  = 1.0 / np.diag(sub.values)
        ivp /= ivp.sum()
        return float(ivp @ sub.values @ ivp)


class PositionSizer:
    """
    Signal-strength-aware position sizing with fractional Kelly.

    Fractional Kelly (25% by default) provides:
    - Smaller positions than full Kelly (more conservative)
    - Appropriate for noisy signals where edge estimate is uncertain
    - Natural degression: stronger signal → larger position

    The max_position_pct cap prevents individual position concentration
    regardless of signal strength.
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,    # 5% max per position
        kelly_fraction  : float = 0.25,    # 25% Kelly (conservative)
    ):
        self.max_position_pct = max_position_pct
        self.kelly_fraction   = kelly_fraction

    def size_position(
        self,
        signal_strength : float,
        portfolio_value : float,
        expected_return : float,    # expected per-trade return (decimal)
        volatility      : float,    # per-trade volatility (decimal)
    ) -> float:
        """
        Compute dollar allocation for a single position.

        Returns dollar amount to allocate (before position direction).
        """
        if volatility <= 0 or expected_return <= 0:
            return 0.0

        # Full Kelly fraction
        full_kelly = expected_return / (volatility ** 2)

        # Fractional Kelly scaled by signal strength (z-score normalised)
        # signal_strength in [-3, 3] range typically
        strength_factor = min(abs(signal_strength) / 2.0, 1.5)
        fractional      = full_kelly * self.kelly_fraction * strength_factor

        dollar_alloc    = portfolio_value * fractional
        max_dollars     = portfolio_value * self.max_position_pct

        return min(dollar_alloc, max_dollars)

    def equal_weight(
        self,
        n_positions  : int,
        portfolio_value: float,
        max_single_pct: Optional[float] = None,
    ) -> float:
        """
        Simple equal-weight allocation per position.
        Useful baseline before fractional Kelly is calibrated.
        """
        if n_positions <= 0:
            return 0.0
        weight = 1.0 / n_positions
        cap    = max_single_pct or self.max_position_pct
        return portfolio_value * min(weight, cap)


# Make Optional importable from this file
from typing import Optional  # noqa: E402 (already imported above, keeping for clarity)
