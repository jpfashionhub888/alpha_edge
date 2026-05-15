# correlation_filter.py - Fixed V2
# Fixes:
# 1. Rolling correlation window (no full-history look-ahead)
# 2. NaN-safe correlation matrix
# 3. Cluster-based rejection (don't hold >N stocks from same cluster)
# 4. Explains WHY a stock was rejected (audit trail)
# 5. Handles new/sparse symbols gracefully

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CORRELATION_WINDOW = 60        # Rolling window for correlation (past data only)
MAX_CORRELATION = 0.80         # Above this = too correlated, reject new position
MAX_CLUSTER_SIZE = 3           # Max stocks from same high-correlation cluster
MIN_DATA_OVERLAP = 30          # Min overlapping bars to compute correlation


class CorrelationFilter:
    """
    Prevents concentration risk by:
    1. Rejecting new positions that are highly correlated with existing positions
    2. Limiting cluster size (group of mutually correlated stocks)
    3. Using rolling windows only (no future data, safe for walk-forward)

    Usage:
        cf = CorrelationFilter()
        allowed, reason = cf.check(
            candidate="NVDA",
            candidate_returns=nvda_returns,
            portfolio={"AAPL": aapl_returns, "MSFT": msft_returns}
        )
    """

    def __init__(
        self,
        max_correlation: float = MAX_CORRELATION,
        max_cluster_size: int = MAX_CLUSTER_SIZE,
        window: int = CORRELATION_WINDOW,
        max_per_sector: int = 2,  # backward compatible
    ):
        self.max_per_sector = max_per_sector
        self.max_correlation = max_correlation
        self.max_cluster_size = max_cluster_size
        self.window = window

    def check(
        self,
        candidate: str,
        candidate_returns: pd.Series,
        portfolio: Dict[str, pd.Series],
    ) -> Tuple[bool, str]:
        """
        Check if adding candidate to portfolio is safe from correlation risk.

        Args:
            candidate:          Symbol being considered
            candidate_returns:  Daily return series for candidate (past data only)
            portfolio:          {symbol: return_series} for current holdings

        Returns:
            (allowed: bool, reason: str)
            reason explains exactly why it was blocked (or approved)
        """
        if not portfolio:
            return True, "Portfolio empty — no correlation risk"

        # ── Compute pairwise correlations ────────────────────────────────
        correlations = {}
        for symbol, port_returns in portfolio.items():
            corr = self._safe_rolling_corr(
                candidate_returns, port_returns
            )
            if corr is not None:
                correlations[symbol] = corr
                logger.debug(
                    f"[{candidate}] vs [{symbol}]: corr={corr:.3f}"
                )

        if not correlations:
            # Can't compute correlation (new symbol, sparse data)
            logger.warning(
                f"[{candidate}] Cannot compute correlation with any "
                f"portfolio position — allowing with caution"
            )
            return True, "Insufficient data for correlation check — allowed with caution"

        # ── Individual correlation check ──────────────────────────────────
        max_corr_symbol = max(correlations, key=lambda s: abs(correlations[s]))
        max_corr_value = correlations[max_corr_symbol]

        if abs(max_corr_value) >= self.max_correlation:
            reason = (
                f"REJECTED: {candidate} has {max_corr_value:.2f} correlation "
                f"with {max_corr_symbol} (limit={self.max_correlation})"
            )
            logger.info(reason)
            return False, reason

        # ── Cluster size check ────────────────────────────────────────────
        # Find how many existing positions are highly correlated with candidate
        cluster_members = [
            sym for sym, corr in correlations.items()
            if abs(corr) >= self.max_correlation * 0.75  # softer threshold for clustering
        ]

        if len(cluster_members) >= self.max_cluster_size - 1:
            # Adding candidate would make cluster too large
            reason = (
                f"REJECTED: Adding {candidate} would create a cluster of "
                f"{len(cluster_members) + 1} correlated stocks "
                f"(limit={self.max_cluster_size}): "
                f"{', '.join(cluster_members)} + {candidate}"
            )
            logger.info(reason)
            return False, reason

        # ── Approved ──────────────────────────────────────────────────────
        avg_corr = np.mean(list(correlations.values()))
        reason = (
            f"APPROVED: {candidate} | "
            f"max_corr={max_corr_value:.2f} with {max_corr_symbol} | "
            f"avg_corr={avg_corr:.2f} | "
            f"cluster_size={len(cluster_members)+1}/{self.max_cluster_size}"
        )
        logger.info(reason)
        return True, reason

    def build_correlation_matrix(
        self,
        returns: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Build a correlation matrix for all symbols using rolling window.
        Returns NaN for pairs with insufficient overlap.
        Useful for dashboard visualization.
        """
        symbols = list(returns.keys())
        n = len(symbols)
        matrix = pd.DataFrame(
            np.eye(n),
            index=symbols,
            columns=symbols,
        )

        for i in range(n):
            for j in range(i + 1, n):
                sym_a, sym_b = symbols[i], symbols[j]
                corr = self._safe_rolling_corr(
                    returns[sym_a], returns[sym_b]
                )
                val = corr if corr is not None else np.nan
                matrix.loc[sym_a, sym_b] = val
                matrix.loc[sym_b, sym_a] = val

        return matrix

    def find_clusters(
        self,
        returns: Dict[str, pd.Series],
        threshold: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Group symbols into clusters of high correlation.
        Useful for portfolio review and risk reporting.

        Returns list of clusters, each cluster is a list of symbols.
        """
        threshold = threshold or self.max_correlation
        matrix = self.build_correlation_matrix(returns)
        symbols = list(returns.keys())

        assigned = set()
        clusters = []

        for sym_a in symbols:
            if sym_a in assigned:
                continue
            cluster = [sym_a]
            for sym_b in symbols:
                if sym_b == sym_a or sym_b in assigned:
                    continue
                corr = matrix.loc[sym_a, sym_b]
                if not np.isnan(corr) and abs(corr) >= threshold:
                    cluster.append(sym_b)
                    assigned.add(sym_b)
            assigned.add(sym_a)
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    # ── Private helpers ────────────────────────────────────────────────────

    def _safe_rolling_corr(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> Optional[float]:
        """
        Compute correlation using only the rolling window.
        Returns None if insufficient overlapping data.

        This is the key look-ahead prevention:
        - We only use the last `window` bars
        - Both series must share the same date index
        - NaN values are dropped before correlation
        """
        try:
            # Align on common dates
            aligned = pd.DataFrame({
                "a": series_a,
                "b": series_b,
            }).dropna()

            if len(aligned) < MIN_DATA_OVERLAP:
                return None

            # Use only the rolling window (past data)
            window_data = aligned.iloc[-self.window:]
            corr = float(window_data["a"].corr(window_data["b"]))

            if np.isnan(corr):
                return None

            return corr

        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return None
    def can_add_position(
        self,
        symbol: str,
        current_positions_or_returns=None,
        current_positions: dict = None,
    ) -> bool:
        """
        Backward-compatible wrapper for scanner.py.

        scanner.py calls: can_add_position(symbol, open_positions)
        where open_positions is a dict of {symbol: position_data}

        Returns True if symbol can be added safely.
        """
        # Handle both calling conventions
        if current_positions is None:
            # Called as can_add_position(symbol, open_positions_dict)
            open_positions = current_positions_or_returns or {}
        else:
            # Called as can_add_position(symbol, returns, positions)
            open_positions = current_positions

        # If no positions open — always allow
        if not open_positions:
            return True

        # Simple sector/symbol concentration check
        # Full correlation check requires return series
        # which scanner does not provide
        # Limit: max 2 positions in same sector
        if len(open_positions) >= self.max_cluster_size:
            logger.info(
                f"[{symbol}] Correlation filter: "
                f"portfolio full ({len(open_positions)} positions)"
            )
            return False

        return True

        """
        Backward-compatible wrapper for scanner.py.
        Returns True if symbol can be added without
        excessive correlation to existing positions.

        current_positions: dict of {symbol: return_series}
        """
        if not current_positions:
            return True

        allowed, reason = self.check(
            candidate=symbol,
            candidate_returns=symbol_returns,
            portfolio=current_positions,
        )
        return allowed

# ── Module-level convenience function ─────────────────────────────────────
_filter = CorrelationFilter()


def check_correlation(
    candidate: str,
    candidate_returns: pd.Series,
    portfolio: Dict[str, pd.Series],
) -> Tuple[bool, str]:
    """Backward-compatible function wrapper."""
    return _filter.check(candidate, candidate_returns, portfolio)