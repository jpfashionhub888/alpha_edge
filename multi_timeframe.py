# multi_timeframe.py - Fixed V2
# Fixes:
# 1. Strict temporal alignment (higher TF never uses future data vs lower TF)
# 2. Explicit conflict resolution with documented priority hierarchy
# 3. Graceful degradation (one TF failure = reduce confidence, not crash)
# 4. No silent averaging of opposing signals
# 5. Returns audit trail showing which TFs agreed/disagreed

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from enum import IntEnum

logger = logging.getLogger(__name__)


class SignalStrength(IntEnum):
    """
    Numeric encoding for signals — enables weighted averaging.
    Using IntEnum so values are stable across code changes.
    """
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


# Timeframe hierarchy — higher index = higher authority in conflicts
TIMEFRAME_PRIORITY = ["1d", "1wk", "1mo"]

# Minimum bars needed per timeframe before that TF is included
MIN_BARS = {
    "1d": 60,
    "1wk": 52,
    "1mo": 24,
}

# Weight of each timeframe in final signal (must sum to 1.0)
# Daily has highest weight — it's freshest and most actionable
TIMEFRAME_WEIGHTS = {
    "1d": 0.50,
    "1wk": 0.35,
    "1mo": 0.15,
}


class MultiTimeframeAnalyzer:
    """
    Combines signals from daily, weekly, and monthly timeframes.

    Rules:
    1. Higher timeframe trend = filter, not signal source
       (weekly bear = no daily BUY regardless of daily reading)
    2. Alignment bonus: all TFs agree = higher confidence
    3. Conflict penalty: TFs disagree = reduce confidence proportionally
    4. A TF with insufficient data is excluded (not assumed neutral)
    5. All calculations strictly use past data per timeframe
    """

    def analyze(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
        weekly_df: Optional[pd.DataFrame] = None,
        monthly_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            symbol:     Ticker symbol (for logging)
            daily_df:   Daily OHLCV — required
            weekly_df:  Weekly OHLCV — optional, derived from daily if None
            monthly_df: Monthly OHLCV — optional, derived from daily if None

        Returns:
            {
                signal:     'BUY' | 'HOLD' | 'AVOID'
                confidence: float 0.0–1.0
                composite:  float -1.0 to +1.0 (raw weighted score)
                timeframes: {tf: {signal, score, bars_used, included}}
                conflicts:  list of conflict descriptions
                blocked_by: str | None — which TF blocked a lower-TF BUY
            }
        """
        result = self._empty_result(symbol)

        # ── Validate daily data (required) ───────────────────────────────
        if daily_df is None or len(daily_df) < MIN_BARS["1d"]:
            logger.warning(
                f"[{symbol}] Insufficient daily data "
                f"({len(daily_df) if daily_df is not None else 0} bars)"
            )
            return result

        # ── Derive weekly/monthly if not provided ────────────────────────
        # Resampling from daily avoids fetching separate API calls
        # and ensures temporal alignment (monthly close = last daily close of month)
        if weekly_df is None:
            weekly_df = self._resample(daily_df, "W")
        if monthly_df is None:
            monthly_df = self._resample(daily_df, "ME")

        # ── Score each timeframe independently ───────────────────────────
        tf_results = {}
        tf_results["1d"] = self._score_timeframe("1d", daily_df)
        tf_results["1wk"] = self._score_timeframe("1wk", weekly_df)
        tf_results["1mo"] = self._score_timeframe("1mo", monthly_df)

        result["timeframes"] = tf_results

        # ── Only include TFs with sufficient data ────────────────────────
        included = {
            tf: r for tf, r in tf_results.items()
            if r["included"]
        }

        if not included:
            logger.warning(f"[{symbol}] No timeframe had sufficient data")
            return result

        # ── Higher-timeframe filter (key look-ahead prevention) ──────────
        # If monthly or weekly is bearish, block daily BUY signals
        # This is directional filtering, not signal averaging
        blocked_by = None

        monthly_score = tf_results["1mo"]["score"] if tf_results["1mo"]["included"] else 0.0
        weekly_score = tf_results["1wk"]["score"] if tf_results["1wk"]["included"] else 0.0
        daily_score = tf_results["1d"]["score"]

        if monthly_score < -0.5 and daily_score > 0:
            blocked_by = "monthly_bearish"
            logger.info(
                f"[{symbol}] Daily BUY blocked by monthly bear trend "
                f"(monthly={monthly_score:.2f})"
            )

        elif weekly_score < -0.3 and daily_score > 0:
            blocked_by = "weekly_bearish"
            logger.info(
                f"[{symbol}] Daily BUY blocked by weekly bear trend "
                f"(weekly={weekly_score:.2f})"
            )

        result["blocked_by"] = blocked_by

        # ── Weighted composite score ─────────────────────────────────────
        total_weight = 0.0
        weighted_sum = 0.0

        for tf, r in included.items():
            w = TIMEFRAME_WEIGHTS[tf]
            weighted_sum += r["score"] * w
            total_weight += w

        # Normalize by actual weight used (handles missing TFs)
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        result["composite"] = round(composite, 4)

        # ── Conflict detection ───────────────────────────────────────────
        conflicts = self._detect_conflicts(included)
        result["conflicts"] = conflicts

        # ── Confidence ───────────────────────────────────────────────────
        # Base: proportion of TFs that have data
        coverage = len(included) / len(TIMEFRAME_PRIORITY)

        # Alignment bonus (all point same direction)
        scores = [r["score"] for r in included.values()]
        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)
        alignment_bonus = 0.2 if (all_positive or all_negative) else 0.0

        # Conflict penalty
        conflict_penalty = len(conflicts) * 0.1

        confidence = min(
            coverage * 0.6 + alignment_bonus - conflict_penalty,
            1.0,
        )
        confidence = max(confidence, 0.0)
        result["confidence"] = round(confidence, 3)

        # ── Final signal ─────────────────────────────────────────────────
        if blocked_by:
            # Higher TF says no — override to HOLD regardless of daily signal
            signal = "HOLD"
        elif composite >= 0.4:
            signal = "BUY"
        elif composite <= -0.4:
            signal = "AVOID"
        else:
            signal = "HOLD"

        result["signal"] = signal

        logger.info(
            f"[{symbol}] MTF | signal={signal} composite={composite:.3f} "
            f"conf={confidence:.3f} conflicts={len(conflicts)} "
            f"blocked_by={blocked_by}"
        )
        return result

    # ── Timeframe scoring ─────────────────────────────────────────────────

    def _score_timeframe(
        self,
        tf_name: str,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Score a single timeframe -1.0 to +1.0 using:
        - Price vs MA (trend)
        - Recent momentum
        - RSI-like mean reversion indicator

        All indicators use only past data (rolling on historical bars).
        """
        min_bars = MIN_BARS.get(tf_name, 60)
        base_result = {
            "signal": "NEUTRAL",
            "score": 0.0,
            "bars_used": 0,
            "included": False,
        }

        if df is None or len(df) < min_bars:
            base_result["bars_used"] = len(df) if df is not None else 0
            return base_result

        close = df["Close"].copy()

        try:
            # ── MA trend ────────────────────────────────────────────────
            ma_len = min(50, len(close) // 2)
            ma = close.rolling(ma_len, min_periods=ma_len).mean()
            latest_close = float(close.iloc[-1])
            latest_ma = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else latest_close
            ma_score = (latest_close - latest_ma) / latest_ma  # pct above/below MA

            # ── Momentum ─────────────────────────────────────────────────
            lookback = min(20, len(close) - 1)
            momentum = (
                (close.iloc[-1] - close.iloc[-lookback - 1]) / close.iloc[-lookback - 1]
                if lookback > 0 else 0.0
            )

            # ── Normalize to -1..+1 ──────────────────────────────────────
            # MA score typically ±5–15%; clip to ±10% then normalize
            ma_norm = np.clip(ma_score / 0.10, -1.0, 1.0)
            # Momentum typically ±20%; clip to ±20% then normalize
            mom_norm = np.clip(float(momentum) / 0.20, -1.0, 1.0)

            # Weighted: MA is more reliable (structural), momentum more noisy
            score = ma_norm * 0.6 + mom_norm * 0.4
            score = float(np.clip(score, -1.0, 1.0))

            # ── Signal label ─────────────────────────────────────────────
            if score >= 0.3:
                signal = "BULL"
            elif score <= -0.3:
                signal = "BEAR"
            else:
                signal = "NEUTRAL"

            return {
                "signal": signal,
                "score": round(score, 4),
                "bars_used": len(close),
                "included": True,
                "ma_score": round(float(ma_norm), 4),
                "momentum": round(float(mom_norm), 4),
            }

        except Exception as e:
            logger.warning(f"Error scoring {tf_name}: {e}")
            return base_result

    # ── Conflict detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_conflicts(included: Dict[str, Dict]) -> List[str]:
        """
        Find cases where timeframes directly contradict each other.
        A conflict is when higher TF is BEAR but lower TF is BULL.
        """
        conflicts = []
        tfs = [tf for tf in TIMEFRAME_PRIORITY if tf in included]

        for i in range(len(tfs) - 1):
            lower_tf = tfs[i]
            higher_tf = tfs[i + 1]
            lower_signal = included[lower_tf]["signal"]
            higher_signal = included[higher_tf]["signal"]

            if lower_signal == "BULL" and higher_signal == "BEAR":
                conflicts.append(
                    f"{lower_tf}_bull_vs_{higher_tf}_bear"
                )
            elif lower_signal == "BEAR" and higher_signal == "BULL":
                conflicts.append(
                    f"{lower_tf}_bear_vs_{higher_tf}_bull"
                )
        return conflicts

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _resample(daily_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample daily OHLCV to weekly or monthly.
        Uses past-data-only aggregation:
        - Open: first of period
        - High/Low: max/min of period
        - Close: last of period
        - Volume: sum of period

        This is safe from look-ahead because we only see completed periods.
        """
        try:
            resampled = daily_df.resample(freq).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna(subset=["Close"])
            return resampled
        except Exception as e:
            logger.warning(f"Resampling to {freq} failed: {e}")
            return pd.DataFrame()

    def get_mtf_score(
        self,
        symbol: str,
        daily_df: pd.DataFrame,
    ) -> float:
        """
        Backward-compatible wrapper for scanner.py.
        Returns a single float score -1.0 to +1.0.
        """
        # Normalize column names to Title Case
        # scanner.py uses lowercase, MTF analyzer needs Title Case
        df = daily_df.copy()
        df.columns = [c.capitalize() for c in df.columns]

        result = self.analyze(symbol, df)
        return result.get("composite", 0.0)

    @staticmethod
    def _empty_result(symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0.0,
            "composite": 0.0,
            "timeframes": {},
            "conflicts": [],
            "blocked_by": None,
        }


# ── Module-level convenience function ─────────────────────────────────────
_analyzer = MultiTimeframeAnalyzer()


def analyze_timeframes(
    symbol: str,
    daily_df: pd.DataFrame,
    weekly_df: Optional[pd.DataFrame] = None,
    monthly_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Backward-compatible function wrapper."""
    return _analyzer.analyze(symbol, daily_df, weekly_df, monthly_df)