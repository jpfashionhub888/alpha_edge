# multi_timeframe.py — Patched V3
# Changes from V2:
#   get_mtf_score() now accepts optional daily_df parameter.
#   When daily_df is provided, skips yfinance fetch entirely.
#   Handles Bybit symbol format ('BTCUSDT') in yfinance fallback path.
#   Column name normalization centralized here.

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from enum import IntEnum

logger = logging.getLogger(__name__)


class SignalStrength(IntEnum):
    STRONG_SELL = -2
    SELL        = -1
    NEUTRAL     =  0
    BUY         =  1
    STRONG_BUY  =  2


TIMEFRAME_PRIORITY = ["1d", "1wk", "1mo"]

MIN_BARS = {
    "1d" : 60,
    "1wk": 52,
    "1mo": 24,
}

TIMEFRAME_WEIGHTS = {
    "1d" : 0.50,
    "1wk": 0.35,
    "1mo": 0.15,
}


class MultiTimeframeAnalyzer:
    """
    Combines signals from daily, weekly, and monthly timeframes.

    Rules:
    1. Higher timeframe trend = filter (weekly bear blocks daily BUY)
    2. Alignment bonus: all TFs agree = higher confidence
    3. Conflict penalty: TFs disagree = lower confidence
    4. TF with insufficient data is excluded (not assumed neutral)
    5. All calcs use only past data — no look-ahead
    """

    def analyze(
        self,
        symbol:     str,
        daily_df:   pd.DataFrame,
        weekly_df:  Optional[pd.DataFrame] = None,
        monthly_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        result = self._empty_result(symbol)

        if daily_df is None or len(daily_df) < MIN_BARS["1d"]:
            logger.warning(
                f"[{symbol}] Insufficient daily data "
                f"({len(daily_df) if daily_df is not None else 0} bars)"
            )
            return result

        if weekly_df is None:
            weekly_df = self._resample(daily_df, "W")
        if monthly_df is None:
            monthly_df = self._resample(daily_df, "ME")

        tf_results = {
            "1d" : self._score_timeframe("1d",  daily_df),
            "1wk": self._score_timeframe("1wk", weekly_df),
            "1mo": self._score_timeframe("1mo", monthly_df),
        }
        result["timeframes"] = tf_results

        included = {tf: r for tf, r in tf_results.items() if r["included"]}
        if not included:
            logger.warning(f"[{symbol}] No timeframe had sufficient data")
            return result

        # Higher-TF filter
        blocked_by   = None
        monthly_score = tf_results["1mo"]["score"] if tf_results["1mo"]["included"] else 0.0
        weekly_score  = tf_results["1wk"]["score"] if tf_results["1wk"]["included"] else 0.0
        daily_score   = tf_results["1d"]["score"]

        if monthly_score < -0.5 and daily_score > 0:
            blocked_by = "monthly_bearish"
        elif weekly_score < -0.3 and daily_score > 0:
            blocked_by = "weekly_bearish"

        result["blocked_by"] = blocked_by

        # Weighted composite
        total_weight = sum(TIMEFRAME_WEIGHTS[tf] for tf in included)
        weighted_sum = sum(r["score"] * TIMEFRAME_WEIGHTS[tf] for tf, r in included.items())
        composite    = weighted_sum / total_weight if total_weight > 0 else 0.0
        result["composite"] = round(composite, 4)

        # Conflicts
        conflicts = self._detect_conflicts(included)
        result["conflicts"] = conflicts

        # Confidence
        coverage        = len(included) / len(TIMEFRAME_PRIORITY)
        scores          = [r["score"] for r in included.values()]
        alignment_bonus = 0.2 if all(s > 0 for s in scores) or all(s < 0 for s in scores) else 0.0
        conflict_penalty = len(conflicts) * 0.1
        confidence = max(0.0, min(coverage * 0.6 + alignment_bonus - conflict_penalty, 1.0))
        result["confidence"] = round(confidence, 3)

        # Final signal
        if blocked_by:
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
            f"conf={confidence:.3f} conflicts={len(conflicts)} blocked_by={blocked_by}"
        )
        return result

    def _score_timeframe(self, tf_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        min_bars = MIN_BARS.get(tf_name, 60)
        base = {"signal": "NEUTRAL", "score": 0.0, "bars_used": 0, "included": False}

        if df is None or len(df) < min_bars:
            base["bars_used"] = len(df) if df is not None else 0
            return base

        close = df["Close"].copy()
        try:
            ma_len       = min(50, len(close) // 2)
            ma           = close.rolling(ma_len, min_periods=ma_len).mean()
            latest_close = float(close.iloc[-1])
            latest_ma    = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else latest_close
            ma_score     = (latest_close - latest_ma) / latest_ma

            lookback = min(20, len(close) - 1)
            momentum = (
                (close.iloc[-1] - close.iloc[-lookback - 1]) / close.iloc[-lookback - 1]
                if lookback > 0 else 0.0
            )

            ma_norm  = float(np.clip(ma_score   / 0.10, -1.0, 1.0))
            mom_norm = float(np.clip(float(momentum) / 0.20, -1.0, 1.0))
            score    = float(np.clip(ma_norm * 0.6 + mom_norm * 0.4, -1.0, 1.0))

            signal = "BULL" if score >= 0.3 else ("BEAR" if score <= -0.3 else "NEUTRAL")
            return {
                "signal"   : signal,
                "score"    : round(score, 4),
                "bars_used": len(close),
                "included" : True,
                "ma_score" : round(ma_norm, 4),
                "momentum" : round(mom_norm, 4),
            }
        except Exception as e:
            logger.warning(f"Error scoring {tf_name}: {e}")
            return base

    @staticmethod
    def _detect_conflicts(included: Dict[str, Dict]) -> List[str]:
        conflicts = []
        tfs = [tf for tf in TIMEFRAME_PRIORITY if tf in included]
        for i in range(len(tfs) - 1):
            lower, higher = tfs[i], tfs[i + 1]
            ls = included[lower]["signal"]
            hs = included[higher]["signal"]
            if ls == "BULL" and hs == "BEAR":
                conflicts.append(f"{lower}_bull_vs_{higher}_bear")
            elif ls == "BEAR" and hs == "BULL":
                conflicts.append(f"{lower}_bear_vs_{higher}_bull")
        return conflicts

    @staticmethod
    def _resample(daily_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        try:
            return daily_df.resample(freq).agg({
                "Open" : "first",
                "High" : "max",
                "Low"  : "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna(subset=["Close"])
        except Exception as e:
            logger.warning(f"Resampling to {freq} failed: {e}")
            return pd.DataFrame()

    def get_mtf_score(
        self,
        symbol:   str,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Return composite MTF score (-1.0 to +1.0).

        Two modes:
          1. Pass daily_df → uses it directly (fast, no network call).
             Used by bybit_live.py which has live candle data in memory.

          2. Pass symbol only → fetches from yfinance.
             Used by main.py scanner (original behavior).

        Column names are normalized automatically — accepts both
        lowercase (open/close/...) and Title Case (Open/Close/...).
        """
        df = None

        if daily_df is not None:
            # ── Fast path: caller provides DataFrame ─────────────────
            df = daily_df.copy()
            # Normalize to Title Case (required by analyze())
            df.columns = [c.capitalize() for c in df.columns]

        else:
            # ── Fetch path: yfinance (original behavior) ──────────────
            try:
                import yfinance as yf

                # Handle Bybit format 'BTCUSDT' → yfinance needs 'BTC-USD'
                fetch_sym = symbol
                if (
                    symbol.endswith('USDT')
                    and '/' not in symbol
                    and '-' not in symbol
                ):
                    base      = symbol.replace('USDT', '')
                    fetch_sym = f'{base}-USD'
                elif '/' in symbol:
                    # 'BTC/USD' → 'BTC-USD'
                    fetch_sym = symbol.replace('/', '-')

                raw = yf.download(fetch_sym, period='1y', interval='1d', progress=False)
                if raw is None or len(raw) < 60:
                    return 0.0

                df = raw.copy()
                # yfinance sometimes returns MultiIndex columns — flatten
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.capitalize() for c in df.columns]

            except Exception as e:
                logger.warning(f"get_mtf_score({symbol}): yfinance fetch failed: {e}")
                return 0.0

        if df is None or len(df) < 60:
            return 0.0

        result = self.analyze(symbol, df)
        return result.get("composite", 0.0)

    @staticmethod
    def _empty_result(symbol: str) -> Dict[str, Any]:
        return {
            "symbol"    : symbol,
            "signal"    : "HOLD",
            "confidence": 0.0,
            "composite" : 0.0,
            "timeframes": {},
            "conflicts" : [],
            "blocked_by": None,
        }


# ── Module-level convenience ──────────────────────────────────────────

_analyzer = MultiTimeframeAnalyzer()


def analyze_timeframes(
    symbol:     str,
    daily_df:   pd.DataFrame,
    weekly_df:  Optional[pd.DataFrame] = None,
    monthly_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Backward-compatible function wrapper."""
    return _analyzer.analyze(symbol, daily_df, weekly_df, monthly_df)


def get_mtf_score(
    symbol:   str,
    daily_df: Optional[pd.DataFrame] = None,
) -> float:
    """
    Module-level convenience wrapper.
    See MultiTimeframeAnalyzer.get_mtf_score() for full docs.
    """
    return _analyzer.get_mtf_score(symbol, daily_df)
