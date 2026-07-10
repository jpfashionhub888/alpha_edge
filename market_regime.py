# market_regime.py - Fixed V2
# Fixes:
# 1. All indicators use only past data (no look-ahead)
# 2. Regime returns confidence score (not just label)
# 3. Removed the sideways→BUY path that existed as a logic flaw
# 4. VIX integration for volatility-aware regime detection
# 5. Regime hysteresis (prevents rapid flip-flopping)
# 6. Graceful fallback when data is insufficient
# 7. Full audit trail in returned dict

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Thresholds (documented, not magic numbers) ──────────────────────────────
# These are standard technical analysis thresholds — empirically adjustable
BULL_MA_SPREAD_PCT = 0.02      # Price > 200MA by at least 2%
BEAR_MA_SPREAD_PCT = -0.02     # Price < 200MA by at least 2%
BULL_MOMENTUM_MIN = 0.0        # RSI momentum must be positive
HIGH_VOLATILITY_THRESHOLD = 25 # VIX above this = elevated risk
EXTREME_VOLATILITY_THRESHOLD = 35  # VIX above this = crisis mode

MIN_BARS_REQUIRED = 210        # Need 200MA + buffer


class MarketRegimeDetector:
    """
    Detects current market regime using:
    - Price vs 200-day MA (trend)
    - 20-day momentum (direction)
    - Volume trend (conviction)
    - VIX proxy (volatility/fear)

    Returns regime label + confidence score so callers can
    apply proportional position sizing rather than binary on/off.

    Regimes:
        'bull'     — uptrend, trade normally
        'bear'     — downtrend, defensive posture only
        'sideways' — range-bound, reduced size, no new longs
        'crisis'   — extreme volatility, halt new positions
        'unknown'  — insufficient data, treat as sideways
    """

    def __init__(self, hysteresis_bars: int = 1):
        """
        hysteresis_bars: regime must hold for N bars before switching.
        Prevents flip-flopping on choppy days.
        """
        self.hysteresis_bars = hysteresis_bars
        self._regime_history: list = []
        self._current_regime: str = "unknown"

    def analyze(self) -> Dict[str, Any]:
        """
        Backward-compatible analyze method that fetches SPY and VIX data,
        runs the detect logic, and returns a dict compatible with older callers.
        """
        import yfinance as yf
        print("\n   Analyzing market regime...")

        spy_df = None
        try:
            spy = yf.Ticker("SPY")
            spy_df = spy.history(period="1y")
            if spy_df.empty:
                spy_df = None
        except Exception as e:
            logger.warning(f"Error fetching SPY data for analyze: {e}")

        vix_df = None
        try:
            vix = yf.Ticker("^VIX")
            vix_df = vix.history(period="5d")
            if vix_df.empty:
                vix_df = None
        except Exception as e:
            logger.warning(f"Error fetching VIX data for analyze: {e}")

        detect_res = self.detect(spy_df, vix_df)

        regime = detect_res["regime"].upper()
        if regime == "CRISIS":
            regime = "CRASH"
        elif regime == "SIDEWAYS":
            regime = "CAUTION"

        spy_return_1m = 0.0
        spy_return_3m = 0.0
        if spy_df is not None and "Close" in spy_df.columns:
            close = spy_df["Close"].dropna()
            if len(close) >= 21:
                spy_return_1m = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21])
            if len(close) >= 63:
                spy_return_3m = float((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63])

        vix_level = detect_res["signals"].get("vix_level", 20.0)
        rec_map = {
            "BULL": "TRADE NORMALLY",
            "BEAR": "CASH MODE - No new buys",
            "CRASH": "CASH MODE - Extreme fear",
            "CAUTION": "REDUCED TRADING - High confidence only",
            "UNKNOWN": "REDUCED TRADING - High confidence only"
        }
        recommendation = rec_map.get(regime, "REDUCED TRADING - High confidence only")
        reason_map = {
            "BULL": f"Bull market: SPY {spy_return_1m:+.1%}, VIX={vix_level:.1f}",
            "BEAR": f"Bear market detected: SPY down {spy_return_1m:.1%} in 1 month, VIX={vix_level:.1f}",
            "CRASH": f"Extreme fear: VIX={vix_level:.1f}",
            "CAUTION": f"Cautious market: SPY {spy_return_1m:+.1%}, VIX={vix_level:.1f}",
            "UNKNOWN": "Insufficient data, cautious mode"
        }
        reason = reason_map.get(regime, "Normal market conditions")
        result = {
            "regime": regime,
            "can_trade": detect_res["tradeable"],
            "confidence": detect_res.get("confidence", 1.0),
            "spy_return_1m": spy_return_1m,
            "spy_return_3m": spy_return_3m,
            "vix": vix_level,
            "reason": reason,
            "recommendation": recommendation,
        }
        regime_emoji = {
            'BULL'   : 'BULL MARKET',
            'CAUTION': 'CAUTION',
            'BEAR'   : 'BEAR MARKET',
            'CRASH'  : 'MARKET CRASH',
        }.get(regime, 'UNKNOWN')

        print(f"   Market Regime: {regime_emoji}")
        print(f"   SPY 1-Month:   {spy_return_1m:+.2%}")
        print(f"   SPY 3-Month:   {spy_return_3m:+.2%}")
        print(f"   VIX Level:     {vix_level:.1f}")
        print(f"   Can Trade:     {result['can_trade']}")
        print(f"   Reason:        {result['reason']}")
        print(f"   Action:        {result['recommendation']}")

        return result


    def detect(
        self,
        price_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            price_df: OHLCV DataFrame, chronological order, NO future data
            vix_df:   Optional VIX OHLCV DataFrame (same index)

        Returns dict with:
            regime:     str label
            confidence: float 0.0–1.0
            signals:    dict of sub-indicator values (for audit/logging)
            tradeable:  bool — False means halt new positions
        """
        result = self._empty_result()

        # ── Validation ──────────────────────────────────────────────────
        if price_df is None or len(price_df) < MIN_BARS_REQUIRED:
            logger.warning(
                f"Insufficient data for regime detection "
                f"({len(price_df) if price_df is not None else 0} bars, "
                f"need {MIN_BARS_REQUIRED})"
            )
            result["regime"] = "unknown"
            result["tradeable"] = False
            return result

        # Normalize column names — handle both 'Close' and 'close'
        if "Close" not in price_df.columns and "close" in price_df.columns:
            price_df = price_df.copy()
            price_df.columns = [c.capitalize() for c in price_df.columns]

        if "Close" not in price_df.columns:
            logger.error("price_df missing 'Close' column")
            result["regime"] = "unknown"
            result["tradeable"] = False
            return result

        # ── Use only past data (NO look-ahead) ──────────────────────────
        # All calculations operate on price_df as received.
        # Callers must NOT pass future-padded DataFrames.
        close = price_df["Close"].copy()
        volume = price_df.get("Volume", pd.Series(dtype=float))

        # ── Indicators ──────────────────────────────────────────────────
        ma200 = close.rolling(200, min_periods=200).mean()
        ma50 = close.rolling(50, min_periods=50).mean()
        ma20 = close.rolling(20, min_periods=20).mean()

        latest_close = float(close.iloc[-1])
        latest_ma200 = float(ma200.iloc[-1]) if not pd.isna(ma200.iloc[-1]) else None
        latest_ma50 = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else None
        latest_ma20 = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else None

        if latest_ma200 is None:
            logger.warning("MA200 not available — insufficient history")
            result["regime"] = "unknown"
            result["tradeable"] = False
            return result

        # MA spread (price vs 200MA)
        ma_spread_pct = (latest_close - latest_ma200) / latest_ma200

        # 20-day momentum (past data only — shift(0) is current, no future)
        momentum_20d = float(
            (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
        ) if len(close) >= 21 else 0.0

        # Volume trend (20d avg vs 50d avg — conviction)
        vol_trend = 0.0
        if len(volume) >= 50 and volume.sum() > 0:
            vol_20d = float(volume.iloc[-20:].mean())
            vol_50d = float(volume.iloc[-50:].mean())
            vol_trend = (vol_20d - vol_50d) / vol_50d if vol_50d > 0 else 0.0

        # MA alignment (50 > 200 = bullish structure)
        ma_aligned_bull = (
            latest_ma50 is not None
            and latest_ma20 is not None
            and latest_ma50 > latest_ma200
            and latest_ma20 > latest_ma50
        )
        ma_aligned_bear = (
            latest_ma50 is not None
            and latest_ma50 < latest_ma200
        )

        # VIX / volatility proxy
        vix_level = self._get_vix_level(vix_df, close)

        # ── Regime scoring ───────────────────────────────────────────────
        # Score 0–4 bull signals, then classify
        bull_signals = 0
        bear_signals = 0

        if ma_spread_pct >= BULL_MA_SPREAD_PCT:
            bull_signals += 1
        elif ma_spread_pct <= BEAR_MA_SPREAD_PCT:
            bear_signals += 1

        if momentum_20d > 0:
            bull_signals += 1
        elif momentum_20d < -0.03:
            bear_signals += 1

        if ma_aligned_bull:
            bull_signals += 1
        elif ma_aligned_bear:
            bear_signals += 1

        if vol_trend > 0.1 and momentum_20d > 0:
            bull_signals += 1  # rising volume on up move = conviction
        elif vol_trend > 0.1 and momentum_20d < 0:
            bear_signals += 1  # rising volume on down move = distribution

        # ── Crisis override ──────────────────────────────────────────────
        if vix_level >= EXTREME_VOLATILITY_THRESHOLD:
            regime = "crisis"
            confidence = 1.0
            tradeable = False

        # ── Primary regime classification ────────────────────────────────
        elif bull_signals >= 3:
            regime = "bull"
            confidence = min(bull_signals / 4.0, 1.0)
            tradeable = True

        elif bear_signals >= 3:
            regime = "bear"
            confidence = min(bear_signals / 4.0, 1.0)
            # In bear: only allow shorts if system supports them
            # For long-only systems, tradeable=False in bear
            tradeable = False

        elif bear_signals >= 2:
            regime = "bear"
            confidence = 0.5 + (bear_signals - 2) * 0.1
            tradeable = False

        else:
            # Neither clearly bull nor bear = sideways
            regime = "sideways"
            confidence = 1.0 - abs(bull_signals - bear_signals) / 4.0
            # KEY FIX: sideways is NEVER tradeable for new long positions
            # This removes the sideways→BUY path that was a logic flaw
            tradeable = False

        # High volatility degrades confidence even in bull regime
        if vix_level >= HIGH_VOLATILITY_THRESHOLD and regime == "bull":
            confidence *= 0.7
            logger.info(
                f"Bull regime confidence reduced to {confidence:.2f} "
                f"due to elevated VIX ({vix_level:.1f})"
            )

        # ── Hysteresis ───────────────────────────────────────────────────
        regime = self._apply_hysteresis(regime)

        # ── Build result ─────────────────────────────────────────────────
        result.update({
            "regime": regime,
            "confidence": round(confidence, 3),
            "tradeable": tradeable,
            "signals": {
                "ma_spread_pct": round(ma_spread_pct, 4),
                "momentum_20d": round(momentum_20d, 4),
                "vol_trend": round(vol_trend, 4),
                "ma_aligned_bull": ma_aligned_bull,
                "ma_aligned_bear": ma_aligned_bear,
                "bull_signal_count": bull_signals,
                "bear_signal_count": bear_signals,
                "vix_level": round(vix_level, 1),
                "latest_close": round(latest_close, 2),
                "ma200": round(latest_ma200, 2),
                "ma50": round(latest_ma50, 2) if latest_ma50 else None,
            },
        })

        logger.info(
            f"Regime: {regime} | confidence={confidence:.2f} | "
            f"tradeable={tradeable} | bull={bull_signals} bear={bear_signals} "
            f"VIX={vix_level:.1f}"
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────

    def _apply_hysteresis(self, new_regime: str) -> str:
        """
        Only switch regime if new_regime has held for hysteresis_bars
        consecutive detections. Prevents day-to-day flip-flopping.
        """
        self._regime_history.append(new_regime)
        # Keep only recent history
        self._regime_history = self._regime_history[-self.hysteresis_bars:]

        if len(self._regime_history) < self.hysteresis_bars:
            return self._current_regime or new_regime

        # All recent readings agree?
        if len(set(self._regime_history)) == 1:
            if self._current_regime != new_regime:
                logger.info(
                    f"Regime switch: {self._current_regime} → {new_regime} "
                    f"(held {self.hysteresis_bars} bars)"
                )
            self._current_regime = new_regime

        return self._current_regime

    def _get_vix_level(
        self,
        vix_df: Optional[pd.DataFrame],
        close: pd.Series,
    ) -> float:
        """
        Get VIX reading. Falls back to realized volatility proxy
        (20-day annualized returns std) if VIX data unavailable.
        """
        if vix_df is not None and len(vix_df) >= 1:
            try:
                return float(vix_df["Close"].iloc[-1])
            except (KeyError, IndexError):
                pass

        # Realized volatility proxy (no VIX data available)
        if len(close) >= 21:
            returns = close.pct_change().dropna()
            realized_vol = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)
            logger.debug(
                f"VIX unavailable, using realized vol proxy: {realized_vol:.1f}"
            )
            return realized_vol

        return 0.0  # Cannot estimate — treat as calm

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "regime": "unknown",
            "confidence": 0.0,
            "tradeable": False,
            "signals": {},
        }


# ── Backward-compatible functional wrapper ─────────────────────────────────
# If older code calls detect_regime(df) directly, this keeps it working
_default_detector = MarketRegimeDetector()


def detect_regime(
    price_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Module-level function for backward compatibility."""
    return _default_detector.detect(price_df, vix_df)


# Alias — both names are valid imports
MarketRegimeFilter = MarketRegimeDetector
