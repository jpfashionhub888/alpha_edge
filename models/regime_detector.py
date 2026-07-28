# models/regime_detector.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime: trending up, trending down,
    or sideways/choppy. This single addition is worth
    more than 50 technical indicators combined.
    """

    def __init__(self, short_window=21, long_window=63,
                 vol_window=21, adx_threshold=25):
        self.short_window = short_window
        self.long_window = long_window
        self.vol_window = vol_window
        self.adx_threshold = adx_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime columns to dataframe."""

        df = df.copy()

        # Trend direction using dual moving average
        ma_short = df['close'].rolling(self.short_window).mean()
        ma_long = df['close'].rolling(self.long_window).mean()

        # Trend strength using slope of long MA
        ma_slope = ma_long.pct_change(10, fill_method=None)

        # Volatility regime
        vol = df['returns'].rolling(self.vol_window).std()
        vol_ma = vol.rolling(63).mean()
        vol_ratio = vol / (vol_ma + 1e-8)

        # Determine regime
        df['regime'] = 'sideways'

        # Strong uptrend — loosened thresholds
        # ma_slope > 0.001 instead of 0.01
        # markets rarely show 1% slope per 10 days
        # Uptrend — price above long MA and short above long
        # Removed slope requirement — too restrictive
        uptrend_mask = (
            (ma_short > ma_long)
            & (df['close'] > ma_long)
        )
        df.loc[uptrend_mask, 'regime'] = 'uptrend'

        # Downtrend — price below long MA and short below long
        downtrend_mask = (
            (ma_short < ma_long)
            & (df['close'] < ma_long)
        )
        df.loc[downtrend_mask, 'regime'] = 'downtrend'

        # High volatility override
        high_vol_mask = vol_ratio > 1.5
        df.loc[high_vol_mask, 'regime'] = 'volatile'

        # Numeric regime for ML
        regime_map = {
            'uptrend': 1,
            'sideways': 0,
            'downtrend': -1,
            'volatile': -2
        }
        df['regime_numeric'] = df['regime'].map(regime_map)

        # Regime duration (how long in current regime)
        regime_changes = (
            df['regime'] != df['regime'].shift()
        ).cumsum()
        df['regime_duration'] = df.groupby(
            regime_changes
        ).cumcount() + 1

        # Trend strength score (0 to 100)
        ma_diff = (ma_short - ma_long) / (ma_long + 1e-8)
        df['trend_strength'] = (
            ma_diff.clip(-0.1, 0.1) / 0.1 * 50 + 50
        )

        # S6 FIX: was print() — replaced with logger.debug() to stop flooding console
        # (was emitting ~200 lines per scan: 40 stocks × 5 regime lines each).
        # Set LOG_LEVEL=DEBUG to re-enable detailed regime breakdown.
        regime_counts = df['regime'].value_counts()
        dominant = regime_counts.index[0] if len(regime_counts) else 'unknown'
        logger.debug("Regime distribution: %s", regime_counts.to_dict())
        logger.info("Regime detection complete — dominant: %s", dominant)

        return df
