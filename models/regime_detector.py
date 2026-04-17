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
        ma_slope = ma_long.pct_change(10)

        # Volatility regime
        vol = df['returns'].rolling(self.vol_window).std()
        vol_ma = vol.rolling(63).mean()
        vol_ratio = vol / (vol_ma + 1e-8)

        # Determine regime
        df['regime'] = 'sideways'

        # Strong uptrend
        uptrend_mask = (
            (ma_short > ma_long)
            & (ma_slope > 0.01)
            & (df['close'] > ma_long)
        )
        df.loc[uptrend_mask, 'regime'] = 'uptrend'

        # Strong downtrend
        downtrend_mask = (
            (ma_short < ma_long)
            & (ma_slope < -0.01)
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

        logger.info("Regime detection complete")

        # Print regime distribution
        regime_counts = df['regime'].value_counts()
        print("   Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"      {regime}: {count} days ({pct:.1f}%)")

        return df