# models/ensemble.py

import pandas as pd
import numpy as np
from risk.manager import RiskManager
import logging

logger = logging.getLogger(__name__)


class EnsembleStrategy:
    """
    Balanced strategy: aggressive in uptrends,
    cautious in downtrends, risk managed always.
    """

    def __init__(self):
        self.signals = None
        # FIX: use the correct parameter names that RiskManager.__init__ expects
        self.risk_manager = RiskManager(
            base_stop_loss=0.03,
            reward_risk_ratio=2.5,       # 0.08 / 0.03 ≈ 2.5x
            trailing_stop_multiplier=0.8, # 0.025 / 0.03 ≈ 0.8x
            max_daily_loss=0.02
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive signals."""

        df = df.copy()
        df['signal'] = 0.0
        df['signal_reason'] = 'flat'

        for idx in df.index:
            row = df.loc[idx]
            pred = row['prediction']
            regime = row['regime']
            trend_str = row['trend_strength']

            signal = 0.0
            reason = 'no_trade'

            if regime == 'uptrend':
                # Uptrend: be aggressive, lower threshold
                if pred > 0.52:
                    signal = 1.0
                    reason = 'uptrend_long'
                elif pred > 0.48 and trend_str > 65:
                    # Strong uptrend, ride it
                    signal = 0.75
                    reason = 'uptrend_trend_follow'

            elif regime == 'sideways':
                # Sideways: need higher confidence
                if pred > 0.57:
                    signal = 0.75
                    reason = 'sideways_high_conf'

            elif regime == 'downtrend':
                # Downtrend: very high bar only
                if pred > 0.62:
                    signal = 0.25
                    reason = 'downtrend_brave'

            elif regime == 'volatile':
                # Volatile: small positions only
                if pred > 0.60:
                    signal = 0.25
                    reason = 'volatile_small'

            df.loc[idx, 'signal'] = signal
            df.loc[idx, 'signal_reason'] = reason

        # Raw returns
        df['strategy_return'] = df['signal'] * df['returns']

        # Apply risk management
        print("\n   Applying risk management...")
        df = self.risk_manager.apply(df)

        # Use managed returns
        df['strategy_return'] = df['managed_return']

        self.signals = df

        total = len(df)
        active = len(df[df['managed_signal'] > 0])
        pct = active / total * 100

        print(f"\n   Final trading stats:")
        print(f"      Total days: {total}")
        print(f"      Active days: {active} ({pct:.1f}%)")

        reasons = df['signal_reason'].value_counts()
        for reason, count in reasons.items():
            rpct = count / total * 100
            print(f"      {reason}: {count} ({rpct:.1f}%)")

        return df