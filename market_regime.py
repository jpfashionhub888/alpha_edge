# market_regime.py
# ALPHAEDGE - Market Regime Filter
# Detects bull/bear market conditions
# Stops trading in bear markets automatically

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """
    Detects overall market regime.
    Prevents buying in bear markets.
    """

    def __init__(self):
        # Market health indicators
        self.spy_symbol  = 'SPY'   # US Market
        self.qqq_symbol  = 'QQQ'   # Tech Market
        self.vix_symbol  = '^VIX'  # Fear Index

        # Thresholds
        self.bear_threshold    = -0.07  # -7% = bear market
        self.recovery_threshold= -0.03  # -3% = recovering
        self.vix_high          = 30     # VIX > 30 = high fear
        self.vix_extreme       = 40     # VIX > 40 = extreme fear

    def get_market_data(self):
        """Fetch market health data."""
        try:
            spy = yf.Ticker(self.spy_symbol)
            df  = spy.history(period='3mo')
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.warning(f"Market data error: {e}")
            return None

    def get_vix(self):
        """Fetch current VIX level."""
        try:
            vix = yf.Ticker(self.vix_symbol)
            df  = vix.history(period='5d')
            if df.empty:
                return 20  # Default neutral
            close = df['Close'].dropna()
            return float(close.iloc[-1])
        except Exception:
            return 20

    def analyze(self):
        """
        Analyze market regime.
        Returns regime dict with trading recommendation.
        """
        print("\n   Analyzing market regime...")

        result = {
            'regime'          : 'BULL',
            'can_trade'       : True,
            'spy_return_1m'   : 0.0,
            'spy_return_3m'   : 0.0,
            'vix'             : 20.0,
            'reason'          : 'Normal market conditions',
            'recommendation'  : 'TRADE NORMALLY',
        }

        # Get market data
        df = self.get_market_data()
        if df is None:
            print("   Could not fetch market data. Allowing trades.")
            return result

        close = df['close'].dropna()

        # Calculate returns
        if len(close) >= 21:
            ret_1m = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
            result['spy_return_1m'] = float(ret_1m)
        else:
            ret_1m = 0.0

        if len(close) >= 63:
            ret_3m = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
            result['spy_return_3m'] = float(ret_3m)
        else:
            ret_3m = 0.0

        # Get VIX
        vix = self.get_vix()
        result['vix'] = vix

        # Determine regime
        if ret_1m <= self.bear_threshold and vix >= self.vix_high:
            result['regime']        = 'BEAR'
            result['can_trade']     = False
            result['reason']        = (
                f"Bear market detected: "
                f"SPY down {ret_1m:.1%} in 1 month, "
                f"VIX={vix:.1f}"
            )
            result['recommendation']= 'CASH MODE - No new buys'

        elif ret_1m <= self.bear_threshold:
            result['regime']        = 'BEAR'
            result['can_trade']     = False
            result['reason']        = (
                f"Market correction: "
                f"SPY down {ret_1m:.1%} in 1 month"
            )
            result['recommendation']= 'CASH MODE - No new buys'

        elif vix >= self.vix_extreme:
            result['regime']        = 'CRASH'
            result['can_trade']     = False
            result['reason']        = (
                f"Extreme fear: VIX={vix:.1f}"
            )
            result['recommendation']= 'CASH MODE - Extreme fear'

        elif ret_1m <= self.recovery_threshold or vix >= self.vix_high:
            result['regime']        = 'CAUTION'
            result['can_trade']     = True
            result['reason']        = (
                f"Cautious market: "
                f"SPY {ret_1m:.1%}, VIX={vix:.1f}"
            )
            result['recommendation']= 'REDUCED TRADING - High confidence only'

        else:
            result['regime']        = 'BULL'
            result['can_trade']     = True
            result['reason']        = (
                f"Bull market: "
                f"SPY {ret_1m:.1%}, VIX={vix:.1f}"
            )
            result['recommendation']= 'TRADE NORMALLY'

        # Print summary
        regime_emoji = {
            'BULL'   : 'BULL MARKET',
            'CAUTION': 'CAUTION',
            'BEAR'   : 'BEAR MARKET',
            'CRASH'  : 'MARKET CRASH',
        }.get(result['regime'], 'UNKNOWN')

        print(f"   Market Regime: {regime_emoji}")
        print(f"   SPY 1-Month:   {ret_1m:+.2%}")
        print(f"   SPY 3-Month:   {ret_3m:+.2%}")
        print(f"   VIX Level:     {vix:.1f}")
        print(f"   Can Trade:     {result['can_trade']}")
        print(f"   Reason:        {result['reason']}")
        print(f"   Action:        {result['recommendation']}")

        return result


def check_regime():
    """Quick regime check."""
    filter = MarketRegimeFilter()
    return filter.analyze()


if __name__ == '__main__':
    print("\nChecking market regime...")
    result = check_regime()
    print(f"\nFinal: {result['regime']} - {result['recommendation']}")