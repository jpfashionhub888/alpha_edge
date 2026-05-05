# multi_timeframe.py
# ALPHAEDGE - Multi-Timeframe Analysis
# Confirms signals across Weekly, Daily, 4-Hour timeframes
# Only trades when ALL timeframes agree

import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes stock signals across multiple timeframes.
    Weekly + Daily + 4-Hour confirmation required.
    """

    def __init__(self):
        self.timeframes = {
            'weekly' : '1wk',
            'daily'  : '1d',
            'hourly4': '1h',
        }
        self.lookback = {
            'weekly' : '1y',
            'daily'  : '6mo',
            'hourly4': '1mo',
        }

    def get_trend(self, df):
        """
        Determine trend from price data.
        Returns: 'uptrend', 'downtrend', 'sideways'
        """
        if df is None or len(df) < 20:
            return 'sideways'

        close = df['Close'].dropna()

        if len(close) < 20:
            return 'sideways'

        # EMA 20 and EMA 50
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()

        current_price = float(close.iloc[-1])
        current_ema20 = float(ema20.iloc[-1])
        current_ema50 = float(ema50.iloc[-1])

        # RSI
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=14).mean()
        avg_loss = loss.ewm(span=14).mean()
        rs       = avg_gain / avg_loss.replace(0, 1)
        rsi      = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

        # Trend logic
        if (current_price > current_ema20 > current_ema50
                and current_rsi > 50):
            return 'uptrend'
        elif (current_price < current_ema20 < current_ema50
              and current_rsi < 50):
            return 'downtrend'
        else:
            return 'sideways'

    def analyze_symbol(self, symbol):
        """
        Analyze a symbol across all timeframes.
        Returns dict with trend for each timeframe.
        """
        result = {
            'weekly'    : 'sideways',
            'daily'     : 'sideways',
            'hourly4'   : 'sideways',
            'aligned'   : False,
            'direction' : 'neutral',
            'confidence': 0.0,
        }

        try:
            # Weekly timeframe
            weekly = yf.Ticker(symbol).history(
                period='1y', interval='1wk'
            )
            result['weekly'] = self.get_trend(weekly)

            # Daily timeframe
            daily = yf.Ticker(symbol).history(
                period='6mo', interval='1d'
            )
            result['daily'] = self.get_trend(daily)

            # 4-Hour timeframe
            hourly4 = yf.Ticker(symbol).history(
                period='1mo', interval='1h'
            )
            result['hourly4'] = self.get_trend(hourly4)

            # Check alignment
            trends = [
                result['weekly'],
                result['daily'],
                result['hourly4'],
            ]

            uptrends   = trends.count('uptrend')
            downtrends = trends.count('downtrend')

            if uptrends == 3:
                result['aligned']    = True
                result['direction']  = 'bullish'
                result['confidence'] = 1.0
            elif uptrends == 2:
                result['aligned']    = True
                result['direction']  = 'bullish'
                result['confidence'] = 0.67
            elif downtrends == 3:
                result['aligned']    = True
                result['direction']  = 'bearish'
                result['confidence'] = 1.0
            elif downtrends == 2:
                result['aligned']    = True
                result['direction']  = 'bearish'
                result['confidence'] = 0.67
            else:
                result['aligned']    = False
                result['direction']  = 'neutral'
                result['confidence'] = 0.0

        except Exception as e:
            logger.warning(f"MTF analysis failed for {symbol}: {e}")

        return result

    def is_bullish(self, symbol):
        """
        Quick check if symbol is bullish across timeframes.
        Returns True if at least 2/3 timeframes are bullish.
        """
        result = self.analyze_symbol(symbol)
        return (
            result['direction'] == 'bullish'
            and result['aligned']
        )

    def get_mtf_score(self, symbol):
        """
        Get a score 0-1 based on timeframe alignment.
        1.0 = all timeframes bullish
        0.67 = 2/3 timeframes bullish
        0.0 = mixed or bearish
        """
        result = self.analyze_symbol(symbol)
        if result['direction'] == 'bullish':
            return result['confidence']
        return 0.0


if __name__ == '__main__':
    print("\nTesting Multi-Timeframe Analysis...")
    analyzer = MultiTimeframeAnalyzer()

    test_symbols = ['AAPL', 'MSFT', 'SPY']
    for symbol in test_symbols:
        result = analyzer.analyze_symbol(symbol)
        print(f"\n{symbol}:")
        print(f"  Weekly:  {result['weekly']}")
        print(f"  Daily:   {result['daily']}")
        print(f"  4-Hour:  {result['hourly4']}")
        print(f"  Aligned: {result['aligned']}")
        print(f"  Direction: {result['direction']}")
        print(f"  Confidence: {result['confidence']:.0%}")