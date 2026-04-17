# models/sector_rotation.py

"""
Sector Rotation Detector.
Tracks money flow between sectors.
When money flows INTO a sector, buy the best stocks there.
When money flows OUT, avoid that sector entirely.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SectorRotation:
    """
    Detects which sectors money is flowing into/out of.
    Uses sector ETFs as proxies.
    """

    def __init__(self):
        # Sector ETFs
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer': 'XLY',
            'Industrials': 'XLI',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communications': 'XLC',
            'Staples': 'XLP',
        }

        # Map stocks to sectors
        self.stock_sectors = {
            # Technology
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'NVDA': 'Technology',
            'AMD': 'Technology',
            'CRM': 'Technology',
            'SNOW': 'Technology',
            'NET': 'Technology',
            'DDOG': 'Technology',
            'CRWD': 'Technology',
            'META': 'Communications',
            'NFLX': 'Communications',

            # Consumer
            'AMZN': 'Consumer',
            'TSLA': 'Consumer',
            'HD': 'Consumer',
            'MCD': 'Consumer',
            'COST': 'Consumer',

            # Healthcare
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'UNH': 'Healthcare',
            'ABBV': 'Healthcare',
            'LLY': 'Healthcare',
            'MRK': 'Healthcare',

            # Financials
            'JPM': 'Financials',
            'V': 'Financials',
            'GS': 'Financials',
            'BAC': 'Financials',
            'MS': 'Financials',
            'SOFI': 'Financials',
            'HOOD': 'Financials',

            # Energy
            'XOM': 'Energy',
            'CVX': 'Energy',
            'OXY': 'Energy',

            # Consumer Staples
            'WMT': 'Staples',

            # Small Caps / Other
            'PLTR': 'Technology',
            'RIVN': 'Consumer',
            'MARA': 'Technology',

            # ETFs
            'SPY': 'Market',
            'QQQ': 'Technology',
            'IWM': 'Market',
            'DIA': 'Market',
        }

        self.sector_scores = {}
        self.sector_data = {}

    def analyze(self, lookback_days=90):
        """Analyze sector rotation and money flow."""

        print("\n🔄 Analyzing sector rotation...")

        for sector, etf in self.sector_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                df = ticker.history(period='6mo')

                if df.empty or len(df) < 20:
                    continue

                df.columns = [
                    c.lower().replace(' ', '_')
                    for c in df.columns
                ]

                close = df['close']
                volume = df['volume']

                # Short term momentum (10 days)
                mom_10 = (
                    close.iloc[-1] / close.iloc[-10] - 1
                )

                # Medium term momentum (21 days)
                mom_21 = (
                    close.iloc[-1] / close.iloc[-21] - 1
                )

                # Long term momentum (63 days)
                if len(close) >= 63:
                    mom_63 = (
                        close.iloc[-1] / close.iloc[-63] - 1
                    )
                else:
                    mom_63 = 0.0

                # Volume trend
                vol_recent = volume.iloc[-5:].mean()
                vol_older = volume.iloc[-21:-5].mean()
                vol_ratio = (
                    vol_recent / (vol_older + 1e-8)
                )

                # Relative strength vs SPY
                spy = yf.Ticker('SPY').history(period='6mo')
                if not spy.empty:
                    spy_close = spy['Close']
                    spy_mom = (
                        spy_close.iloc[-1]
                        / spy_close.iloc[-21] - 1
                    )
                    relative_strength = mom_21 - spy_mom
                else:
                    relative_strength = 0.0

                # Combined sector score
                # Positive = money flowing in
                # Negative = money flowing out
                score = (
                    mom_10 * 0.3
                    + mom_21 * 0.3
                    + mom_63 * 0.2
                    + relative_strength * 0.2
                )

                # Volume confirms the move
                if vol_ratio > 1.2:
                    score *= 1.3
                elif vol_ratio < 0.8:
                    score *= 0.7

                self.sector_scores[sector] = {
                    'score': score,
                    'momentum_10d': mom_10,
                    'momentum_21d': mom_21,
                    'momentum_63d': mom_63,
                    'volume_ratio': vol_ratio,
                    'relative_strength': relative_strength,
                    'etf': etf,
                    'flow': 'INFLOW' if score > 0.02
                            else 'OUTFLOW' if score < -0.02
                            else 'NEUTRAL',
                }

            except Exception as e:
                logger.warning(
                    f"Error analyzing {sector}: {e}"
                )

        # Print results
        sorted_sectors = sorted(
            self.sector_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        print("\n   Sector Rankings (best to worst):")
        print("   " + "-"*55)

        for sector, data in sorted_sectors:
            score = data['score']
            flow = data['flow']
            mom = data['momentum_21d']

            if flow == 'INFLOW':
                emoji = "🟢"
            elif flow == 'OUTFLOW':
                emoji = "🔴"
            else:
                emoji = "⚪"

            print(
                f"   {emoji} {sector:15s}"
                f" | Score: {score:+.3f}"
                f" | 21d: {mom:+.1%}"
                f" | {flow}"
            )

        return self.sector_scores

    def get_sector_for_stock(self, symbol):
        """Get which sector a stock belongs to."""

        return self.stock_sectors.get(symbol, 'Unknown')

    def get_sector_signal(self, symbol):
        """
        Get sector-based signal for a stock.
        Returns multiplier:
          > 1.0 = sector has inflow, boost signal
          = 1.0 = neutral
          < 1.0 = sector has outflow, reduce signal
        """

        sector = self.get_sector_for_stock(symbol)

        if sector == 'Unknown' or sector == 'Market':
            return 1.0

        if sector not in self.sector_scores:
            return 1.0

        data = self.sector_scores[sector]
        score = data['score']

        if score > 0.05:
            return 1.3
        elif score > 0.02:
            return 1.15
        elif score < -0.05:
            return 0.7
        elif score < -0.02:
            return 0.85
        else:
            return 1.0

    def get_top_sectors(self, n=3):
        """Return top N sectors with most inflow."""

        sorted_sectors = sorted(
            self.sector_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )

        return sorted_sectors[:n]

    def get_bottom_sectors(self, n=3):
        """Return bottom N sectors with most outflow."""

        sorted_sectors = sorted(
            self.sector_scores.items(),
            key=lambda x: x[1]['score']
        )

        return sorted_sectors[:n]