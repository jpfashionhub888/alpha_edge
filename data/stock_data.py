# data/stock_data.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches and processes stock market data.
    Uses yfinance (free, no API key needed).
    """

    def __init__(self, watchlist: list, lookback_days: int = 365):
        self.watchlist = watchlist
        self.lookback_days = lookback_days
        self.data_cache = {}

    def fetch_single_stock(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data for a single stock."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Keep only what we need
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Remove timezone info for consistency
            df.index = df.index.tz_localize(None)

            # Add symbol column
            df['symbol'] = symbol

            # Add basic derived columns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Cache it
            self.data_cache[symbol] = df

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all(self) -> dict:
        """Fetch data for all stocks in watchlist."""
        print(f"\n📊 Fetching data for {len(self.watchlist)} stocks...")
        all_data = {}

        for i, symbol in enumerate(self.watchlist):
            print(f"  [{i+1}/{len(self.watchlist)}] Fetching {symbol}...", end=" ")
            df = self.fetch_single_stock(symbol)

            if not df.empty:
                all_data[symbol] = df
                print(f"✅ {len(df)} rows")
            else:
                print("❌ Failed")

        print(f"\n✅ Successfully fetched {len(all_data)}/{len(self.watchlist)} stocks")
        return all_data

    def get_combined_data(self) -> pd.DataFrame:
        """Get all stock data combined into one DataFrame."""
        if not self.data_cache:
            self.fetch_all()

        frames = []
        for symbol, df in self.data_cache.items():
            frames.append(df)

        if frames:
            return pd.concat(frames)
        return pd.DataFrame()

    def get_latest_prices(self) -> dict:
        """Get the most recent price for each stock."""
        latest = {}
        for symbol, df in self.data_cache.items():
            if not df.empty:
                latest[symbol] = {
                    'price': df['close'].iloc[-1],
                    'change': df['returns'].iloc[-1],
                    'volume': df['volume'].iloc[-1],
                    'date': df.index[-1]
                }
        return latest


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fetcher = StockDataFetcher(
        watchlist=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        lookback_days=365
    )

    # Fetch all data
    data = fetcher.fetch_all()

    # Show sample
    for symbol, df in data.items():
        print(f"\n{symbol} - Last 5 days:")
        print(df[['close', 'volume', 'returns']].tail())

    # Latest prices
    print("\n📈 Latest Prices:")
    for symbol, info in fetcher.get_latest_prices().items():
        print(f"  {symbol}: ${info['price']:.2f} "
              f"({info['change']*100:+.2f}%)")