# data/crypto_data.py

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """
    Fetches crypto data from Coinbase.
    No API key needed for public data.
    """

    def __init__(self, watchlist=None, timeframe='4h',
                 lookback_days=365):

        if watchlist is None:
            watchlist = ['BTC/USDT']

        self.watchlist = watchlist
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.data_cache = {}
        self.exchange = None

        self._init_exchange()

    def _init_exchange(self):
        """Initialize exchange connection."""

        try:
            self.exchange = ccxt.coinbaseadvanced({
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.warning(
                f"Coinbase failed: {e}. Trying Coinbase."
            )
            try:
                self.exchange = ccxt.coinbase({
                    'enableRateLimit': True,
                })
            except Exception as e2:
                logger.warning(
                    f"Coinbase also failed: {e2}. "
                    f"Trying Kraken."
                )
                try:
                    self.exchange = ccxt.kraken({
                        'enableRateLimit': True,
                    })
                except Exception as e3:
                    logger.error(
                        f"All exchanges failed: {e3}"
                    )
                    self.exchange = None

    def fetch_single_crypto(self, symbol):
        """Fetch OHLCV data for one crypto pair."""

        if self.exchange is None:
            logger.error("No exchange available")
            return pd.DataFrame()

        try:
            since = self.exchange.parse8601(
                (datetime.now()
                 - timedelta(days=self.lookback_days))
                .strftime('%Y-%m-%dT%H:%M:%S')
            )

            all_candles = []
            current_since = since

            while True:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.timeframe,
                    since=current_since,
                    limit=300
                )

                if not candles:
                    break

                all_candles.extend(candles)
                current_since = candles[-1][0] + 1

                if len(candles) < 300:
                    break

                time.sleep(0.5)

            if not all_candles:
                return pd.DataFrame()

            df = pd.DataFrame(
                all_candles,
                columns=[
                    'timestamp', 'open', 'high',
                    'low', 'close', 'volume'
                ]
            )

            df['timestamp'] = pd.to_datetime(
                df['timestamp'], unit='ms'
            )
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]

            df['symbol'] = symbol
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(
                df['close'] / df['close'].shift(1)
            )

            self.data_cache[symbol] = df
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all(self):
        """Fetch data for all crypto in watchlist."""

        print(
            f"\n🪙 Fetching data for"
            f" {len(self.watchlist)} crypto pairs..."
        )
        all_data = {}

        for i, symbol in enumerate(self.watchlist):
            n = i + 1
            total = len(self.watchlist)
            print(
                f"   [{n}/{total}] Fetching {symbol}...",
                end=" "
            )
            df = self.fetch_single_crypto(symbol)

            if not df.empty:
                all_data[symbol] = df
                print(f"✅ {len(df)} candles")
            else:
                print("❌ Failed")

            time.sleep(1)

        n_ok = len(all_data)
        n_total = len(self.watchlist)
        print(f"\n   ✅ Fetched {n_ok}/{n_total} pairs")

        return all_data

    def get_latest_prices(self):
        """Get most recent price for each crypto."""

        latest = {}
        for symbol, df in self.data_cache.items():
            if not df.empty:
                latest[symbol] = {
                    'price': df['close'].iloc[-1],
                    'change_24h': df['returns'].iloc[-1],
                }
        return latest