# backtesting/data/loader.py
"""
Data loader for backtesting.

Primary source: yfinance (free, reliable for daily OHLCV + earnings data)
Production source: Polygon.io (when API key is available)

All data returned is adjusted for splits and dividends.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv('ALPHAEDGE_DATA_CACHE', 'backtesting/data/cache'))


class DataLoader:
    """
    Multi-source data loader with local caching.

    Usage:
        loader = DataLoader()
        data   = loader.get_ohlcv(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')
    """

    def __init__(
        self,
        cache          : bool = True,
        polygon_api_key: Optional[str] = None,
    ):
        self.cache_enabled  = cache
        self.polygon_key    = polygon_api_key or os.getenv('POLYGON_API_KEY')
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_ohlcv(
        self,
        symbols   : List[str],
        start     : str,
        end       : str,
        interval  : str = '1d',
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for a list of symbols.
        Returns {symbol: DataFrame} with columns: open, high, low, close, volume
        """
        results = {}
        for sym in symbols:
            df = self._get_single_ohlcv(sym, start, end, interval)
            if df is not None and len(df) > 0:
                results[sym] = df
            else:
                logger.warning('%s: no data returned for %s to %s', sym, start, end)
        return results

    def get_earnings_history(
        self,
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical earnings using actual announcement dates.

        Uses ticker.get_earnings_dates(limit=24) which returns:
          - Actual announcement dates (not fiscal quarter ends)
          - EPS Estimate vs Reported EPS
          - Up to 24 quarters of history

        Returns {symbol: DataFrame} with columns:
          actual_eps, estimated_eps, surprise_pct
        Index: DatetimeIndex of announcement dates (timezone-naive)
        """
        import yfinance as yf

        results = {}
        for sym in symbols:
            cache_path = CACHE_DIR / (sym + '_earnings.parquet')
            if self.cache_enabled and cache_path.exists():
                age_days = (time.time() - cache_path.stat().st_mtime) / 86400
                if age_days < 7:
                    try:
                        results[sym] = pd.read_parquet(cache_path)
                        continue
                    except Exception:
                        pass

            try:
                ticker = yf.Ticker(sym)
                hist = ticker.get_earnings_dates(limit=24)

                if hist is None or len(hist) == 0:
                    logger.debug('%s: no earnings dates available', sym)
                    continue

                hist = hist.copy()

                # Strip timezone from index
                if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)

                # Rename columns to internal names
                col_map = {
                    'Reported EPS' : 'actual_eps',
                    'EPS Estimate' : 'estimated_eps',
                    'Surprise(%)'  : 'surprise_pct',
                }
                hist = hist.rename(columns={k: v for k, v in col_map.items() if k in hist.columns})

                # Drop future earnings (no actual yet)
                if 'actual_eps' in hist.columns:
                    hist = hist[hist['actual_eps'].notna()]

                if len(hist) == 0:
                    logger.debug('%s: no past earnings with actuals', sym)
                    continue

                # Compute dollar surprise
                if 'estimated_eps' in hist.columns:
                    hist['surprise'] = hist['actual_eps'] - hist['estimated_eps']

                hist.index = pd.to_datetime(hist.index)
                hist = hist.sort_index()

                if self.cache_enabled:
                    hist.to_parquet(cache_path)

                results[sym] = hist
                time.sleep(0.15)

            except Exception as e:
                logger.warning('%s: earnings fetch failed (%s)', sym, e)

        return results

    def get_universe(
        self,
        min_avg_volume_m: float = 5.0,
        use_sp500       : bool  = True,
    ) -> List[str]:
        """Build a tradeable universe."""
        if use_sp500:
            try:
                return self._fetch_sp500_symbols()
            except Exception as e:
                logger.warning('S&P 500 fetch failed (%s) -- using fallback', e)

        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM',  'JNJ',  'V',     'PG',   'UNH',  'HD',   'MA',
            'BAC',  'ABBV', 'CVX',   'MRK',  'LLY',  'COST',
            'AVGO', 'PEP',  'KO',    'WMT',  'MCD',  'CRM',  'ACN',
            'TMO',  'DHR',  'NEE',   'ABT',  'TXN',  'QCOM',
            'PM',   'AMGN', 'RTX',   'CAT',  'HON',  'IBM',
        ]

    def _get_single_ohlcv(
        self,
        symbol  : str,
        start   : str,
        end     : str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        cache_key  = '%s_%s_%s_%s' % (symbol, start, end, interval)
        cache_path = CACHE_DIR / (cache_key + '.parquet')

        if self.cache_enabled and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        df = None
        if self.polygon_key:
            df = self._fetch_polygon(symbol, start, end)

        if df is None:
            df = self._fetch_yfinance(symbol, start, end, interval)

        if df is not None and len(df) > 0 and self.cache_enabled:
            df.to_parquet(cache_path)

        return df

    def _fetch_yfinance(
        self,
        symbol  : str,
        start   : str,
        end     : str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            raw = yf.download(
                symbol,
                start       = start,
                end         = end,
                interval    = interval,
                auto_adjust = True,
                progress    = False,
            )
            if raw is None or len(raw) == 0:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [c.lower() for c in raw.columns]
            raw.index   = pd.to_datetime(raw.index)
            raw.index.name = 'date'

            required = ['open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in raw.columns]
            if missing:
                logger.warning('%s: missing columns %s', symbol, missing)
                return None

            return raw[required].dropna(subset=['close'])

        except Exception as e:
            logger.warning('%s: yfinance fetch failed (%s)', symbol, e)
            return None

    def _fetch_polygon(
        self,
        symbol: str,
        start : str,
        end   : str,
    ) -> Optional[pd.DataFrame]:
        try:
            from polygon import RESTClient
            client = RESTClient(self.polygon_key)
            bars   = client.get_aggs(symbol, 1, 'day', start, end, adjusted=True)
            if not bars:
                return None
            rows = []
            for b in bars:
                rows.append({
                    'date'  : pd.Timestamp(b.timestamp, unit='ms'),
                    'open'  : b.open,
                    'high'  : b.high,
                    'low'   : b.low,
                    'close' : b.close,
                    'volume': b.volume,
                })
            df = pd.DataFrame(rows)
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.debug('%s: Polygon fetch failed (%s)', symbol, e)
            return None

    @staticmethod
    def _fetch_sp500_symbols() -> List[str]:
        """Fetch current S&P 500 constituents from Wikipedia."""
        import lxml  # noqa: F401 -- required by pd.read_html
        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={'id': 'constituents'},
        )[0]
        symbols = table['Symbol'].str.replace('.', '-', regex=False).tolist()
        return [s.strip() for s in symbols if s.strip()]
