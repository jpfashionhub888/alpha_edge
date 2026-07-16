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

# Cache directory — avoids re-downloading on repeated runs
CACHE_DIR = Path(os.getenv('ALPHAEDGE_DATA_CACHE', 'backtesting/data/cache'))


class DataLoader:
    """
    Multi-source data loader with local caching.

    Usage:
        loader = DataLoader()
        data   = loader.get_ohlcv(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')
        # Returns {symbol: DataFrame(open, high, low, close, volume)}
    """

    def __init__(
        self,
        cache          : bool = True,
        polygon_api_key: Optional[str] = None,
    ):
        self.cache_enabled  = cache
        self.polygon_key    = polygon_api_key or os.getenv('POLYGON_API_KEY')
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

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
        Index: DatetimeIndex (trading days only)
        Adjusted for splits + dividends.
        """
        results = {}
        for sym in symbols:
            df = self._get_single_ohlcv(sym, start, end, interval)
            if df is not None and len(df) > 0:
                results[sym] = df
            else:
                logger.warning(f'{sym}: no data returned for {start}→{end}')
        return results

    def get_earnings_history(
        self,
        symbols: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical earnings (actual EPS vs consensus estimate).

        Returns {symbol: DataFrame} with columns:
          date, actual_eps, estimated_eps, surprise, surprise_pct

        Used for: SUE (Standardised Unexpected Earnings) signal.
        Data source: yfinance earnings history (free).
        Note: point-in-time earnings data requires Polygon; yfinance gives
              current revision of historical actuals which is slightly
              subject to restatement. Acceptable for initial signal research.
        """
        import yfinance as yf

        results = {}
        for sym in symbols:
            cache_path = CACHE_DIR / f'{sym}_earnings.parquet'
            if self.cache_enabled and cache_path.exists():
                # Only use cache if < 7 days old
                age_days = (time.time() - cache_path.stat().st_mtime) / 86400
                if age_days < 7:
                    try:
                        results[sym] = pd.read_parquet(cache_path)
                        continue
                    except Exception:
                        pass

            try:
                ticker = yf.Ticker(sym)

                # earnings_history: actual vs estimated EPS per quarter
                hist = ticker.earnings_history
                if hist is None or len(hist) == 0:
                    logger.debug(f'{sym}: no earnings history available')
                    continue

                # Normalise column names
                hist = hist.copy()
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                # Expected columns: epsactual, epsestimate, epsdifference, surprisepct
                col_map = {
                    'epsactual'  : 'actual_eps',
                    'epsestimate': 'estimated_eps',
                    'epsdifference': 'surprise',
                    'surprisepct': 'surprise_pct',
                }
                hist = hist.rename(columns={k: v for k, v in col_map.items() if k in hist.columns})

                if 'actual_eps' not in hist.columns:
                    logger.debug(f'{sym}: unexpected earnings columns: {list(hist.columns)}')
                    continue

                hist.index = pd.to_datetime(hist.index)
                hist = hist.sort_index()

                if self.cache_enabled:
                    hist.to_parquet(cache_path)

                results[sym] = hist
                time.sleep(0.1)   # rate limit

            except Exception as e:
                logger.warning(f'{sym}: earnings fetch failed ({e})')

        return results

    def get_universe(
        self,
        min_avg_volume_m: float = 5.0,   # minimum $5M avg daily volume
        use_sp500       : bool  = True,
    ) -> List[str]:
        """
        Build a tradeable universe.

        Default: S&P 500 constituents (well-known, liquid, no survivorship bias
        concern for the training period since we use current constituents).
        For research-grade backtesting, use a point-in-time universe instead.
        """
        if use_sp500:
            try:
                return self._fetch_sp500_symbols()
            except Exception as e:
                logger.warning(f'S&P 500 fetch failed ({e}) — using hardcoded fallback')

        # Fallback: liquid large-caps
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM',  'JNJ',  'V',     'PG',   'UNH',  'HD',   'MA',
            'BAC',  'ABBV', 'CVX',   'MRK',  'LLY',  'COST',
            'AVGO', 'PEP',  'KO',    'WMT',  'MCD',  'CRM',  'ACN',
            'TMO',  'DHR',  'NEE',   'ABT',  'TXN',  'LIN',  'QCOM',
            'PM',   'AMGN', 'RTX',   'CAT',  'HON',  'IBM',
        ]

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_single_ohlcv(
        self,
        symbol  : str,
        start   : str,
        end     : str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single symbol, checking cache first."""
        cache_key  = f'{symbol}_{start}_{end}_{interval}'
        cache_path = CACHE_DIR / f'{cache_key}.parquet'

        if self.cache_enabled and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        # Try Polygon first (if key available), fall back to yfinance
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

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.columns = [c.lower() for c in raw.columns]
            raw.index   = pd.to_datetime(raw.index)
            raw.index.name = 'date'

            required = ['open', 'high', 'low', 'close', 'volume']
            missing  = [c for c in required if c not in raw.columns]
            if missing:
                logger.warning(f'{symbol}: missing columns {missing}')
                return None

            return raw[required].dropna(subset=['close'])

        except Exception as e:
            logger.warning(f'{symbol}: yfinance fetch failed ({e})')
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
            df = pd.DataFrame([{
                'date'  : pd.Timestamp(b.timestamp, unit='ms'),
                'open'  : b.open,
                'high'  : b.high,
                'low'   : b.low,
                'close' : b.close,
                'volume': b.volume,
            } for b in bars])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.debug(f'{symbol}: Polygon fetch failed ({e}), falling back to yfinance')
            return None

    @staticmethod
    def _fetch_sp500_symbols() -> List[str]:
        """Fetch current S&P 500 constituents from Wikipedia."""
        table = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={'id': 'constituents'},
        )[0]
        symbols = table['Symbol'].str.replace('.', '-', regex=False).tolist()
        return [s.strip() for s in symbols if s.strip()]
