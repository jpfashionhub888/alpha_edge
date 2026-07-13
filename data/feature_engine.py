# data/feature_engine.py
# FIXED VERSION — All look-ahead bias removed
#
# Fix 1.2 Audit (alphaedge_roadmap.md):
# FeatureEngine contains NO global scalers (StandardScaler / MinMaxScaler).
# All normalization is purely rolling-window arithmetic:
#   - price_vs_ma{N}  = (close - rolling_mean) / rolling_mean
#   - volatility_Nd   = returns.rolling(N).std() * sqrt(252)
#   - volume_ratio_Nd = volume / rolling_mean(volume)
# These computations use ONLY past bars at each timestamp, making them
# safe for walk-forward backtesting. No future data leaks here.

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

# Alpha158 — Microsoft qlib feature set (158 institutional-grade features)
try:
    from data.alpha158 import build_alpha158 as _build_alpha158
    _ALPHA158_AVAILABLE = True
except ImportError:
    try:
        from alpha158 import build_alpha158 as _build_alpha158
        _ALPHA158_AVAILABLE = True
    except ImportError:
        _ALPHA158_AVAILABLE = False
        logger.warning('alpha158 module not found — running with base features only')


class FeatureEngine:
    """
    Transforms raw OHLCV data into ML-ready features.
    All features are strictly point-in-time safe.
    No future data leaks into any calculation.
    """

    def __init__(self):
        self.feature_names = []

    def add_all_features(
        self,
        df: pd.DataFrame,
        end_date: str = None    # ← NEW: for backtest safety
    ) -> pd.DataFrame:
        """Add all features to a dataframe."""

        df = df.copy()

        # ✅ Ensure returns exists before volume features
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(
                df['close'] / df['close'].shift(1)
            )

        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_pattern_features(df)
        df = self._add_market_context(df, end_date)

        # ── Alpha158 (Microsoft qlib) ─────────────────────────────────────
        # Adds 158 institutional-grade features after base TA features.
        # Only new alpha_ columns are merged; no existing columns overwritten.
        if _ALPHA158_AVAILABLE:
            try:
                alpha_df = _build_alpha158(df)
                new_cols = [c for c in alpha_df.columns
                            if c.startswith('alpha_') and c not in df.columns]
                df = pd.concat([df, alpha_df[new_cols]], axis=1)
                logger.info(f'Alpha158: merged {len(new_cols)} new features')
            except Exception as e:
                logger.warning(f'Alpha158 build failed (non-fatal): {e}')

        df = self._add_target_variable(df)

        df.dropna(inplace=True)

        self.feature_names = [
            col for col in df.columns
            if col not in [
                'open', 'high', 'low', 'close',
                'volume', 'symbol', 'returns',
                'log_returns', 'target',
                'future_return'     # safety: exclude if present
            ]
        ]

        logger.info(
            f"Generated {len(self.feature_names)} features "
            f"({'with' if _ALPHA158_AVAILABLE else 'without'} Alpha158)"
        )

        return df

    def _add_price_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Price-based features — all clean."""

        for period in [1, 2, 3, 5, 10, 21, 63]:
            df[f'return_{period}d'] = (
                df['close'].pct_change(period)
            )

        for period in [5, 10, 21, 50, 200]:
            ma = df['close'].rolling(period).mean()
            df[f'price_vs_ma{period}'] = (
                (df['close'] - ma) / (ma + 1e-8)
            )

        df['ma_5_21_cross'] = (
            df['close'].rolling(5).mean()
            - df['close'].rolling(21).mean()
        )
        df['ma_21_50_cross'] = (
            df['close'].rolling(21).mean()
            - df['close'].rolling(50).mean()
        )

        for period in [5, 10, 21]:
            highest = df['high'].rolling(period).max()
            lowest = df['low'].rolling(period).min()
            df[f'price_position_{period}d'] = (
                (df['close'] - lowest)
                / (highest - lowest + 1e-8)
            )

        df['gap'] = (
            (df['open'] - df['close'].shift(1))
            / (df['close'].shift(1) + 1e-8)
        )

        return df

    def _add_volume_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Volume-based features — all clean."""

        for period in [5, 10, 21]:
            vol_ma = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}d'] = (
                df['volume'] / (vol_ma + 1e-8)
            )

        df['volume_trend'] = (
            df['volume'].rolling(5).mean()
            / (df['volume'].rolling(21).mean() + 1e-8)
        )

        # ✅ returns now guaranteed to exist
        df['price_volume_corr'] = (
            df['returns'].rolling(21).corr(
                df['volume'].pct_change()
            )
        )

        obv = (
            np.sign(df['returns']) * df['volume']
        ).cumsum()
        df['obv_ratio'] = (
            obv / (obv.rolling(21).mean() + 1e-8)
        )

        return df

    def _add_trend_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Trend indicators — all clean."""

        adx = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close']
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # ⚠️ Ichimoku removed — forward projection risk
        # Uncomment only if you verify your ta version
        # is point-in-time safe
        # ichi = ta.trend.IchimokuIndicator(
        #     df['high'], df['low']
        # )
        # df['ichimoku_a'] = ichi.ichimoku_a()
        # df['ichimoku_b'] = ichi.ichimoku_b()

        aroon = ta.trend.AroonIndicator(
            df['high'], df['low'], window=25
        )
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()

        return df

    def _add_momentum_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Momentum indicators — all clean."""

        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = (
                ta.momentum.RSIIndicator(
                    df['close'], window=period
                ).rsi()
            )

        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['williams_r'] = (
            ta.momentum.WilliamsRIndicator(
                df['high'], df['low'], df['close']
            ).williams_r()
        )

        for period in [5, 10, 21]:
            df[f'roc_{period}'] = (
                ta.momentum.ROCIndicator(
                    df['close'], window=period
                ).roc()
            )

        df['cci'] = ta.trend.CCIIndicator(
            df['high'], df['low'], df['close']
        ).cci()

        return df

    def _add_volatility_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Volatility indicators — all clean."""

        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (
            (df['bb_high'] - df['bb_low'])
            / (df['close'] + 1e-8)
        )
        df['bb_position'] = (
            (df['close'] - df['bb_low'])
            / (df['bb_high'] - df['bb_low'] + 1e-8)
        )

        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        df['atr_ratio'] = (
            df['atr'] / (df['close'] + 1e-8)
        )

        for period in [5, 10, 21, 63]:
            df[f'volatility_{period}d'] = (
                df['returns'].rolling(period).std()
                * np.sqrt(252)
            )

        df['vol_ratio'] = (
            df['volatility_5d']
            / (df['volatility_63d'] + 1e-8)
        )

        return df

    def _add_pattern_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Pattern features — all clean."""

        df['candle_body'] = (
            abs(df['close'] - df['open'])
            / (df['open'] + 1e-8)
        )

        df['upper_shadow'] = (
            (
                df['high']
                - df[['open', 'close']].max(axis=1)
            )
            / (df['open'] + 1e-8)
        )

        df['lower_shadow'] = (
            (
                df[['open', 'close']].min(axis=1)
                - df['low']
            )
            / (df['open'] + 1e-8)
        )

        df['up_day'] = (
            df['returns'] > 0
        ).astype(int)
        df['consecutive_up'] = (
            df['up_day'].groupby(
                (
                    df['up_day']
                    != df['up_day'].shift()
                ).cumsum()
            ).cumsum() * df['up_day']
        )

        df['down_day'] = (
            df['returns'] < 0
        ).astype(int)
        df['consecutive_down'] = (
            df['down_day'].groupby(
                (
                    df['down_day']
                    != df['down_day'].shift()
                ).cumsum()
            ).cumsum() * df['down_day']
        )

        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

        return df

    def _add_market_context(
        self,
        df: pd.DataFrame,
        end_date: str = None    # ← FIXED: date boundary
    ) -> pd.DataFrame:
        """
        Cross-asset context features.
        Fixed to never use data beyond end_date.
        """

        import yfinance as yf

        context_tickers = {
            'vix': '^VIX',
            'dxy': 'DX-Y.NYB',
            'tlt': 'TLT',
            'gld': 'GLD',
        }

        # ✅ Strict date boundary
        max_date = (
            pd.Timestamp(end_date)
            if end_date
            else df.index.max()
        )

        if not hasattr(self, '_context_cache'):
            self._context_cache = {}

        for name, ticker in context_tickers.items():
            try:
                if ticker not in self._context_cache:
                    # Fetch from 2020-01-01 up to today once
                    data = yf.Ticker(ticker).history(
                        start='2020-01-01',
                        progress=False
                    )
                    if not data.empty:
                        self._context_cache[ticker] = data

                data = self._context_cache.get(ticker)
                if data is None or data.empty:
                    continue

                close = data['Close'].copy()
                close.index = close.index.tz_localize(
                    None
                )

                # ✅ Hard cutoff — no future data
                close = close[close.index <= max_date]

                aligned = close.reindex(
                    df.index, method='ffill'
                )

                df[f'{name}_level'] = aligned
                df[f'{name}_return'] = (
                    aligned.pct_change()
                )
                df[f'{name}_ma_ratio'] = (
                    aligned
                    / (aligned.rolling(21).mean() + 1e-8)
                )

            except Exception as e:
                logger.debug(f'Context ticker {ticker} fetch failed: {e}')  # non-blocking

        return df

    def _add_target_variable(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Only create target if it does not already exist.
        If caller pre-computed target (e.g. walk-forward optimizer),
        respect that and do not overwrite with future data.
        """
        if 'target' in df.columns:
            # Target already set by caller — do not overwrite
            # Overwriting would use shift(-5) on test rows = look-ahead
            return df

        forward_period = 5

        df['future_return'] = (
            df['close']
            .pct_change(forward_period)
            .shift(-forward_period)
        )

        # Drop rows where future_return is NaN (the last forward_period rows
        # have no forward data). Without this, NaN > 0 evaluates to False,
        # so those rows get target=0 regardless of actual future price movement.
        df = df[df['future_return'].notna()]

        df['target'] = (df['future_return'] > 0).astype(int)

        # CRITICAL: