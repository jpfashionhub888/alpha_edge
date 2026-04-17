# data/feature_engine.py

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Transforms raw OHLCV data into ML-ready features.
    Includes price, volume, technical indicators,
    volatility, patterns and cross-asset context.
    """

    def __init__(self):
        self.feature_names = []

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to a dataframe."""

        df = df.copy()

        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_pattern_features(df)
        df = self._add_market_context(df)
        df = self._add_target_variable(df)

        df.dropna(inplace=True)

        self.feature_names = [
            col for col in df.columns
            if col not in [
                'open', 'high', 'low', 'close',
                'volume', 'symbol', 'returns',
                'log_returns', 'target', 'future_return'
            ]
        ]

        logger.info(
            f"Generated {len(self.feature_names)} features"
        )

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""

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

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""

        for period in [5, 10, 21]:
            vol_ma = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}d'] = (
                df['volume'] / (vol_ma + 1e-8)
            )

        df['volume_trend'] = (
            df['volume'].rolling(5).mean()
            / (df['volume'].rolling(21).mean() + 1e-8)
        )

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

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following indicators."""

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

        ichi = ta.trend.IchimokuIndicator(
            df['high'], df['low']
        )
        df['ichimoku_a'] = ichi.ichimoku_a()
        df['ichimoku_b'] = ichi.ichimoku_b()

        aroon = ta.trend.AroonIndicator(
            df['high'], df['low'], window=25
        )
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators."""

        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                df['close'], window=period
            ).rsi()

        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close']
        ).williams_r()

        for period in [5, 10, 21]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(
                df['close'], window=period
            ).roc()

        df['cci'] = ta.trend.CCIIndicator(
            df['high'], df['low'], df['close']
        ).cci()

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators."""

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

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick and pattern features."""

        df['candle_body'] = (
            abs(df['close'] - df['open'])
            / (df['open'] + 1e-8)
        )

        df['upper_shadow'] = (
            (df['high'] - df[['open', 'close']].max(axis=1))
            / (df['open'] + 1e-8)
        )

        df['lower_shadow'] = (
            (df[['open', 'close']].min(axis=1) - df['low'])
            / (df['open'] + 1e-8)
        )

        df['up_day'] = (df['returns'] > 0).astype(int)
        df['consecutive_up'] = (
            df['up_day'].groupby(
                (df['up_day'] != df['up_day'].shift()).cumsum()
            ).cumsum() * df['up_day']
        )

        df['down_day'] = (df['returns'] < 0).astype(int)
        df['consecutive_down'] = (
            df['down_day'].groupby(
                (df['down_day'] != df['down_day'].shift()).cumsum()
            ).cumsum() * df['down_day']
        )

        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

        return df

    def _add_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-asset market context features.
        VIX fear index, Dollar index, Treasury bonds
        and Gold are among the most powerful predictors
        in all of finance. Worth more than any extra model.
        """

        import yfinance as yf

        context_tickers = {
            'vix': '^VIX',
            'dxy': 'DX-Y.NYB',
            'tlt': 'TLT',
            'gld': 'GLD',
        }

        for name, ticker in context_tickers.items():
            try:
                data = yf.Ticker(ticker).history(
                    period='2y'
                )

                if data.empty:
                    continue

                close = data['Close']
                close.index = close.index.tz_localize(None)

                # Align with our dataframe dates
                aligned = close.reindex(
                    df.index, method='ffill'
                )

                # Raw level
                df[f'{name}_level'] = aligned

                # Daily return
                df[f'{name}_return'] = aligned.pct_change()

                # Position relative to moving average
                df[f'{name}_ma_ratio'] = (
                    aligned
                    / (aligned.rolling(21).mean() + 1e-8)
                )

            except Exception:
                pass

        return df

    def _add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for ML prediction."""

        forward_period = 5
        df['future_return'] = (
            df['close']
            .pct_change(forward_period)
            .shift(-forward_period)
        )
        df['target'] = (
            df['future_return'] > 0
        ).astype(int)

        return df

    def get_feature_names(self) -> list:
        """Return list of feature column names."""

        return self.feature_names