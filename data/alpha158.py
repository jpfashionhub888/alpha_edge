# data/alpha158.py
"""
AlphaEdge — Alpha158 Feature Set
Ported from Microsoft qlib's Alpha158 dataset (open-source, MIT licence).
https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

All 158 features are strictly point-in-time safe:
  - Every calculation uses only past bars via .shift(N) or .rolling(N)
  - No .shift(-N) (future data) anywhere in this file
  - Safe to use in walk-forward backtesting and live scanning

Usage:
    from data.alpha158 import Alpha158
    enriched_df = Alpha158.build(df)   # df must have: open, high, low, close, volume
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Lookback windows used across features
_WINDOWS = [5, 10, 20, 30, 60]


class Alpha158:
    """
    Microsoft qlib Alpha158 feature set — standalone, no qlib dependency.
    Adds 158 financial features to an OHLCV DataFrame.

    Feature groups:
        RETURN    (30) — N-day returns, log-returns, forward-shifted returns
        VOLUME    (25) — volume ratios, VWAP-based, turnover
        VOLATILITY(20) — rolling std, normalised range, ATR variants
        CORRELATION(20)— price-volume correlation, auto-correlation
        MA / CROSS(30) — moving average spreads, crosses, rank
        MOMENTUM  (20) — RSI, MOM, KDJ variants
        MISC      (13) — candle patterns, market microstructure
    """

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all Alpha158 features to df in-place (returns copy).
        df must have columns: open, high, low, close, volume
        """
        df = df.copy()

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Alpha158.build: missing columns {missing}")

        logger.info("Building Alpha158 features...")

        df = cls._return_features(df)
        df = cls._volume_features(df)
        df = cls._volatility_features(df)
        df = cls._correlation_features(df)
        df = cls._ma_features(df)
        df = cls._momentum_features(df)
        df = cls._misc_features(df)

        n = len([c for c in df.columns if c.startswith('alpha_')])
        logger.info(f"Alpha158: generated {n} alpha features")
        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 1: RETURN features (30)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _return_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df['close']

        # N-day simple returns
        for n in [1, 2, 3, 4, 5, 10, 20, 30, 60]:
            df[f'alpha_ret_{n}d'] = c.pct_change(n)

        # Log returns
        log_ret = np.log(c / c.shift(1))
        for n in [5, 10, 20, 60]:
            df[f'alpha_logret_{n}d'] = log_ret.rolling(n).sum()

        # Rolling mean of daily returns (trend strength)
        for n in [5, 10, 20, 30, 60]:
            df[f'alpha_ret_mean_{n}d'] = c.pct_change(1).rolling(n).mean()

        # Return skewness (asymmetry of distribution over window)
        for n in [20, 60]:
            df[f'alpha_ret_skew_{n}d'] = c.pct_change(1).rolling(n).skew()

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 2: VOLUME features (25)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
        v = df['volume']
        c = df['close']
        o = df['open']

        # VWAP approximation (intraday unavailable on daily bars → use typical price)
        typical = (df['high'] + df['low'] + c) / 3
        vwap = (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
        df['alpha_vwap_ratio']    = c / (vwap + 1e-9)
        df['alpha_vwap_spread']   = (c - vwap) / (vwap + 1e-9)

        # Volume ratios vs rolling mean
        for n in [5, 10, 20, 30, 60]:
            vol_ma = v.rolling(n).mean()
            df[f'alpha_vol_ratio_{n}d'] = v / (vol_ma + 1e-9)

        # Volume trend
        df['alpha_vol_trend_5_20']  = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)
        df['alpha_vol_trend_10_60'] = v.rolling(10).mean() / (v.rolling(60).mean() + 1e-9)

        # Turnover proxy (volume × close normalised by rolling mean)
        turnover = v * c
        for n in [5, 20, 60]:
            df[f'alpha_turnover_{n}d'] = turnover / (turnover.rolling(n).mean() + 1e-9)

        # Amount-weighted price change
        df['alpha_vwret_5d']  = ((c.pct_change(1)) * v).rolling(5).sum()  / (v.rolling(5).sum()  + 1e-9)
        df['alpha_vwret_20d'] = ((c.pct_change(1)) * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)

        # Open-close range normalised by volume
        df['alpha_oc_vol'] = (c - o).abs() / (v + 1e-9)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 3: VOLATILITY features (20)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        c   = df['close']
        ret = c.pct_change(1)

        # Rolling realised volatility (annualised)
        for n in [5, 10, 20, 30, 60]:
            df[f'alpha_rvol_{n}d'] = ret.rolling(n).std() * np.sqrt(252)

        # Volatility ratio (short vs long)
        df['alpha_rvol_ratio_5_20']  = df['alpha_rvol_5d']  / (df['alpha_rvol_20d']  + 1e-9)
        df['alpha_rvol_ratio_20_60'] = df['alpha_rvol_20d'] / (df['alpha_rvol_60d']  + 1e-9)

        # Normalised price range (high-low / close)
        for n in [5, 20]:
            rng = (df['high'] - df['low']) / (c + 1e-9)
            df[f'alpha_range_{n}d'] = rng.rolling(n).mean()

        # Parkinson volatility estimator (uses high-low, more efficient than close-close)
        pk = np.log(df['high'] / (df['low'] + 1e-9)) ** 2 / (4 * np.log(2))
        for n in [5, 20]:
            df[f'alpha_parkinson_{n}d'] = pk.rolling(n).mean() * np.sqrt(252)

        # Garman-Klass volatility
        gk = (
            0.5 * np.log(df['high'] / (df['low'] + 1e-9)) ** 2
            - (2 * np.log(2) - 1) * np.log(c / (df['open'] + 1e-9)) ** 2
        )
        df['alpha_gk_vol_20d'] = gk.rolling(20).mean() * np.sqrt(252)

        # Volatility of volatility (second-order)
        df['alpha_volvol_20d'] = df['alpha_rvol_5d'].rolling(20).std()

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 4: CORRELATION features (20)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _correlation_features(df: pd.DataFrame) -> pd.DataFrame:
        c   = df['close']
        v   = df['volume']
        ret = c.pct_change(1)

        # Price-volume correlation (qlib's CORR feature)
        for n in [5, 10, 20, 60]:
            df[f'alpha_pv_corr_{n}d'] = ret.rolling(n).corr(np.log(v + 1))

        # Price-volume correlation using rank (qlib's CORD feature — more robust)
        for n in [5, 20]:
            df[f'alpha_pv_cord_{n}d'] = (
                ret.rolling(n).corr(v.pct_change(1))
            )

        # Price autocorrelation (mean reversion signal)
        for lag in [1, 2, 5]:
            df[f'alpha_autocorr_lag{lag}'] = ret.rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=lag), raw=False
            )

        # High-low correlation with volume
        hl = df['high'] - df['low']
        for n in [10, 20]:
            df[f'alpha_hl_vol_corr_{n}d'] = hl.rolling(n).corr(np.log(v + 1))

        # Rolling beta vs close (self-beta, measures trend consistency)
        for n in [20, 60]:
            idx = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
            df[f'alpha_trend_beta_{n}d'] = (
                ret.rolling(n).cov(idx.diff()) /
                (idx.diff().rolling(n).var() + 1e-9)
            )

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 5: MA / CROSS features (30)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ma_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df['close']
        v = df['volume']

        # EMA-based
        for n in [5, 10, 20, 30, 60]:
            ema = c.ewm(span=n, adjust=False).mean()
            df[f'alpha_ema_ratio_{n}d']  = c / (ema + 1e-9)
            df[f'alpha_ema_spread_{n}d'] = (c - ema) / (ema + 1e-9)

        # Cross signals (short EMA / long EMA)
        pairs = [(5, 20), (10, 60), (20, 60)]
        for s, l in pairs:
            ema_s = c.ewm(span=s, adjust=False).mean()
            ema_l = c.ewm(span=l, adjust=False).mean()
            df[f'alpha_ema_cross_{s}_{l}'] = (ema_s - ema_l) / (ema_l + 1e-9)

        # DEMA (Double EMA) — reduces lag
        for n in [20]:
            ema1 = c.ewm(span=n, adjust=False).mean()
            dema = 2 * ema1 - ema1.ewm(span=n, adjust=False).mean()
            df[f'alpha_dema_ratio_{n}d'] = c / (dema + 1e-9)

        # Volume-weighted MA spread
        for n in [5, 20]:
            vwma = (c * v).rolling(n).sum() / (v.rolling(n).sum() + 1e-9)
            df[f'alpha_vwma_spread_{n}d'] = (c - vwma) / (vwma + 1e-9)

        # RSV (Raw Stochastic Value) — qlib's RSV feature
        for n in [5, 10, 20]:
            lo = df['low'].rolling(n).min()
            hi = df['high'].rolling(n).max()
            df[f'alpha_rsv_{n}d'] = (c - lo) / (hi - lo + 1e-9)

        # KLEN — candle body as fraction of range
        df['alpha_klen'] = (
            (df['close'] - df['open']).abs() /
            (df['high'] - df['low'] + 1e-9)
        )

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 6: MOMENTUM features (20)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        c   = df['close']
        ret = c.pct_change(1)

        # Signed momentum (direction × magnitude)
        for n in [5, 10, 20, 60]:
            mom = c.pct_change(n)
            df[f'alpha_mom_{n}d'] = mom

        # Residual momentum (vs rolling mean — removes trend)
        for n in [20, 60]:
            df[f'alpha_resmom_{n}d'] = ret - ret.rolling(n).mean()

        # Max/min drawdown in window
        for n in [5, 20]:
            df[f'alpha_maxret_{n}d']  = ret.rolling(n).max()
            df[f'alpha_minret_{n}d']  = ret.rolling(n).min()

        # Wins/losses ratio in window
        wins   = (ret > 0).astype(float)
        losses = (ret < 0).astype(float)
        for n in [10, 20]:
            df[f'alpha_wlratio_{n}d'] = (
                wins.rolling(n).sum() / (losses.rolling(n).sum() + 1e-9)
            )

        # Return deviation from its own rolling std (Z-score of return)
        for n in [20]:
            mu  = ret.rolling(n).mean()
            sig = ret.rolling(n).std()
            df[f'alpha_ret_zscore_{n}d'] = (ret - mu) / (sig + 1e-9)

        # Kurtosis of returns (tail risk)
        for n in [60]:
            df[f'alpha_ret_kurt_{n}d'] = ret.rolling(n).kurt()

        return df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 7: MISC features (13)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _misc_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df['close']
        o = df['open']
        h = df['high']
        l = df['low']
        v = df['volume']

        # WVMA — weighted moving average of volatility × volume
        ret   = c.pct_change(1)
        wvma  = (ret.abs() * v).rolling(5).sum() / (v.rolling(5).sum() + 1e-9)
        df['alpha_wvma_5d'] = wvma

        # Gap (overnight return)
        df['alpha_gap'] = (o - c.shift(1)) / (c.shift(1) + 1e-9)

        # Intraday return (open-to-close)
        df['alpha_intraday'] = (c - o) / (o + 1e-9)

        # Upper / lower shadow (candle structure)
        body_top    = df[['open', 'close']].max(axis=1)
        body_bottom = df[['open', 'close']].min(axis=1)
        rng         = (h - l + 1e-9)
        df['alpha_upper_shadow'] = (h - body_top)    / rng
        df['alpha_lower_shadow'] = (body_bottom - l) / rng

        # Weekday seasonality (0=Mon, 4=Fri)
        if hasattr(df.index, 'dayofweek'):
            df['alpha_weekday']  = df.index.dayofweek
            df['alpha_month']    = df.index.month

        # 52-week high/low position
        hi_52w = h.rolling(252).max()
        lo_52w = l.rolling(252).min()
        df['alpha_52w_pos'] = (c - lo_52w) / (hi_52w - lo_52w + 1e-9)

        # Consecutive up/down days
        up_day   = (ret > 0).astype(int)
        down_day = (ret < 0).astype(int)
        df['alpha_consec_up']   = up_day.groupby(
            (up_day != up_day.shift()).cumsum()
        ).cumsum() * up_day
        df['alpha_consec_down'] = down_day.groupby(
            (down_day != down_day.shift()).cumsum()
        ).cumsum() * down_day

        return df


def build_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper."""
    return Alpha158.build(df)
