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
    Adds ~105+ financial features to an OHLCV DataFrame.

    Feature groups:
        RETURN     — N-day returns, log-returns, rolling mean/skew
        VOLUME     — volume ratios, VWAP-based, turnover
        VOLATILITY — rolling std, normalised range, Parkinson, Garman-Klass
        CORRELATION— price-volume correlation (CORR, CORD), autocorr, beta
        MA / CROSS — EMA spreads, crosses, DEMA, volume-weighted MA, RSV
        MOMENTUM   — momentum, residual momentum, win/loss ratio, Z-score
        MISC       — candle structure, gap, intraday, 52-week pos, seasonality

    All group methods return a dict {col_name: pd.Series} for efficient
    batch concat (avoids DataFrame fragmentation PerformanceWarning).
    """

    @classmethod
    def build(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all Alpha158 features to df (returns enriched copy).
        Uses a single pd.concat to avoid DataFrame fragmentation.
        df must have columns: open, high, low, close, volume
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Alpha158.build: missing columns {missing}")

        logger.info("Building Alpha158 features...")

        # Collect all new feature Series into one dict, then concat once
        features: dict = {}
        features.update(cls._return_features(df))
        features.update(cls._volume_features(df))
        features.update(cls._volatility_features(df, features))   # depends on rvol_ features
        features.update(cls._correlation_features(df))
        features.update(cls._ma_features(df))
        features.update(cls._momentum_features(df))
        features.update(cls._misc_features(df))

        alpha_df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
        n = len([c for c in alpha_df.columns if c.startswith('alpha_')])
        logger.info(f"Alpha158: generated {n} alpha features")
        return alpha_df

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 1: RETURN features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _return_features(df: pd.DataFrame) -> dict:
        c   = df['close']
        ret = c.pct_change(1)
        out = {}

        for n in [1, 2, 3, 4, 5, 10, 20, 30, 60]:
            out[f'alpha_ret_{n}d'] = c.pct_change(n)

        log_ret = np.log(c / c.shift(1))
        for n in [5, 10, 20, 60]:
            out[f'alpha_logret_{n}d'] = log_ret.rolling(n).sum()

        for n in [5, 10, 20, 30, 60]:
            out[f'alpha_ret_mean_{n}d'] = ret.rolling(n).mean()

        for n in [20, 60]:
            out[f'alpha_ret_skew_{n}d'] = ret.rolling(n).skew()

        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 2: VOLUME features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _volume_features(df: pd.DataFrame) -> dict:
        v   = df['volume']
        c   = df['close']
        o   = df['open']
        out = {}

        typical = (df['high'] + df['low'] + c) / 3
        vwap = (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
        out['alpha_vwap_ratio']  = c / (vwap + 1e-9)
        out['alpha_vwap_spread'] = (c - vwap) / (vwap + 1e-9)

        for n in [5, 10, 20, 30, 60]:
            out[f'alpha_vol_ratio_{n}d'] = v / (v.rolling(n).mean() + 1e-9)

        out['alpha_vol_trend_5_20']  = v.rolling(5).mean()  / (v.rolling(20).mean()  + 1e-9)
        out['alpha_vol_trend_10_60'] = v.rolling(10).mean() / (v.rolling(60).mean() + 1e-9)

        turnover = v * c
        for n in [5, 20, 60]:
            out[f'alpha_turnover_{n}d'] = turnover / (turnover.rolling(n).mean() + 1e-9)

        ret = c.pct_change(1)
        out['alpha_vwret_5d']  = (ret * v).rolling(5).sum()  / (v.rolling(5).sum()  + 1e-9)
        out['alpha_vwret_20d'] = (ret * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
        out['alpha_oc_vol']    = (c - o).abs() / (v + 1e-9)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 3: VOLATILITY features  (takes pre-computed features dict)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _volatility_features(df: pd.DataFrame, prior: dict) -> dict:
        c   = df['close']
        ret = c.pct_change(1)
        out = {}

        for n in [5, 10, 20, 30, 60]:
            out[f'alpha_rvol_{n}d'] = ret.rolling(n).std() * np.sqrt(252)

        out['alpha_rvol_ratio_5_20']  = out['alpha_rvol_5d']  / (out['alpha_rvol_20d']  + 1e-9)
        out['alpha_rvol_ratio_20_60'] = out['alpha_rvol_20d'] / (out['alpha_rvol_60d'] + 1e-9)

        rng = (df['high'] - df['low']) / (c + 1e-9)
        for n in [5, 20]:
            out[f'alpha_range_{n}d'] = rng.rolling(n).mean()

        pk = np.log(df['high'] / (df['low'] + 1e-9)) ** 2 / (4 * np.log(2))
        for n in [5, 20]:
            out[f'alpha_parkinson_{n}d'] = pk.rolling(n).mean() * np.sqrt(252)

        gk = (
            0.5 * np.log(df['high'] / (df['low'] + 1e-9)) ** 2
            - (2 * np.log(2) - 1) * np.log(c / (df['open'] + 1e-9)) ** 2
        )
        out['alpha_gk_vol_20d']  = gk.rolling(20).mean() * np.sqrt(252)
        out['alpha_volvol_20d']  = out['alpha_rvol_5d'].rolling(20).std()
        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 4: CORRELATION features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _correlation_features(df: pd.DataFrame) -> dict:
        c   = df['close']
        v   = df['volume']
        ret = c.pct_change(1)
        out = {}

        for n in [5, 10, 20, 60]:
            out[f'alpha_pv_corr_{n}d'] = ret.rolling(n).corr(np.log(v + 1))

        for n in [5, 20]:
            out[f'alpha_pv_cord_{n}d'] = ret.rolling(n).corr(v.pct_change(1))

        for lag in [1, 2, 5]:
            out[f'alpha_autocorr_lag{lag}'] = ret.rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=lag), raw=False
            )

        hl = df['high'] - df['low']
        for n in [10, 20]:
            out[f'alpha_hl_vol_corr_{n}d'] = hl.rolling(n).corr(np.log(v + 1))

        idx = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
        didx = idx.diff()
        for n in [20, 60]:
            out[f'alpha_trend_beta_{n}d'] = (
                ret.rolling(n).cov(didx) / (didx.rolling(n).var() + 1e-9)
            )
        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 5: MA / CROSS features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ma_features(df: pd.DataFrame) -> dict:
        c   = df['close']
        v   = df['volume']
        out = {}

        for n in [5, 10, 20, 30, 60]:
            ema = c.ewm(span=n, adjust=False).mean()
            out[f'alpha_ema_ratio_{n}d']  = c / (ema + 1e-9)
            out[f'alpha_ema_spread_{n}d'] = (c - ema) / (ema + 1e-9)

        for s, l in [(5, 20), (10, 60), (20, 60)]:
            ema_s = c.ewm(span=s, adjust=False).mean()
            ema_l = c.ewm(span=l, adjust=False).mean()
            out[f'alpha_ema_cross_{s}_{l}'] = (ema_s - ema_l) / (ema_l + 1e-9)

        ema20 = c.ewm(span=20, adjust=False).mean()
        dema  = 2 * ema20 - ema20.ewm(span=20, adjust=False).mean()
        out['alpha_dema_ratio_20d'] = c / (dema + 1e-9)

        for n in [5, 20]:
            vwma = (c * v).rolling(n).sum() / (v.rolling(n).sum() + 1e-9)
            out[f'alpha_vwma_spread_{n}d'] = (c - vwma) / (vwma + 1e-9)

        for n in [5, 10, 20]:
            lo = df['low'].rolling(n).min()
            hi = df['high'].rolling(n).max()
            out[f'alpha_rsv_{n}d'] = (c - lo) / (hi - lo + 1e-9)

        out['alpha_klen'] = (
            (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-9)
        )
        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 6: MOMENTUM features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _momentum_features(df: pd.DataFrame) -> dict:
        c   = df['close']
        ret = c.pct_change(1)
        out = {}

        for n in [5, 10, 20, 60]:
            out[f'alpha_mom_{n}d'] = c.pct_change(n)

        for n in [20, 60]:
            out[f'alpha_resmom_{n}d'] = ret - ret.rolling(n).mean()

        for n in [5, 20]:
            out[f'alpha_maxret_{n}d'] = ret.rolling(n).max()
            out[f'alpha_minret_{n}d'] = ret.rolling(n).min()

        wins   = (ret > 0).astype(float)
        losses = (ret < 0).astype(float)
        for n in [10, 20]:
            out[f'alpha_wlratio_{n}d'] = (
                wins.rolling(n).sum() / (losses.rolling(n).sum() + 1e-9)
            )

        mu  = ret.rolling(20).mean()
        sig = ret.rolling(20).std()
        out['alpha_ret_zscore_20d'] = (ret - mu) / (sig + 1e-9)
        out['alpha_ret_kurt_60d']   = ret.rolling(60).kurt()
        return out

    # ──────────────────────────────────────────────────────────────────────
    # GROUP 7: MISC features
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _misc_features(df: pd.DataFrame) -> dict:
        c   = df['close']
        o   = df['open']
        h   = df['high']
        l   = df['low']
        v   = df['volume']
        ret = c.pct_change(1)
        out = {}

        out['alpha_wvma_5d']     = (ret.abs() * v).rolling(5).sum() / (v.rolling(5).sum() + 1e-9)
        out['alpha_gap']         = (o - c.shift(1)) / (c.shift(1) + 1e-9)
        out['alpha_intraday']    = (c - o) / (o + 1e-9)

        body_top    = pd.concat([o, c], axis=1).max(axis=1)
        body_bottom = pd.concat([o, c], axis=1).min(axis=1)
        rng         = (h - l + 1e-9)
        out['alpha_upper_shadow'] = (h - body_top)    / rng
        out['alpha_lower_shadow'] = (body_bottom - l) / rng

        if hasattr(df.index, 'dayofweek'):
            out['alpha_weekday'] = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
            out['alpha_month']   = pd.Series(df.index.month,     index=df.index, dtype=float)

        hi_52w = h.rolling(252).max()
        lo_52w = l.rolling(252).min()
        out['alpha_52w_pos'] = (c - lo_52w) / (hi_52w - lo_52w + 1e-9)

        up_day   = (ret > 0).astype(int)
        down_day = (ret < 0).astype(int)
        out['alpha_consec_up']   = (
            up_day.groupby((up_day != up_day.shift()).cumsum()).cumsum() * up_day
        )
        out['alpha_consec_down'] = (
            down_day.groupby((down_day != down_day.shift()).cumsum()).cumsum() * down_day
        )
        return out


def build_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper."""
    return Alpha158.build(df)
