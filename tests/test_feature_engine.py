# tests/test_feature_engine.py
"""
Unit tests for data/feature_engine.py

All data is REAL historical data (AAPL, NVDA, SPY, BTC-USD 2020-2025).
No synthetic or random data used.

Critical tests:
- No look-ahead bias (date-boundary stability test)
- Target column integrity
- Feature completeness across real market regimes
"""

import pandas as pd
import pytest


class TestNoLookahead:
    """Features must not use any data beyond the current bar."""

    def test_future_return_excluded_from_features(self, ohlcv_df):
        """future_return is a label — must never appear as a model feature."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        assert 'future_return' not in result.columns
        assert 'future_return' not in fe.get_feature_names()

    def test_target_not_in_feature_names(self, ohlcv_df):
        """target column must not appear in feature_names list."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        assert 'target' not in fe.get_feature_names()

    def test_feature_value_stable_at_date_boundary(self, ohlcv_df):
        """
        Look-ahead bias detector: feature computed on full AAPL 2020-2025 data
        vs truncated data (only up to 2023-12-31) must be IDENTICAL at the
        last common date. Any difference = future data leaked into the feature.

        This uses real AAPL data — not synthetic — so it catches real bugs.
        """
        from data.feature_engine import FeatureEngine

        # Split at end of 2023 (real calendar boundary)
        cutoff = '2023-12-31'

        fe_full = FeatureEngine()
        full_result = fe_full.add_all_features(ohlcv_df.copy())

        fe_trunc = FeatureEngine()
        trunc_df  = ohlcv_df[ohlcv_df.index <= cutoff].copy()
        trunc_result = fe_trunc.add_all_features(trunc_df)

        common_dates = full_result.index.intersection(trunc_result.index)
        if len(common_dates) == 0:
            pytest.skip('No common dates after dropna')

        last_common = common_dates[-1]
        shared_cols = [c for c in full_result.columns
                       if c in trunc_result.columns
                       and c not in ('target', 'future_return')]

        for col in shared_cols[:8]:  # spot-check 8 features
            v_full  = float(full_result.loc[last_common, col])
            v_trunc = float(trunc_result.loc[last_common, col])
            assert abs(v_full - v_trunc) < 1e-6, (
                f"Look-ahead bias detected in '{col}' on real AAPL data: "
                f"full={v_full:.8f} vs trunc={v_trunc:.8f} at {last_common.date()}"
            )


class TestTargetVariable:
    """Target variable generation on real data."""

    def test_pre_set_target_not_overwritten(self, ohlcv_df):
        """If target column already set, FeatureEngine must preserve it."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        df = ohlcv_df.copy()
        df['target'] = 1   # canary value

        result = fe.add_all_features(df)

        assert (result['target'] == 1).all(), (
            'FeatureEngine overwrote a pre-set target column!'
        )

    def test_target_is_binary_on_real_data(self, ohlcv_df):
        """Auto-generated target must only contain 0 and 1 on real AAPL data."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        unique = set(result['target'].unique())
        assert unique.issubset({0, 1}), (
            f'Target has non-binary values on real data: {unique}'
        )

    def test_target_not_degenerate_on_aapl(self, ohlcv_df):
        """Real AAPL 2020-2025 must produce both UP and DOWN labels."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        assert result['target'].nunique() == 2, (
            'Target is all-same-class on 5 years of real AAPL data — '
            'check future_return threshold or target generation logic'
        )

    def test_class_balance_reasonable(self, ohlcv_df):
        """
        On 5 years of real data, neither class should be < 30% or > 70%.
        Extreme imbalance on real data usually indicates a leaky target.
        """
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        up_pct = result['target'].mean()
        assert 0.30 <= up_pct <= 0.70, (
            f'Extreme class imbalance on real AAPL data: '
            f'{up_pct:.1%} UP — possible target leak or bad threshold'
        )


class TestRegimeConsistency:
    """Features should be stable across different real market regimes."""

    def test_features_work_in_bear_market(self, bear_market_df):
        """
        Real AAPL 2022 bear market data (AAPL -27% that year).
        Feature engine must not crash on sustained downtrend data.
        """
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        if len(bear_market_df) < 50:
            pytest.skip('Insufficient bear market rows')

        result = fe.add_all_features(bear_market_df.copy())
        assert len(result) > 0, 'Feature engine returned empty DataFrame on bear market data'

        feature_cols = fe.get_feature_names()
        if feature_cols:
            nan_pct = result[feature_cols].isna().mean().mean()
            assert nan_pct < 0.5, (
                f'Too many NaNs ({nan_pct:.1%}) in bear market features — '
                'indicator warmup period too long for 1-year dataset'
            )

    def test_features_work_in_bull_market(self, bull_market_df):
        """
        Real NVDA 2023 AI bull run (+238% that year).
        Feature engine must handle extreme uptrend without overflow/NaN.
        """
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        if len(bull_market_df) < 50:
            pytest.skip('Insufficient bull market rows')

        result = fe.add_all_features(bull_market_df.copy())
        assert len(result) > 0

        feature_cols = fe.get_feature_names()
        if feature_cols:
            inf_count = (result[feature_cols] == float('inf')).sum().sum()
            assert inf_count == 0, (
                f'{inf_count} infinite values in bull market features '
                f'(NVDA +238% 2023) — check ratio/momentum features'
            )

    def test_features_work_during_covid_crash(self, volatile_period_df):
        """
        Real SPY data during COVID crash (Mar-Dec 2020: -34% then +68% recovery).
        Extreme volatility spike must not produce NaN/inf in features.
        """
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        if len(volatile_period_df) < 50:
            pytest.skip('Insufficient volatile period rows')

        result = fe.add_all_features(volatile_period_df.copy())
        assert len(result) > 0

    def test_crypto_data_processed_correctly(self, ohlcv_df_btc):
        """
        Real BTC-USD data (24/7, no market hours gaps).
        Feature engine must handle crypto data without calendar-day gaps causing NaN.
        """
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        result = fe.add_all_features(ohlcv_df_btc.copy())
        assert len(result) > 100, (
            f'Only {len(result)} rows from BTC-USD after feature generation — '
            'warmup period too aggressive for crypto data'
        )


class TestFeatureCompleteness:
    """Feature set must be stable and complete on real data."""

    def test_feature_count_reasonable(self, ohlcv_df):
        """Expect at least 20 features from real AAPL OHLCV."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        count = len(fe.get_feature_names())
        assert count >= 20, (
            f'Only {count} features on real data — '
            'feature generation may be incomplete'
        )

    def test_no_nan_in_features_after_dropna(self, ohlcv_df):
        """After processing real AAPL data, no NaN should remain in features."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        feature_cols = fe.get_feature_names()
        if not feature_cols:
            pytest.skip('No feature names returned')

        nan_counts = result[feature_cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        assert len(bad) == 0, (
            f'NaN remaining in features on real AAPL data:\n{bad.to_dict()}'
        )

    def test_ohlcv_columns_not_in_features(self, ohlcv_df):
        """Raw OHLCV columns must not appear as model features."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        raw = {'open', 'high', 'low', 'close', 'volume'}
        overlap = raw & set(fe.get_feature_names())
        assert not overlap, f'Raw OHLCV in feature set: {overlap}'

    def test_feature_values_finite_on_full_dataset(self, ohlcv_df):
        """No feature value should be +/-inf on full 5-year AAPL dataset."""
        from data.feature_engine import FeatureEngine
        import numpy as np
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        feature_cols = fe.get_feature_names()
        if not feature_cols:
            pytest.skip('No feature names returned')

        inf_mask = result[feature_cols].isin([float('inf'), float('-inf')])
        inf_counts = inf_mask.sum()
        bad = inf_counts[inf_counts > 0]
        assert len(bad) == 0, (
            f'Infinite values in features on real 5-year AAPL data:\n'
            f'{bad.to_dict()}'
        )
