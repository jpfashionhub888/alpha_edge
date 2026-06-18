# tests/test_feature_engine.py
"""
Unit tests for data/feature_engine.py

Critical: No look-ahead bias allowed.
Every feature must be computable using only data available at that point in time.

Tests:
- future_return column excluded from feature set
- Pre-set target column not overwritten
- Rolling windows don't bleed future data
- Features are stable when data is truncated to any boundary date
"""

import numpy as np
import pandas as pd
import pytest


class TestNoLookahead:
    """Features must not use any data beyond the current bar."""

    def test_future_return_excluded_from_features(self, ohlcv_df):
        """future_return is a training label — must never appear as a feature."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        assert 'future_return' not in result.columns
        assert 'future_return' not in fe.get_feature_names()

    def test_target_not_in_feature_names(self, ohlcv_df):
        """target column excluded from feature_names list."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        assert 'target' not in fe.get_feature_names()

    def test_feature_value_stable_at_date_boundary(self, ohlcv_df):
        """
        Feature computed on full data vs truncated data must be identical
        at the truncation point. Any difference = look-ahead leak.
        """
        from data.feature_engine import FeatureEngine

        cutoff_idx = 200  # use first 200 bars as "known" data

        fe1 = FeatureEngine()
        full_result = fe1.add_all_features(ohlcv_df.copy())

        fe2 = FeatureEngine()
        trunc_result = fe2.add_all_features(ohlcv_df.iloc[:cutoff_idx].copy())

        # Find the last common date
        common_dates = full_result.index.intersection(trunc_result.index)
        if len(common_dates) == 0:
            pytest.skip('No common dates after dropna — increase ohlcv_df size')

        last_common = common_dates[-1]
        shared_cols = [c for c in full_result.columns
                       if c in trunc_result.columns
                       and c not in ('target', 'future_return')]

        for col in shared_cols[:5]:  # spot-check 5 features
            v_full  = full_result.loc[last_common, col]
            v_trunc = trunc_result.loc[last_common, col]
            assert abs(float(v_full) - float(v_trunc)) < 1e-6, (
                f"Look-ahead detected in '{col}': "
                f"full={v_full:.6f} vs trunc={v_trunc:.6f}"
            )


class TestTargetVariable:
    """Target variable generation."""

    def test_pre_set_target_not_overwritten(self, ohlcv_df):
        """If target column already exists, FeatureEngine must not overwrite it."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()

        df = ohlcv_df.copy()
        # Pre-set a custom target (all 1s as a canary)
        df['target'] = 1

        result = fe.add_all_features(df)

        # All surviving rows should have target=1 (our canary)
        assert (result['target'] == 1).all(), (
            'FeatureEngine overwrote pre-set target column!'
        )

    def test_target_is_binary(self, ohlcv_df):
        """Auto-generated target must be 0 or 1 — no other values."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        unique = result['target'].unique()
        assert set(unique).issubset({0, 1}), (
            f'Target has non-binary values: {unique}'
        )

    def test_target_not_all_same_class(self, ohlcv_df):
        """Target must have both 0 and 1 — degenerate labels = broken training."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        assert result['target'].nunique() == 2, (
            'Target is degenerate (all same class) on synthetic data'
        )


class TestFeatureCompleteness:
    """Feature set must be stable and complete."""

    def test_feature_count_reasonable(self, ohlcv_df):
        """Expect at least 20 features generated from standard OHLCV."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        assert len(fe.get_feature_names()) >= 20, (
            f'Only {len(fe.get_feature_names())} features generated — '
            'check _add_* methods'
        )

    def test_no_nan_in_features_after_dropna(self, ohlcv_df):
        """After add_all_features(), no NaN should remain in feature columns."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        result = fe.add_all_features(ohlcv_df.copy())

        feature_cols = fe.get_feature_names()
        nan_counts = result[feature_cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        assert len(bad) == 0, f'NaN in features after dropna: {bad.to_dict()}'

    def test_ohlcv_columns_not_in_features(self, ohlcv_df):
        """Raw OHLCV columns must not appear as model features."""
        from data.feature_engine import FeatureEngine
        fe = FeatureEngine()
        fe.add_all_features(ohlcv_df.copy())

        raw = {'open', 'high', 'low', 'close', 'volume'}
        overlap = raw & set(fe.get_feature_names())
        assert not overlap, f'Raw OHLCV columns in feature set: {overlap}'
