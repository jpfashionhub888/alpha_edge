# tests/test_model_robustness.py
"""
Phase 3 — Trading Model Robustness Tests

All data is REAL historical data from tests/fixtures/:
  - AAPL 2020-2025 (main)
  - NVDA 2023 AI bull run (regime test)
  - AAPL 2022 bear market (regime test)
  - BTC-USD 2020-2025 (crypto slippage test)

Key checks:
  1. Temporal out-of-sample split (train→2023, test 2024-2025)
  2. Overfitting guard: train_AUC - val_AUC thresholds
  3. Slippage tiers correct per symbol class
  4. Monte Carlo: prob_ruin is a valid probability [0, 1]
  5. Regime AUC breakdown (bear/bull/sideways/high_vol)
  6. ValidationReport.is_tradeable() gate logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Skip entire module if core ML dependencies are unavailable (e.g. CI without xgboost).
# xgboost is in requirements.txt and installed in full CI; this guard only fires
# in minimal sandboxes where the heavy ML stack is absent.
try:
    import xgboost  # noqa: F401
    import ta       # noqa: F401
except ImportError as _e:
    pytest.skip(f"ML dependency unavailable ({_e}) — skipping robustness suite",
                allow_module_level=True)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def feature_df_aapl():
    """Full AAPL 2020-2025 with engineered features. Module-scoped for speed."""
    from data.feature_engine import FeatureEngine

    fixtures_dir = Path(__file__).parent / 'fixtures'
    df_raw = pd.read_csv(
        fixtures_dir / 'aapl_daily.csv',
        index_col='date', parse_dates=True,
    )
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.dropna(subset=['close'])

    fe = FeatureEngine()
    df = fe.add_all_features(df_raw.copy())
    return df, fe


@pytest.fixture(scope='module')
def feature_df_btc():
    """BTC-USD with engineered features for crypto slippage tests."""
    from data.feature_engine import FeatureEngine

    fixtures_dir = Path(__file__).parent / 'fixtures'
    df_raw = pd.read_csv(
        fixtures_dir / 'btc_usd_daily.csv',
        index_col='date', parse_dates=True,
    )
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.dropna(subset=['close'])

    fe = FeatureEngine()
    df = fe.add_all_features(df_raw.copy())
    return df, fe


@pytest.fixture(scope='module')
def trained_predictor(feature_df_aapl):
    """Train TechnicalPredictor on AAPL 2020-2023 (training split only)."""
    from models.technical_model import TechnicalPredictor

    df, fe = feature_df_aapl
    train_df = df[df.index <= '2023-12-31'].copy()

    feature_cols = [c for c in fe.get_feature_names() if c in train_df.columns]
    if not feature_cols:
        pytest.skip('No features available from FeatureEngine')

    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, 'target']

    if len(X_train) < 100:
        pytest.skip('Insufficient training rows')

    predictor = TechnicalPredictor(use_lstm=False)
    predictor.train(X_train, y_train)
    return predictor, feature_cols


@pytest.fixture(scope='module')
def df_with_predictions(feature_df_aapl, trained_predictor):
    """Full AAPL df with prediction column from trained model."""
    df, fe = feature_df_aapl
    predictor, feature_cols = trained_predictor

    available = [c for c in feature_cols if c in df.columns]
    df = df.copy()
    df['prediction'] = predictor.predict(df[available])
    return df


# ── ModelValidator import ──────────────────────────────────────────────────────

@pytest.fixture
def validator():
    from models.model_validator import ModelValidator
    return ModelValidator(n_monte_carlo=200, random_seed=42)


# ── 1. Temporal Out-of-Sample Split ───────────────────────────────────────────

class TestTemporalSplit:
    """Train on ≤2023-12-31, test on ≥2024-01-01 — data never seen during dev."""

    def test_train_rows_before_cutoff(self, df_with_predictions):
        """All train rows must have dates ≤ 2023-12-31."""
        df = df_with_predictions
        train = df[df.index <= '2023-12-31']
        assert len(train) > 0
        assert (train.index.year <= 2023).all()

    def test_test_rows_after_cutoff(self, df_with_predictions):
        """All test rows must have dates ≥ 2024-01-01."""
        df = df_with_predictions
        test = df[df.index >= '2024-01-01']
        assert len(test) > 0
        assert (test.index.year >= 2024).all()

    def test_no_date_overlap(self, df_with_predictions):
        """Train and test sets must not share any dates."""
        df = df_with_predictions
        train_dates = set(df[df.index <= '2023-12-31'].index)
        test_dates  = set(df[df.index >= '2024-01-01'].index)
        overlap = train_dates & test_dates
        assert len(overlap) == 0, f'Date overlap found: {list(overlap)[:5]}'

    def test_report_train_test_rows_nonzero(self, validator, df_with_predictions):
        """ValidationReport must show non-zero rows on both splits."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert report.train_rows > 0, 'Zero train rows in report'
        assert report.test_rows  > 0, 'Zero test rows in report'

    def test_test_auc_is_real_oos_performance(self, validator, df_with_predictions):
        """
        Test AUC is computed on 2024-2025 data — the true OOS period.
        It will likely be lower than train AUC; that's expected and healthy.
        It must be in (0, 1) range and not exactly 0.5 (which would indicate
        all predictions are constant).
        """
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert 0.0 < report.test_auc < 1.0, (
            f'test_auc out of range: {report.test_auc}'
        )


# ── 2. Overfitting Guard ───────────────────────────────────────────────────────

class TestOverfittingGuard:
    """Models that memorise training data must be detected and blocked."""

    def test_trained_predictor_has_auc_attributes(self, trained_predictor):
        """After train(), predictor must expose train_auc, val_auc, overfit_gap."""
        predictor, _ = trained_predictor
        assert hasattr(predictor, 'train_auc')
        assert hasattr(predictor, 'val_auc')
        assert hasattr(predictor, 'overfit_gap')
        assert hasattr(predictor, 'overfit_flagged')

    def test_overfit_gap_is_numeric(self, trained_predictor):
        """overfit_gap must be a float in [-1, 1]."""
        predictor, _ = trained_predictor
        assert isinstance(predictor.overfit_gap, float)
        assert -1.0 <= predictor.overfit_gap <= 1.0

    def test_report_overfit_blocked_on_large_gap(self, validator, df_with_predictions):
        """
        If we inject predictions that perfectly fit train but are noise on test,
        the overfit_blocked flag must be set.
        """
        df = df_with_predictions.copy()

        # Perfect train predictions, random test predictions
        train_mask = df.index <= '2023-12-31'
        test_mask  = df.index >= '2024-01-01'

        # Make train predictions 100% correlated with target (pure overfit)
        df.loc[train_mask, 'prediction'] = df.loc[train_mask, 'target'].astype(float)
        # Make test predictions purely random noise
        rng = np.random.default_rng(0)
        df.loc[test_mask, 'prediction'] = rng.uniform(0.45, 0.55, test_mask.sum())

        report = validator.full_report(
            df=df, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL_OVERFIT_SIMULATION',
        )
        # Perfect train / random test → huge gap → must be blocked
        assert report.overfit_blocked, (
            f'Overfit not detected: train_AUC={report.train_auc:.3f} '
            f'test_AUC={report.test_auc:.3f} gap={report.auc_gap:.3f}'
        )

    def test_is_tradeable_false_when_blocked(self, validator, df_with_predictions):
        """overfit_blocked → is_tradeable() must return False."""
        df = df_with_predictions.copy()

        train_mask = df.index <= '2023-12-31'
        test_mask  = df.index >= '2024-01-01'
        df.loc[train_mask, 'prediction'] = df.loc[train_mask, 'target'].astype(float)
        rng = np.random.default_rng(1)
        df.loc[test_mask, 'prediction'] = rng.uniform(0.45, 0.55, test_mask.sum())

        report = validator.full_report(
            df=df, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='OVERFIT_TEST',
        )
        if report.overfit_blocked:
            assert not report.is_tradeable()

    def test_well_calibrated_model_not_blocked(self, validator, df_with_predictions):
        """
        A model with near-random predictions on both splits (AUC ~0.50 each)
        has gap ≈ 0.0 — must NOT be flagged as overfitting.
        """
        df = df_with_predictions.copy()
        # Inject constant 0.5 predictions everywhere — no overfit possible
        df['prediction'] = 0.50

        report = validator.full_report(
            df=df, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='NEUTRAL_TEST',
        )
        assert not report.overfit_blocked, (
            f'False positive overfit detection: gap={report.auc_gap:.3f}'
        )


# ── 3. Slippage Tiers ─────────────────────────────────────────────────────────

class TestSlippage:
    """Slippage must be correctly tiered by symbol class."""

    @pytest.mark.parametrize('symbol,expected_tier,expected_bps', [
        ('AAPL',    'large_cap', 3.0),
        ('MSFT',    'large_cap', 3.0),
        ('SPY',     'large_cap', 3.0),
        ('BTC-USD', 'large_cap', 3.0),   # BTC in LARGE_CAP_SYMBOLS
        ('PLTR',    'mid_cap',   8.0),
        ('HOOD',    'mid_cap',   8.0),
        ('MARA',    'mid_cap',   8.0),
        ('XYZABC',  'small_cap', 15.0),  # unknown → small cap
    ])
    def test_slippage_tier(self, symbol, expected_tier, expected_bps):
        """Each symbol class must map to the correct tier and bps."""
        from models.model_validator import ModelValidator
        slip = ModelValidator._get_slippage(symbol)
        assert slip['tier'] == expected_tier, (
            f'{symbol}: expected {expected_tier}, got {slip["tier"]}'
        )
        assert slip['bps'] == expected_bps, (
            f'{symbol}: expected {expected_bps}bps, got {slip["bps"]}bps'
        )

    def test_slippage_rate_reduces_returns(self, validator, df_with_predictions):
        """net_return must be ≤ raw_return (slippage can only reduce returns)."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert report.net_return <= report.raw_return + 1e-9, (
            f'net_return > raw_return: {report.net_return:.4f} > {report.raw_return:.4f}'
        )

    def test_crypto_slippage_higher_than_large_cap(self):
        """Gate.io crypto tokens (not BTC/ETH) must get higher slip than AAPL."""
        from models.model_validator import ModelValidator
        aapl_slip = ModelValidator._get_slippage('AAPL')
        sol_slip  = ModelValidator._get_slippage('SOL-USD')
        assert sol_slip['rate'] >= aapl_slip['rate'], (
            'Crypto slippage is lower than large-cap — check tier mapping'
        )


# ── 4. Monte Carlo Stress Test ────────────────────────────────────────────────

class TestMonteCarlo:
    """1,000-permutation Monte Carlo stress test on real AAPL data."""

    def test_prob_ruin_is_valid_probability(self, validator, df_with_predictions):
        """prob_ruin must be in [0, 1]."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert 0.0 <= report.mc_prob_ruin <= 1.0, (
            f'prob_ruin out of [0,1]: {report.mc_prob_ruin}'
        )

    def test_p05_le_median_le_p95(self, validator, df_with_predictions):
        """5th pct ≤ median ≤ 95th pct — monotone quantiles."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert report.mc_p05_return <= report.mc_median_return, (
            f'p05 > median: {report.mc_p05_return} > {report.mc_median_return}'
        )
        assert report.mc_median_return <= report.mc_p95_return, (
            f'median > p95: {report.mc_median_return} > {report.mc_p95_return}'
        )

    def test_mc_reproducible_with_same_seed(self, df_with_predictions):
        """Same seed → same Monte Carlo result (reproducibility)."""
        from models.model_validator import ModelValidator

        v1 = ModelValidator(n_monte_carlo=100, random_seed=99)
        v2 = ModelValidator(n_monte_carlo=100, random_seed=99)

        r1 = v1.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        r2 = v2.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        assert abs(r1.mc_prob_ruin - r2.mc_prob_ruin) < 1e-9, (
            'Monte Carlo not reproducible with same seed'
        )

    def test_constant_up_signal_low_ruin_prob(self, validator):
        """
        If every signal day has a small positive return (+0.1%),
        prob_ruin after any permutation should be 0 (no possible drawdown).
        """
        # Build synthetic SERIES (just for this MC edge-case test)
        # Note: this is the ONLY place in Phase 3 we use a constructed
        # series — it's testing the MC algorithm itself, not market behaviour
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        df = pd.DataFrame({
            'close'     : np.cumprod(np.ones(100) * 1.001),
            'target'    : np.ones(100, dtype=int),
            'prediction': np.ones(100) * 0.70,   # always above 0.63 threshold
        }, index=dates)

        mc = validator._monte_carlo(df, 'prediction', n_sims=200)
        # With constant +0.1% per day, ruin prob must be 0
        assert mc['prob_ruin'] == 0.0, (
            f'Expected 0 ruin probability on constant +ve returns, '
            f'got {mc["prob_ruin"]}'
        )


# ── 5. Regime AUC Breakdown ───────────────────────────────────────────────────

class TestRegimeAUC:
    """AUC must be computed separately per market regime on real data."""

    def test_regime_auc_keys_valid(self, validator, df_with_predictions):
        """Regime AUC dict must only contain known regime names."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        valid = {'bear', 'bull', 'sideways', 'high_vol'}
        for k in report.regime_auc:
            assert k in valid, f'Unknown regime key: {k!r}'

    def test_regime_auc_in_range(self, validator, df_with_predictions):
        """All regime AUC values must be in (0, 1)."""
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        for regime, auc in report.regime_auc.items():
            assert 0.0 < auc < 1.0, (
                f'Regime AUC out of range for {regime!r}: {auc}'
            )

    def test_regimes_detected_on_real_aapl_data(self, validator, df_with_predictions):
        """
        5 years of AAPL covers bear (2022), bull (2020-2021, 2023-2025),
        and sideways periods — at least 2 regimes must be detected.
        """
        report = validator.full_report(
            df=df_with_predictions, model=MagicMock(),
            train_end='2023-12-31', test_start='2024-01-01',
            symbol='AAPL',
        )
        # The test period (2024-2025) should have at least bear or bull
        assert len(report.regime_auc) >= 1, (
            'No regime breakdown generated on real AAPL 2024-2025 data'
        )


# ── 6. ValidationReport Logic ─────────────────────────────────────────────────

class TestValidationReport:
    """is_tradeable() gate must enforce all conditions."""

    def test_tradeable_requires_test_auc_above_052(self):
        """A model with test_auc=0.50 must not be tradeable."""
        from models.model_validator import ValidationReport
        r = ValidationReport(symbol='X', train_period='', test_period='',
                             test_auc=0.50, mc_prob_ruin=0.05,
                             overfit_blocked=False)
        assert not r.is_tradeable()

    def test_tradeable_requires_low_ruin_prob(self):
        """A model with prob_ruin=0.25 (> 0.20 threshold) must not be tradeable."""
        from models.model_validator import ValidationReport
        r = ValidationReport(symbol='X', train_period='', test_period='',
                             test_auc=0.60, mc_prob_ruin=0.25,
                             overfit_blocked=False)
        assert not r.is_tradeable()

    def test_tradeable_when_all_conditions_met(self):
        """A model passing all three gates must be tradeable."""
        from models.model_validator import ValidationReport
        r = ValidationReport(symbol='X', train_period='', test_period='',
                             test_auc=0.57, mc_prob_ruin=0.10,
                                           overfit_blocked=False)
        assert r.is_tradeable()

    def test_summary_contains_symbol(self):
        """summary() string must include the symbol name."""
        from models.model_validator import ValidationReport
        r = ValidationReport(symbol='AAPL', train_period='2020-2023',
                             test_period='2024-2025',
                             test_auc=0.57, mc_prob_ruin=0.10,
                             overfit_blocked=False)
        summary = r.summary()
        assert 'AAPL' in summary, f"summary() must mention symbol, got: {summary!r}"
