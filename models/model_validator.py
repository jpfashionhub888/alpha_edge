# models/model_validator.py
"""
Phase 3 — Trading Model Robustness

ModelValidator: enforces professional-grade ML validation standards.

Features:
  1. True temporal out-of-sample split (train 2020-2023, test 2024-2025)
     — data never seen during development is the final judge
  2. Overfitting guard: warn when train_AUC - val_AUC > 0.10, refuse to
     trade when > 0.20 (severe overfit)
  3. Per-regime AUC breakdown (bear / bull / sideways / high-vol)
  4. Slippage tiers (large-cap 3bps, mid-cap 8bps, crypto 20bps)
  5. Monte Carlo position sizing stress test (1,000 permutations)

Usage:
    from models.model_validator import ModelValidator

    validator = ModelValidator()

    # Fit on train split only, evaluate on test split only
    report = validator.full_report(
        df         = feature_df,      # full engineered df with DatetimeIndex
        model      = trained_predictor,
        train_end  = '2023-12-31',    # everything before this = train
        test_start = '2024-01-01',    # everything from here = test (never-seen)
        symbol     = 'AAPL',
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Overfit thresholds
OVERFIT_WARN_GAP  = 0.10   # train_AUC - val_AUC > 10pp → warn
OVERFIT_HARD_STOP = 0.20   # train_AUC - val_AUC > 20pp → block trading

# Default temporal split
DEFAULT_TRAIN_END  = '2023-12-31'
DEFAULT_TEST_START = '2024-01-01'

# Slippage tiers (one-way, in decimal — 3bps = 0.0003)
SLIPPAGE_TIERS = {
    'large_cap' : 0.0003,   # 3bps  — AAPL, MSFT, SPY etc.
    'mid_cap'   : 0.0008,   # 8bps  — mid-size stocks
    'small_cap' : 0.0015,   # 15bps — thinly traded
    'crypto'    : 0.0020,   # 20bps — BTC/ETH/SOL on Gate.io
}

# Symbols that qualify as large-cap (slip = 3bps)
LARGE_CAP_SYMBOLS = {
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'JPM', 'V', 'MA', 'UNH', 'JNJ', 'XOM', 'CVX',
    'SPY', 'QQQ', 'IWM', 'DIA',
    'BTC-USD', 'ETH-USD',   # deep liquidity in crypto
}
MID_CAP_SYMBOLS = {
    'AMD', 'PLTR', 'SOFI', 'HOOD', 'MARA', 'RIVN',
    'DDOG', 'CRWD', 'NET', 'SNOW', 'CRM',
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Complete model validation result."""

    symbol          : str
    train_period    : str
    test_period     : str

    # AUC
    train_auc       : float = 0.5
    test_auc        : float = 0.5
    auc_gap         : float = 0.0   # train - test
    overfit_warning : bool  = False
    overfit_blocked : bool  = False

    # Regime breakdown
    regime_auc      : dict  = field(default_factory=dict)

    # Slippage-adjusted returns
    slippage_tier   : str   = 'large_cap'
    slippage_bps    : float = 3.0
    raw_return      : float = 0.0
    slippage_cost   : float = 0.0
    net_return      : float = 0.0

    # Monte Carlo
    mc_median_return : float = 0.0
    mc_p05_return    : float = 0.0   # 5th percentile (stress case)
    mc_p95_return    : float = 0.0   # 95th percentile (best case)
    mc_prob_ruin     : float = 0.0   # P(drawdown > 20%)

    # Diagnostics
    train_rows      : int   = 0
    test_rows       : int   = 0
    n_signals       : int   = 0
    diagnostics     : dict  = field(default_factory=dict)

    def is_tradeable(self) -> bool:
        """Return True only if model passes all robustness checks."""
        return (
            not self.overfit_blocked
            and self.test_auc > 0.52
            and self.mc_prob_ruin < 0.20
        )

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  Validation Report — {self.symbol}",
            f"{'='*60}",
            f"  Train period : {self.train_period}  ({self.train_rows} rows)",
            f"  Test period  : {self.test_period}   ({self.test_rows} rows)",
            f"",
            f"  Train AUC    : {self.train_auc:.4f}",
            f"  Test AUC     : {self.test_auc:.4f}",
            f"  AUC gap      : {self.auc_gap:+.4f}",
        ]
        if self.overfit_blocked:
            lines.append(f"  [BLOCKED] Severe overfit: gap {self.auc_gap:.2f} > {OVERFIT_HARD_STOP}")
        elif self.overfit_warning:
            lines.append(f"  [WARN] Overfit gap {self.auc_gap:.2f} > {OVERFIT_WARN_GAP} — monitor closely")
        else:
            lines.append(f"  [OK] Overfit gap within tolerance")

        if self.regime_auc:
            lines.append(f"")
            lines.append(f"  Regime AUC breakdown:")
            for regime, auc in sorted(self.regime_auc.items()):
                lines.append(f"    {regime:12s}: {auc:.4f}")

        lines += [
            f"",
            f"  Slippage tier  : {self.slippage_tier} ({self.slippage_bps:.1f}bps)",
            f"  Raw return     : {self.raw_return:+.2%}",
            f"  Slippage cost  : {self.slippage_cost:+.2%}",
            f"  Net return     : {self.net_return:+.2%}",
            f"",
            f"  Monte Carlo (1,000 permutations):",
            f"    Median return  : {self.mc_median_return:+.2%}",
            f"    5th pct return : {self.mc_p05_return:+.2%}",
            f"    95th pct return: {self.mc_p95_return:+.2%}",
            f"    P(ruin >20%dd) : {self.mc_prob_ruin:.1%}",
            f"",
            f"  TRADEABLE: {'YES' if self.is_tradeable() else 'NO'}",
            f"{'='*60}",
        ]
        return '\n'.join(lines)


# ── Main validator ─────────────────────────────────────────────────────────────

class ModelValidator:
    """
    Professional-grade model validation for AlphaEdge.

    All methods are stateless — results are returned in ValidationReport.
    """

    def __init__(self, n_monte_carlo: int = 1_000, random_seed: int = 42):
        self.n_mc   = n_monte_carlo
        self.rng    = np.random.default_rng(random_seed)

    # ── Public: full report ────────────────────────────────────────────────────

    def full_report(
        self,
        df         : pd.DataFrame,
        model,                          # TechnicalPredictor instance (trained)
        train_end  : str = DEFAULT_TRAIN_END,
        test_start : str = DEFAULT_TEST_START,
        symbol     : str = 'UNKNOWN',
        target_col : str = 'target',
        pred_col   : str = 'prediction',
    ) -> ValidationReport:
        """
        Run all validation checks and return a ValidationReport.

        Parameters
        ----------
        df        : feature-engineered DataFrame with DatetimeIndex.
                    Must include `target_col` and `pred_col` columns.
        model     : trained TechnicalPredictor; used to get train-set AUC.
        train_end : last date of training data (inclusive).
        test_start: first date of test data (never seen by model).
        symbol    : ticker for display.
        """
        report = ValidationReport(
            symbol      = symbol,
            train_period= f"start → {train_end}",
            test_period = f"{test_start} → end",
        )

        # ── Split ──────────────────────────────────────────────────────────────
        train_df = df[df.index <= train_end].copy()
        test_df  = df[df.index >= test_start].copy()

        report.train_rows = len(train_df)
        report.test_rows  = len(test_df)

        if len(train_df) < 100 or len(test_df) < 30:
            logger.warning(
                f'[{symbol}] Insufficient rows for validation: '
                f'train={len(train_df)}, test={len(test_df)}'
            )
            report.diagnostics['error'] = 'insufficient_rows'
            return report

        if target_col not in df.columns or pred_col not in df.columns:
            logger.warning(
                f'[{symbol}] Missing {target_col!r} or {pred_col!r} column'
            )
            report.diagnostics['error'] = 'missing_columns'
            return report

        # ── AUC on train split ─────────────────────────────────────────────────
        report.train_auc = self._safe_auc(
            train_df[target_col], train_df[pred_col]
        )

        # ── AUC on test split (never-seen data) ───────────────────────────────
        report.test_auc = self._safe_auc(
            test_df[target_col], test_df[pred_col]
        )

        # ── Overfit check ──────────────────────────────────────────────────────
        report.auc_gap         = report.train_auc - report.test_auc
        report.overfit_warning = report.auc_gap > OVERFIT_WARN_GAP
        report.overfit_blocked = report.auc_gap > OVERFIT_HARD_STOP

        if report.overfit_blocked:
            logger.error(
                f'[{symbol}] TRADING BLOCKED — severe overfit: '
                f'train_AUC={report.train_auc:.3f} '
                f'test_AUC={report.test_auc:.3f} '
                f'gap={report.auc_gap:.3f}'
            )
        elif report.overfit_warning:
            logger.warning(
                f'[{symbol}] Overfit warning: gap={report.auc_gap:.3f}'
            )

        # ── Regime breakdown ───────────────────────────────────────────────────
        report.regime_auc = self._regime_auc(test_df, target_col, pred_col)

        # ── Slippage-adjusted returns ──────────────────────────────────────────
        slip = self._get_slippage(symbol)
        report.slippage_tier = slip['tier']
        report.slippage_bps  = slip['bps']

        if pred_col in test_df.columns:
            signals    = (test_df[pred_col] > 0.63).astype(float)
            report.n_signals = int(signals.sum())

            if 'close' in test_df.columns and len(test_df) > 1:
                daily_ret       = test_df['close'].pct_change().fillna(0)
                strategy_ret    = daily_ret * signals.shift(1).fillna(0)
                report.raw_return    = float((1 + strategy_ret).prod() - 1)
                # Slippage cost = slip per trade * number of trades * 2 (round-trip)
                n_trades             = int(signals.diff().fillna(0).gt(0).sum())
                report.slippage_cost = -(slip['rate'] * 2 * n_trades)
                report.net_return    = report.raw_return + report.slippage_cost

        # ── Monte Carlo stress test ────────────────────────────────────────────
        if pred_col in test_df.columns and 'close' in test_df.columns:
            mc = self._monte_carlo(test_df, pred_col)
            report.mc_median_return  = mc['median']
            report.mc_p05_return     = mc['p05']
            report.mc_p95_return     = mc['p95']
            report.mc_prob_ruin      = mc['prob_ruin']

        logger.info(f'[{symbol}] Validation complete. '
                    f'Train AUC={report.train_auc:.3f} '
                    f'Test AUC={report.test_auc:.3f} '
                    f'Tradeable={report.is_tradeable()}')

        return report

    # ── Public: run on raw OHLCV with fixture data ─────────────────────────────

    def validate_from_fixtures(
        self,
        symbol    : str,
        model,
        feature_engine,
        train_end : str = DEFAULT_TRAIN_END,
        test_start: str = DEFAULT_TEST_START,
    ) -> ValidationReport:
        """
        Load real OHLCV from fixtures, engineer features, run full_report.
        This is the primary interface for Phase 3 validation scripts.
        """
        from pathlib import Path
        import os

        fixtures_dir = Path(__file__).parent.parent / 'tests' / 'fixtures'
        ticker_file  = symbol.replace('-', '_').lower() + '_daily.csv'
        fixture_path = fixtures_dir / ticker_file

        if not fixture_path.exists():
            raise FileNotFoundError(
                f'Fixture not found: {fixture_path}\n'
                f'Run: python scripts/fetch_test_fixtures.py'
            )

        df_raw = pd.read_csv(fixture_path, index_col='date', parse_dates=True)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        df_raw = df_raw.dropna(subset=['close'])

        df = feature_engine.add_all_features(df_raw.copy())

        # Use model predictions if model is trained, else zero
        feature_names = feature_engine.get_feature_names()
        available     = [f for f in feature_names if f in df.columns]

        if model.trained and available:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df['prediction'] = model.predict(df[available])
        else:
            df['prediction'] = 0.5

        return self.full_report(
            df         = df,
            model      = model,
            train_end  = train_end,
            test_start = test_start,
            symbol     = symbol,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_auc(y_true: pd.Series, y_pred: pd.Series) -> float:
        """ROC-AUC with graceful fallback."""
        try:
            if y_true.nunique() < 2:
                return 0.5
            return float(roc_auc_score(y_true.values, y_pred.values))
        except Exception as e:
            logger.debug(f'AUC failed: {e}')
            return 0.5

    def _regime_auc(
        self,
        df        : pd.DataFrame,
        target_col: str,
        pred_col  : str,
    ) -> dict:
        """
        Compute AUC broken down by market regime.

        Regime detection heuristic (from real data patterns):
          - bear     : 50-day return < -10%
          - bull     : 50-day return > +10%
          - high_vol : rolling 20-day return std > 1.5× median
          - sideways : everything else
        """
        if 'close' not in df.columns:
            return {}

        result = {}

        try:
            ret_50d  = df['close'].pct_change(50).fillna(0)
            ret_20d  = df['close'].pct_change(20).fillna(0)
            vol_20d  = ret_20d.rolling(20).std().fillna(0)
            vol_med  = float(vol_20d.median())

            regimes = pd.Series('sideways', index=df.index)
            regimes[ret_50d < -0.10] = 'bear'
            regimes[ret_50d >  0.10] = 'bull'
            regimes[vol_20d > vol_med * 1.5] = 'high_vol'

            for regime in ['bear', 'bull', 'sideways', 'high_vol']:
                mask = regimes == regime
                if mask.sum() < 20:
                    continue
                auc = self._safe_auc(
                    df.loc[mask, target_col],
                    df.loc[mask, pred_col],
                )
                result[regime] = round(auc, 4)

        except Exception as e:
            logger.debug(f'Regime AUC failed: {e}')

        return result

    @staticmethod
    def _get_slippage(symbol: str) -> dict:
        """Return slippage tier and rate for a given symbol."""
        sym = symbol.upper()
        if sym in LARGE_CAP_SYMBOLS:
            return {'tier': 'large_cap', 'bps': 3.0,  'rate': SLIPPAGE_TIERS['large_cap']}
        if sym in MID_CAP_SYMBOLS:
            return {'tier': 'mid_cap',   'bps': 8.0,  'rate': SLIPPAGE_TIERS['mid_cap']}
        # Check if crypto by suffix
        if sym.endswith('-USD') or sym.endswith('USDT'):
            return {'tier': 'crypto',    'bps': 20.0, 'rate': SLIPPAGE_TIERS['crypto']}
        return {'tier': 'small_cap',     'bps': 15.0, 'rate': SLIPPAGE_TIERS['small_cap']}

    def _monte_carlo(
        self,
        df      : pd.DataFrame,
        pred_col: str,
        n_sims  : int | None = None,
        ruin_dd : float      = 0.20,
    ) -> dict:
        """
        Monte Carlo simulation:
        - Take each day's return where model signals (score > 0.63)
        - Permute the ORDER of those trade returns 1,000 times
        - Record cumulative return distribution

        This answers: "Is this strategy robust to the ORDER of trades,
        or did it just get lucky on timing?"
        """
        n_sims = n_sims or self.n_mc
        try:
            signals  = (df[pred_col] > 0.63).astype(float)
            if 'close' not in df.columns or len(df) < 10:
                return self._mc_empty()

            daily_ret   = df['close'].pct_change().fillna(0)
            trade_rets  = daily_ret[signals.shift(1).fillna(0).astype(bool)].values

            if len(trade_rets) < 10:
                logger.debug('Monte Carlo: not enough signal days for meaningful simulation')
                return self._mc_empty()

            sim_totals   = np.zeros(n_sims)
            sim_max_dds  = np.zeros(n_sims)

            for i in range(n_sims):
                perm   = self.rng.permutation(trade_rets)
                cumret = np.cumprod(1 + perm)
                sim_totals[i] = cumret[-1] - 1

                # Max drawdown from peak in this permutation
                peak   = np.maximum.accumulate(cumret)
                dd     = (cumret - peak) / np.where(peak > 0, peak, 1)
                sim_max_dds[i] = float(dd.min())

            prob_ruin = float(np.mean(sim_max_dds < -ruin_dd))

            return {
                'median'   : float(np.median(sim_totals)),
                'p05'      : float(np.percentile(sim_totals, 5)),
                'p95'      : float(np.percentile(sim_totals, 95)),
                'prob_ruin': prob_ruin,
            }

        except Exception as e:
            logger.warning(f'Monte Carlo failed: {e}')
            return self._mc_empty()

    @staticmethod
    def _mc_empty() -> dict:
        return {'median': 0.0, 'p05': 0.0, 'p95': 0.0, 'prob_ruin': 0.0}
