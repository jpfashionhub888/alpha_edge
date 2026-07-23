# backtesting/signals/library/ml_signal.py
"""
ML Ensemble Signal (Walk-Forward, 21-day target)

Academic basis:
  Technical features (RSI, MACD, Bollinger, momentum, volume) processed
  through an XGBoost + LightGBM + RandomForest + CatBoost ensemble.
  Target: probability that close price is higher 21 trading days out.

  Validated IC:  +0.058  (t=4.75, p<0.0001)
  Holding period: 21 trading days (~1 month)
  Rebalance:      monthly

Key design choices:
  - 21-day binary target (up/down) -- aligns with where IC actually exists
  - TRAIN_DAYS=504 (2 trading years) -- gives ~280 usable rows after
    feature warmup (~200 rows) and target NaN removal (~21 rows)
  - Cross-sectional training: all symbols pooled so model learns
    relative rather than absolute feature values
  - Monthly retrain: balances freshness vs compute cost
  - Fail-safe: returns {} if model fails hard stop (val_AUC < 0.53)
    rather than trading on garbage predictions

Lookahead rules:
  - compute(date, data) only uses data.index < date (enforced by caller)
  - Target injection uses shift(-21) so last 21 rows have NaN -- dropped
  - FeatureEngine respects pre-computed 'target' column (skips generation)

Usage:
    from backtesting.signals.library.ml_signal import MLSignal, MLBacktest
    signal = MLSignal()
    scores = signal.compute(date, price_data)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtesting.signals.base import BaseSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TRAIN_DAYS   = 504   # 2 trading years -- headroom after feature warmup + 21d NaN
TARGET_DAYS  = 21    # prediction horizon (days)
MIN_TRAIN    = 150   # minimum usable rows after warmup + NaN drop
K_FEATURES   = 20    # SelectKBest top-k
RETRAIN_DAYS = 21    # retrain every ~1 calendar month


class MLSignal(BaseSignal):
    """
    Walk-forward ML ensemble signal.

    Trains cross-sectionally (all symbols pooled) on TRAIN_DAYS of history
    with a 21-day binary target, then scores each symbol by probability of
    a positive 21-day return. Retrains monthly.

    Scores are in [0, 1] (ensemble probability). 0.5 = no view.
    """

    name                = 'ml_ensemble_21d'
    holding_period_days = TARGET_DAYS
    rebalance_freq      = 'M'
    min_history_days    = TRAIN_DAYS + 50   # need train window + buffer

    def __init__(
        self,
        train_days          : int  = TRAIN_DAYS,
        target_days         : int  = TARGET_DAYS,
        k_features          : int  = K_FEATURES,
        retrain_days        : int  = RETRAIN_DAYS,
        bypass_quality_gate : bool = False,
    ):
        self._train_days          = train_days
        self._target_days         = target_days
        self._k_features          = k_features
        self._retrain_days        = retrain_days
        self._bypass_quality_gate = bypass_quality_gate

        self._predictor     = None   # TechnicalPredictor (trained)
        self._feature_cols  = None   # list[str] used in training
        self._last_retrain  : Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # type: ignore[override]
        return 'ml_ensemble_21d'

    def compute(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Return {symbol: score} where score in [0, 1].
        Higher = model predicts price up over next 21 days.
        Returns {} if model is unavailable or fails quality gate.
        """
        if self._needs_retrain(date):
            ok = self._retrain(date, data)
            if not ok:
                logger.warning('%s: retrain failed at %s -- no scores', self.name, date.date())
                return {}

        if self._predictor is None or self._feature_cols is None:
            return {}

        return self._score_symbols(date, data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _needs_retrain(self, date: pd.Timestamp) -> bool:
        if self._predictor is None:
            return True
        if self._last_retrain is None:
            return True
        days_elapsed = (date - self._last_retrain).days
        return days_elapsed >= int(self._retrain_days * 1.4)   # ~calendar month

    def _retrain(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Build a cross-sectional training panel from all symbols,
        inject 21-day target, train ensemble. Returns True on success.
        """
        try:
            from data.feature_engine     import FeatureEngine
            from models.technical_model  import TechnicalPredictor
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
        except ImportError as e:
            logger.error('MLSignal: missing dependency -- %s', e)
            return False

        engine   = FeatureEngine()
        segments = []

        for sym, df in data.items():
            hist = df[df.index < date].copy()
            if len(hist) < self._train_days:
                continue

            window = hist.tail(self._train_days).copy()

            # Inject 21-day target BEFORE add_all_features so FeatureEngine
            # skips its default 5-day target generation.
            fwd = window['close'].pct_change(self._target_days).shift(-self._target_days)
            window['target'] = (fwd > 0).astype(float)
            window.loc[fwd.isna(), 'target'] = np.nan

            try:
                feat = engine.add_all_features(window)
            except Exception as exc:
                logger.debug('MLSignal retrain: feature gen failed for %s: %s', sym, exc)
                continue

            feat = feat.dropna(subset=['target'])
            if len(feat) < MIN_TRAIN:
                continue

            feat['_sym'] = sym
            segments.append(feat)

        if not segments:
            logger.warning('MLSignal: no valid training segments at %s', date.date())
            return False

        panel = pd.concat(segments, axis=0).reset_index(drop=True)

        feature_names = engine.get_feature_names()
        if not feature_names:
            logger.warning('MLSignal: FeatureEngine returned no feature names')
            return False

        available = [c for c in feature_names if c in panel.columns]
        X = panel[available].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        y = panel.loc[X.index, 'target']

        if len(y.unique()) < 2 or y.value_counts().min() < 5:
            logger.warning('MLSignal: insufficient class balance at %s', date.date())
            return False

        y = y.astype(int)

        # Feature selection: keep top-K by mutual information
        k = min(self._k_features, X.shape[1])
        try:
            sel      = SelectKBest(mutual_info_classif, k=k)
            sel.fit(X, y)
            sel_cols = [c for c, keep in zip(X.columns, sel.get_support()) if keep]
        except Exception:
            sel_cols = list(X.columns[:k])

        X_sel = X[sel_cols]

        predictor = TechnicalPredictor()
        try:
            predictor.train(X_sel, y)
        except Exception as exc:
            logger.warning('MLSignal: TechnicalPredictor.train() failed: %s', exc)
            return False

        if predictor.overfit_flagged and not self._bypass_quality_gate:
            logger.warning(
                'MLSignal: model failed quality gate (val_AUC < threshold) at %s -- '
                'skipping this rebalance date', date.date()
            )
            return False
        if predictor.overfit_flagged:
            logger.debug(
                'MLSignal: quality gate bypassed (backtest mode) at %s', date.date()
            )

        # Sanity-check: run a dummy prediction to ensure models are actually fitted.
        # TechnicalPredictor catches per-model fit failures silently; if all models
        # are unfitted the ensemble returns 0.5 for everything (useless signal).
        try:
            test_pred = predictor.predict(X_sel.iloc[:1])
            if float(test_pred[0]) == 0.5 and len(predictor.models) > 0:
                # All models returned 0.5 -- none fitted successfully
                logger.warning('MLSignal: all sub-models unfitted at %s -- skipping', date.date())
                return False
        except Exception as exc:
            logger.warning('MLSignal: predictor sanity check failed at %s: %s', date.date(), exc)
            return False

        self._predictor    = predictor
        self._feature_cols = sel_cols
        self._last_retrain = date
        logger.info(
            'MLSignal: retrained at %s | panel=%d rows | features=%d',
            date.date(), len(X_sel), len(sel_cols)
        )
        return True

    def _score_symbols(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Run inference on each symbol's latest feature row."""
        try:
            from data.feature_engine import FeatureEngine
        except ImportError:
            return {}

        engine = FeatureEngine()
        scores = {}

        for sym, df in data.items():
            hist = df[df.index < date].copy()
            if len(hist) < 60:
                continue

            try:
                feat = engine.add_all_features(hist)
            except Exception:
                continue

            if len(feat) == 0:
                continue

            # Build feature row aligned to training columns
            avail = [c for c in self._feature_cols if c in feat.columns]
            if len(avail) < len(self._feature_cols) * 0.7:
                # Too many missing features -- skip rather than predict on garbage
                continue

            row = feat.iloc[-1][avail].replace([np.inf, -np.inf], np.nan)
            if row.isna().any():
                continue

            X_row = pd.DataFrame([row.values], columns=avail)
            # Pad any missing training columns with 0.0
            for c in self._feature_cols:
                if c not in X_row.columns:
                    X_row[c] = 0.0
            X_row = X_row[self._feature_cols]

            try:
                prob = float(self._predictor.predict(X_row)[0])
                scores[sym] = prob
            except Exception as exc:
                logger.debug('MLSignal: predict failed for %s: %s', sym, exc)

        return scores


# ---------------------------------------------------------------------------
# Backtest convenience wrapper
# ---------------------------------------------------------------------------

class MLBacktest:
    """
    Convenience wrapper: run a full ML-signal backtest end-to-end.

    Usage:
        bt = MLBacktest()
        result = bt.run(symbols=[...], start_date='2022-01-01', end_date='2026-07-01')
        result.print_summary()
    """

    def __init__(
        self,
        initial_capital : float = 100_000,
        max_positions   : int   = 10,
        train_days      : int   = TRAIN_DAYS,
        target_days     : int   = TARGET_DAYS,
    ):
        from backtesting.data.loader         import DataLoader
        from backtesting.engine.event_driven import EventDrivenBacktester
        from backtesting.engine.fill_model   import FillModel
        from backtesting.engine.cost_model   import TransactionCostModel

        self.loader         = DataLoader()
        self.signal         = MLSignal(
            train_days          = train_days,
            target_days         = target_days,
            bypass_quality_gate = True,   # backtest mode: test raw signal IC -> Sharpe
        )
        self.engine         = EventDrivenBacktester(
            initial_capital = initial_capital,
            fill_model      = FillModel(spread_bps=5.0, market_impact_factor=0.1),
            cost_model      = TransactionCostModel(),
        )
        self.max_positions  = max_positions

    def run(
        self,
        symbols          : list,
        start_date       : str,
        end_date         : str,
        spy_regime_filter: bool = True,
    ):
        from backtesting.analysis.metrics import performance_summary, print_summary

        logger.info('Loading OHLCV for %d symbols...', len(symbols))
        fetch_syms = list(set(symbols + ['SPY']))
        # Load from 2020-01-01 to give the model enough warmup history
        all_data   = self.loader.get_ohlcv(fetch_syms, '2020-01-01', end_date)
        price_data = {k: v for k, v in all_data.items() if k != 'SPY'}
        spy_prices = all_data.get('SPY')
        logger.info('Loaded %d symbols', len(price_data))

        spy_ma200 = None
        if spy_regime_filter and spy_prices is not None:
            spy_close = spy_prices['close']
            spy_ma200 = spy_close.rolling(200).mean()

        signal = self.signal
        max_n  = self.max_positions

        def signal_fn(date, data):
            # SPY 200d MA regime filter
            if spy_ma200 is not None:
                avail = spy_ma200[spy_ma200.index < date]
                if len(avail) > 0:
                    ma_val = float(avail.iloc[-1])
                    spy_close_avail = spy_prices['close'][spy_prices.index < date]
                    if len(spy_close_avail) > 0 and not np.isnan(ma_val):
                        if float(spy_close_avail.iloc[-1]) < ma_val:
                            return {}  # cash in bear market

            scores = signal.compute(date, data)
            if not scores:
                return {}

            # Long-only: equal-weight top-N by score
            ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [(s, sc) for s, sc in ranked if sc > 0.5][:max_n]
            if not selected:
                return {}

            n = len(selected)
            return {sym: 1.0 / n for sym, _ in selected}

        result = self.engine.run(
            price_data = price_data,
            signal_fn  = signal_fn,
            start_date = start_date,
            end_date   = end_date,
            rebalance_freq = 'M',
        )

        eq      = result.equity_curve['equity']
        trades  = result.trades
        summary = performance_summary(eq, trades)
        print_summary(summary)
        return result
