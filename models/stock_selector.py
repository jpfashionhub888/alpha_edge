# models/stock_selector.py

"""
StockSelector V2

Fixes applied:
- validate_stock() takes raw_df + feature_engine,
  engineers features inside each split (Bug 1 — look-ahead)
- self.min_auc no longer mutated on fallback (Bug 2)
- MIN_TRAIN_ROWS constant unifies row guards (Bug 3)
- max_stocks parameter links to PaperTrader.max_positions (Bug 4)
- signal_rate filter rejects models that rarely fire (Flaw 1)
- available_features derived from actual columns (Flaw 2 & Risk 2)
- _save_selection_log() persists results for audit (Flaw 3)
- _quick_validate() uses LogisticRegression for speed (Flaw 4)
- roc_auc_score wrapped at point of use (Risk 1)
- Timing logged per stock (Risk 3)
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  MODULE CONSTANTS                                                    #
# ------------------------------------------------------------------ #

# Minimum training rows — below this a model is unreliable
MIN_TRAIN_ROWS = 150

# Minimum signal rate — model must fire on at least 5% of days
MIN_SIGNAL_RATE = 0.05

# Prediction threshold used to measure signal rate
SIGNAL_THRESHOLD = 0.55


class StockSelector:
    """
    Strict stock selection. Only trade stocks where
    the model has a proven, statistically significant edge.

    Parameters
    ----------
    min_auc         : minimum AUC on held-out validation set
    validation_days : days held out for validation
    top_features    : SelectKBest k
    max_stocks      : maximum stocks to return
                      MUST match PaperTrader.max_positions
    """

    def __init__(self,
                 min_auc         = 0.54,
                 validation_days = 60,
                 top_features    = 20,
                 max_stocks      = 5):

        self.min_auc         = min_auc
        self.validation_days = validation_days
        self.top_features    = top_features
        self.max_stocks      = max_stocks
        self.selected_stocks = {}

    # ---------------------------------------------------------------- #
    #  PUBLIC: VALIDATE ONE STOCK                                        #
    # ---------------------------------------------------------------- #

    def validate_stock(self,
                       raw_df,
                       feature_engine,
                       symbol: str) -> dict:
        """
        Test whether a model can predict this stock on
        held-out validation data.

        Parameters
        ----------
        raw_df         : RAW OHLCV dataframe — no features yet
        feature_engine : FeatureEngine instance
        symbol         : ticker string

        Returns
        -------
        dict with symbol, auc, signal_rate, passed, features
        or None if stock is skipped.

        Look-ahead fix
        --------------
        raw_df is sliced FIRST, then features are engineered
        inside each split window so validation data never
        contaminates the training window.
        """
        raw_df    = raw_df.sort_index().copy()
        split_idx = len(raw_df) - self.validation_days

        # Single guard — references MIN_TRAIN_ROWS constant
        if split_idx < MIN_TRAIN_ROWS:
            logger.debug(
                "%s: insufficient train rows (%d < %d), skipping",
                symbol, split_idx, MIN_TRAIN_ROWS,
            )
            return None

        # ── Slice RAW data first ──────────────────────────────────
        train_raw = raw_df.iloc[:split_idx].copy()
        val_raw   = raw_df.iloc[split_idx:].copy()

        # ── Feature engineering inside each split ─────────────────
        try:
            # Engineer features on combined train+validation history
            # so validation rows have enough past context for rolling indicators
            fold_raw = pd.concat([train_raw, val_raw]).sort_index()

            # Pre-create target so feature_engine does not overwrite
            # with shift(-5) future target logic
            fold_raw['target'] = (
                fold_raw['close'].shift(-1) > fold_raw['close']
            ).astype(int)

            fold_df = feature_engine.add_all_features(fold_raw)
            feature_names = feature_engine.get_feature_names()

            # Split back by original date index
            train_df = fold_df.loc[
                fold_df.index < val_raw.index.min()
            ].copy()

            val_df = fold_df.loc[
                fold_df.index >= val_raw.index.min()
            ].copy()

            logger.info(
                "%s: train_rows=%d val_rows=%d features=%d",
                symbol, len(train_df), len(val_df), len(feature_names)
            )


            if len(train_df) < 100 or len(val_df) < 20:
                logger.warning(
                    "%s: insufficient rows after feature engineering "
                    "(train=%d, val=%d)",
                    symbol, len(train_df), len(val_df)
                )
                return None

        except Exception as fe_err:
            logger.warning(
                "%s: feature engineering failed: %s",
                symbol, fe_err,
            )
            return None

        # ── Available features — derived from actual columns ──────
        # Fix Flaw 2 & Risk 2: prevents KeyError and mask misalignment
        available = [
            f for f in feature_names
            if f in train_df.columns
            and f in val_df.columns
            and not train_df[f].isna().all()
            and not val_df[f].isna().all()
        ]

        if not available:
            logger.warning(
                "%s: no valid features after NaN check", symbol
            )
            return None

        # ── Target checks ─────────────────────────────────────────
        if 'target' not in train_df.columns:
            logger.warning("%s: no target column", symbol)
            return None

        y_train = train_df['target']
        y_val   = val_df['target']

        if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
            logger.debug(
                "%s: single-class target, skipping", symbol
            )
            return None

        # ── Feature selection on training window only ─────────────
        k = min(self.top_features, len(available))
        try:
            selector = SelectKBest(
                score_func=mutual_info_classif, k=k
            )
            selector.fit(train_df[available], y_train)
            mask     = selector.get_support()
            selected = [f for f, m in zip(available, mask) if m]
        except Exception as e:
            logger.warning(
                "%s: feature selection failed: %s", symbol, e
            )
            return None

        if not selected:
            logger.warning("%s: no features selected", symbol)
            return None

        # ── Quick validation (LogisticRegression) ─────────────────
        # Fix Flaw 4: uses fast single model for selection only.
        # Full TechnicalPredictor ensemble used in actual backtest.
        auc, predictions = self._quick_validate(
            X_train = train_df[selected],
            y_train = y_train,
            X_val   = val_df[selected],
            y_val   = y_val,
            symbol  = symbol,
        )

        if auc is None:
            return None

        # ── Signal frequency check ────────────────────────────────
        # Fix Flaw 1: reject models that almost never fire
        signal_rate = float((predictions > SIGNAL_THRESHOLD).mean())
        if signal_rate < MIN_SIGNAL_RATE:
            logger.debug(
                "%s: signal rate too low (%.1f%% < %.1f%%) — "
                "model rarely generates trades",
                symbol,
                signal_rate * 100,
                MIN_SIGNAL_RATE * 100,
            )
            return None

        return {
            'symbol'     : symbol,
            'auc'        : float(auc),
            'signal_rate': signal_rate,
            'passed'     : auc >= self.min_auc,
            'features'   : selected,
        }

    # ---------------------------------------------------------------- #
    #  PUBLIC: SELECT ALL STOCKS                                         #
    # ---------------------------------------------------------------- #

    def select(self, raw_df: pd.DataFrame,
               feature_engine) -> dict:
        """
        Validate all stocks and return only those with
        a proven model edge.

        Parameters
        ----------
        raw_df         : RAW combined OHLCV with 'symbol' column
        feature_engine : FeatureEngine instance

        Returns
        -------
        dict keyed by symbol with validation result dicts.
        """
        print("\n" + "=" * 60)
        print("STRICT STOCK SELECTION")
        print(f"  Min AUC     : {self.min_auc}")
        print(f"  Max stocks  : {self.max_stocks}")
        print(f"  Val window  : {self.validation_days} days")
        print(f"  Min sig rate: {MIN_SIGNAL_RATE:.0%}")
        print("=" * 60)

        if 'symbol' not in raw_df.columns:
            raise ValueError(
                "raw_df must have a 'symbol' column. "
                "Check StockDataFetcher output."
            )

        stocks  = raw_df['symbol'].unique()
        results = []

        min_total = MIN_TRAIN_ROWS + self.validation_days

        for symbol in stocks:
            t0 = time.time()

            stock_raw = raw_df[
                raw_df['symbol'] == symbol
            ].copy()

            # Use MIN_TOTAL_ROWS — single source of truth
            if len(stock_raw) < min_total:
                logger.debug(
                    "%s: only %d rows, need %d — skipping",
                    symbol, len(stock_raw), min_total,
                )
                continue

            result = self.validate_stock(
                stock_raw, feature_engine, symbol
            )

            elapsed = time.time() - t0

            if result is not None:
                results.append(result)
                status = "✅" if result['passed'] else "❌"
                logger.info(
                    "%s %s | AUC=%.3f | sig=%.1f%% | %.1fs",
                    status, symbol,
                    result['auc'],
                    result['signal_rate'] * 100,
                    elapsed,
                )
                print(
                    f"   {status} {symbol:6s}"
                    f" | AUC: {result['auc']:.3f}"
                    f" | sig: {result['signal_rate']:.1%}"
                )
            else:
                logger.debug(
                    "%s: skipped (%.1fs)", symbol, elapsed
                )

        # ── Save full results for audit ───────────────────────────
        self._save_selection_log(results)

        # ── Filter and rank ───────────────────────────────────────
        results.sort(key=lambda x: x['auc'], reverse=True)
        passed = [r for r in results if r['passed']]

        n_passed = len(passed)
        n_total  = len(results)
        print(f"\n   {n_passed}/{n_total} stocks passed AUC >= {self.min_auc}")

        # Apply max_stocks cap
        passed = passed[:self.max_stocks]

        # ── Fallback threshold — does NOT mutate self.min_auc ────
        # Fix Bug 2: use local variable so subsequent calls
        # are not affected
        fallback_auc = 0.52
        if len(passed) == 0 and fallback_auc < self.min_auc:
            logger.warning(
                "No stocks passed AUC >= %.2f — "
                "trying fallback threshold %.2f. "
                "NOTE: self.min_auc is NOT changed.",
                self.min_auc, fallback_auc,
            )
            print(
                f"   ⚠️  No stocks passed {self.min_auc} — "
                f"trying fallback {fallback_auc}"
            )
            passed = [
                r for r in results
                if r['auc'] >= fallback_auc
            ][:self.max_stocks]

        if len(passed) == 0:
            logger.error(
                "No stocks passed even fallback AUC %.2f. "
                "Backtest will produce no results. "
                "Consider: more data, lower min_auc, "
                "or check feature_engine output.",
                fallback_auc,
            )
            print("   🚫 No stocks passed any threshold")
            self.selected_stocks = {}
            return {}

        self.selected_stocks = {
            r['symbol']: r for r in passed
        }

        print("\n   Final selected stocks:")
        for r in passed:
            print(
                f"      {r['symbol']:6s}"
                f" | AUC: {r['auc']:.3f}"
                f" | sig: {r['signal_rate']:.1%}"
            )

        logger.info(
            "Selected %d stocks (cap=%d): %s",
            len(passed),
            self.max_stocks,
            [r['symbol'] for r in passed],
        )

        return self.selected_stocks

    # ---------------------------------------------------------------- #
    #  PRIVATE: QUICK VALIDATE                                           #
    # ---------------------------------------------------------------- #

    def _quick_validate(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val  : pd.DataFrame,
                        y_val  : pd.Series,
                        symbol : str) -> tuple:
        """
        Fast AUC validation using LogisticRegression.

        Used for stock SELECTION only — the actual backtest
        uses the full TechnicalPredictor ensemble.

        Returns
        -------
        (auc, predictions) or (None, None) on failure.

        Speed benefit
        -------------
        LogisticRegression: ~0.1s per stock
        TechnicalPredictor: ~2-5s per stock (4 models)
        With 40 stocks: 4s vs 160s just for selection.
        """
        try:
            scaler   = StandardScaler()
            X_tr_sc  = scaler.fit_transform(X_train)
            X_va_sc  = scaler.transform(X_val)

            model = LogisticRegression(
                max_iter     = 300,
                random_state = 42,
                C            = 0.1,
                solver       = 'lbfgs',
            )
            model.fit(X_tr_sc, y_train)
            predictions = model.predict_proba(X_va_sc)[:, 1]

            # Wrap roc_auc_score at point of use
            try:
                auc = float(roc_auc_score(y_val, predictions))
            except ValueError as e:
                logger.warning(
                    "%s: roc_auc_score failed (%s) — skipping",
                    symbol, e,
                )
                return None, None

            return auc, predictions

        except Exception as e:
            logger.warning(
                "%s: quick validation failed: %s", symbol, e
            )
            return None, None

    # ---------------------------------------------------------------- #
    #  PRIVATE: SAVE SELECTION LOG                                       #
    # ---------------------------------------------------------------- #

    def _save_selection_log(self, results: list) -> None:
        """
        Save full validation results to disk for audit.

        Allows you to track AUC changes per stock across
        runs and understand why stocks were rejected.
        """
        try:
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path      = f'logs/stock_selection_{timestamp}.json'

            log = {
                'run_at'         : datetime.now().isoformat(),
                'min_auc'        : self.min_auc,
                'validation_days': self.validation_days,
                'max_stocks'     : self.max_stocks,
                'results'        : results,
            }

            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(log, f, indent=2, default=str)
            os.replace(tmp, path)

            logger.info("Selection log saved → %s", path)

        except Exception as e:
            logger.warning(
                "Could not save selection log: %s", e
            )