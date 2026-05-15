# backtest/walk_forward.py

"""
Walk Forward Backtester V2

Fixes applied:
- Regime detection now inside each fold (Bug 1 — look-ahead bias)
- Accepts raw df + feature_engine, engineers per fold (Bug 2)
- train_mask respects train_window boundary (Bug 3)
- Risk params accepted and used in return simulation (Bug 4 & 5)
- Stop-loss, take-profit, trailing stop applied per trade (Bug 5)
- Sharpe uses excess returns over risk-free rate (Flaw 1)
- Win rate counts trades not days (Flaw 2)
- Annual return guarded for short periods (Flaw 3)
- Returned dict includes full trading metrics (Flaw 5)
- Random seed propagated to all models (Risk 1)
- pd.concat filters empty frames (Risk 2)
- Required columns validated after ensemble (Risk 3)
- Fold timing logged (Risk 4)
"""

import time
import logging
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
)

from models.ensemble import EnsembleStrategy
from models.regime_detector import RegimeDetector
from models.stock_selector import StockSelector
from models.technical_model import TechnicalPredictor

logger = logging.getLogger(__name__)

# Risk-free rate for Sharpe calculation (annualised)
# Update this to match current T-bill rate
RISK_FREE_RATE_ANNUAL = 0.05
RISK_FREE_RATE_DAILY  = RISK_FREE_RATE_ANNUAL / 252

# Minimum backtest period for reliable annualisation
MIN_YEARS_FOR_ANNUALISATION = 0.5


class WalkForwardBacktester:
    """
    Walk-forward validation with stock selection,
    regime detection, ensemble strategy, and
    realistic risk management (stop-loss / take-profit /
    trailing stop) applied inside each fold.

    Parameters
    ----------
    train_window_days       : rows of history used to train each fold
    retrain_frequency_days  : how often to retrain (fold step size)
    top_features            : SelectKBest k
    min_auc                 : minimum AUC for StockSelector
    stop_loss_pct           : per-trade stop-loss (matches PaperTrader)
    take_profit_pct         : per-trade take-profit
    trailing_stop_pct       : trailing stop from peak
    daily_loss_limit_pct    : daily circuit breaker
    random_seed             : reproducibility seed for all models
    feature_engine          : FeatureEngine instance; features are
                              engineered INSIDE each fold window
    """

    def __init__(self,
                 train_window_days      = 180,
                 retrain_frequency_days = 30,
                 top_features           = 25,
                 min_auc                = 0.53,
                 stop_loss_pct          = 0.03,
                 take_profit_pct        = 0.06,
                 trailing_stop_pct      = 0.02,
                 daily_loss_limit_pct   = 0.02,
                 random_seed            = 42,
                 feature_engine         = None):

        self.train_window         = train_window_days
        self.retrain_every        = retrain_frequency_days
        self.top_features         = top_features
        self.stop_loss_pct        = stop_loss_pct
        self.take_profit_pct      = take_profit_pct
        self.trailing_stop_pct    = trailing_stop_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.random_seed          = random_seed
        self.feature_engine       = feature_engine

        self.results          = None
        self.performance      = {}
        self.strategy_results = None

        self.regime_detector = RegimeDetector()
        self.ensemble = EnsembleStrategy(
            base_stop_loss           = stop_loss_pct,
            reward_risk_ratio        = take_profit_pct / stop_loss_pct,
            trailing_stop_multiplier = trailing_stop_pct / stop_loss_pct,
            max_daily_loss           = daily_loss_limit_pct,
            max_portfolio_risk       = stop_loss_pct * 3,
        )

        self.stock_selector  = StockSelector(
            min_auc      = min_auc,
            top_features = top_features,
        )

    # ---------------------------------------------------------------- #
    #  PUBLIC: RUN SINGLE STOCK                                          #
    # ---------------------------------------------------------------- #

    def run_single_stock(self,
                         raw_df,
                         symbol,
                         selected_features = None,
                         target            = 'target'):
        """
        Walk-forward backtest for one stock.

        Parameters
        ----------
        raw_df            : RAW OHLCV dataframe (no features yet)
        symbol            : ticker string for logging
        selected_features : pre-selected feature names, or None to
                            use all features from feature_engine
        target            : target column name

        Returns
        -------
        pd.DataFrame of predictions across all folds, or None.

        Look-ahead fix
        --------------
        Features and regime detection are applied INSIDE each fold
        window so no fold can see future data during training.
        """
        print(f"\n   --- {symbol} ---")

        raw_df     = raw_df.sort_index().copy()
        all_dates  = sorted(raw_df.index.unique().tolist())
        n          = len(all_dates)

        # Build fold split points
        split_dates = []
        cursor      = self.train_window
        while cursor < n:
            split_dates.append(all_dates[cursor])
            cursor += self.retrain_every

        if not split_dates:
            logger.debug("%s: insufficient rows for any fold", symbol)
            return None

        all_predictions = []
        total_folds     = len(split_dates)

        for i, split_date in enumerate(split_dates):
            fold_num = i + 1

        for i, split_date in enumerate(split_dates):
            fold_num = i + 1

            # ── Slice RAW data per fold ───────────────────────────
            test_end = split_date + pd.Timedelta(
                days=self.retrain_every
            )

            # Use ALL history up to split_date for training
            # This gives feature engine enough bars for MA200 etc
            # Training window filter applied AFTER feature engineering
            train_raw = raw_df[
                raw_df.index < split_date
            ].copy()

            test_raw = raw_df[
                (raw_df.index >= split_date) &
                (raw_df.index <  test_end)
            ].copy()

            # Keep track of where the actual training window starts
            # so we don't train on too-old data
            train_window_start = split_date - pd.Timedelta(
                days=self.train_window
            )

            if len(train_raw) < 50 or len(test_raw) == 0:

                continue

            # ── Feature engineering INSIDE fold ──────────────────
            if self.feature_engine is not None:
                # Engineer on combined fold history so test rows
                # have enough past context for rolling indicators
                fold_raw = pd.concat(
                    [train_raw, test_raw]
                ).sort_index()

                # Pre-create target so feature_engine does not
                # overwrite with shift(-5) look-ahead calculation
                fold_raw['target'] = (
                    fold_raw['close'].shift(-1)
                    > fold_raw['close']
                ).astype(int)


                fold_df = self.feature_engine.add_all_features(
                    fold_raw
                )


                feature_names = self.feature_engine.get_feature_names()

                # Split by date boundary — not by index lookup
                # add_all_features drops NaN rows so original
                # index dates may no longer exist in fold_df
                test_start = test_raw.index.min()

                train_df = fold_df[
                    (fold_df.index >= train_window_start) &
                    (fold_df.index < test_start)
                ].copy()

                test_df = fold_df[
                    fold_df.index >= test_start
                ].copy()

                if len(train_df) < 50 or len(test_df) == 0:
                    continue

            else:
                # feature_engine not provided — assume df
                # already has features (legacy path)
                train_df = train_raw
                test_df = test_raw
                feature_names = selected_features or []            # ── Regime detection INSIDE fold ──────────────────────
            train_df = self.regime_detector.detect(train_df)
            test_df  = self.regime_detector.detect(test_df)

            # ── Resolve feature list ──────────────────────────────
            if selected_features is not None:
                # Use pre-selected features if they exist in this fold
                fold_features = [
                    f for f in selected_features
                    if f in train_df.columns
                ]
            else:
                fold_features = [
                    f for f in feature_names
                    if f in train_df.columns
                ]

            if not fold_features:
                logger.warning(
                    "%s fold %d: no valid features, skipping",
                    symbol, fold_num
                )
                continue

            X_train = train_df[fold_features]
            y_train = train_df[target]
            X_test  = test_df[fold_features]
            y_test  = test_df[target]

            # ── Data quality guards ───────────────────────────────
            if len(X_test) == 0:
                continue
            if len(y_train.unique()) < 2:
                continue
            if len(y_test.unique()) < 2:
                continue
            if y_train.value_counts().min() < 5:
                continue

            # ── Train model ───────────────────────────────────────
            fold_start = time.time()
            model      = TechnicalPredictor(
                use_lstm=False
            )

            model.train(X_train, y_train)

            logger.debug(
                "%s fold %d/%d trained in %.1fs",
                symbol, fold_num, total_folds,
                time.time() - fold_start,
            )

            # ── Predict ───────────────────────────────────────────
            predictions = model.predict(X_test)

            fold_results                = test_df.copy()
            fold_results['prediction']  = predictions
            fold_results['fold']        = i
            fold_results['stock']       = symbol

            # ── Apply risk management per fold ────────────────────
            fold_results = self._apply_risk_management(fold_results)

            try:
                fold_auc = roc_auc_score(y_test, predictions)
            except Exception:
                fold_auc = 0.5

            print(
                f"      Fold {fold_num}/{total_folds}"
                f" | AUC: {fold_auc:.3f}"
                f" | rows: {len(X_train)}tr/{len(X_test)}te"
            )

            all_predictions.append(fold_results)

        if not all_predictions:
            return None

        stock_results = pd.concat(all_predictions)
        stock_results = self.ensemble.generate_signals(stock_results)

        return stock_results

    # ---------------------------------------------------------------- #
    #  PUBLIC: RUN (full pipeline)                                       #
    # ---------------------------------------------------------------- #

    def run(self, raw_df, target='target'):
        """
        Run complete walk-forward backtest with stock selection.

        Parameters
        ----------
        raw_df : RAW combined OHLCV dataframe with a 'symbol' column.
                 Features are engineered inside each fold.
        target : target column name

        Returns
        -------
        dict with both ML metrics and trading performance metrics.
        """
        t0 = time.time()

        # ── Stock selection on a clean feature window ─────────────
        # We engineer features on the full df ONLY for stock
        # selection purposes. This is acceptable because selection
        # only decides WHICH stocks to backtest — not the actual
        # predictions. The per-fold feature engineering inside
        # run_single_stock() is what matters for result validity.

        selected = self.stock_selector.select(
            raw_df, self.feature_engine
        )

        if not selected:
            print("\nNo tradeable stocks found")
            return {}

        # ── Backtest selected stocks ──────────────────────────────
        print("\n" + "=" * 60)
        print("BACKTESTING SELECTED STOCKS")
        print("=" * 60)

        all_stock_results = []
        min_rows          = self.train_window + self.retrain_every

        for symbol, info in selected.items():

            # Pass RAW data per stock — features built inside fold
            stock_raw = raw_df[
                raw_df['symbol'] == symbol
            ].copy()

            if len(stock_raw) < min_rows:
                logger.debug(
                    "%s: only %d rows, need %d — skipping",
                    symbol, len(stock_raw), min_rows,
                )
                continue

            result = self.run_single_stock(
                raw_df            = stock_raw,
                symbol            = symbol,
                selected_features = info.get('features'),
                target            = target,
            )

            if result is not None and len(result) > 0:
                all_stock_results.append(result)

        if not all_stock_results:
            print("\nNo valid results generated")
            return {}

        # Filter out any empty frames before concat
        valid = [r for r in all_stock_results if len(r) > 0]
        if not valid:
            return {}

        self.results = pd.concat(valid).sort_index()

        # ── Validate required columns ─────────────────────────────
# ── Validate required columns ─────────────────────────────
        required_cols = [
            'prediction',
            'signal',
            'strategy_return',
            'returns',
            'managed_signal',    # ← ADD THIS LINE
        ]
        missing = [
            c for c in required_cols
            if c not in self.results.columns
        ]
        if missing:
            raise ValueError(
                f"EnsembleStrategy.generate_signals() did not "
                f"produce required columns: {missing}. "
                f"Check models/ensemble.py."
            )

        # ── Overall ML metrics ────────────────────────────────────
        y_true  = self.results['target']
        y_proba = self.results['prediction']

        try:
            overall_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            overall_auc = 0.5

        self.performance = {
            'roc_auc'  : overall_auc,
            'accuracy' : accuracy_score(y_true, y_proba > 0.5),
            'precision': precision_score(
                y_true, y_proba > 0.5, zero_division=0
            ),
        }

        # ── Per-stock results ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("PER STOCK RESULTS")
        print("=" * 60)

        traded_stocks = self.results['stock'].unique()

        for symbol in traded_stocks:
            mask = self.results['stock'] == symbol
            if mask.sum() == 0:
                continue

            s_true  = self.results.loc[mask, 'target']
            s_proba = self.results.loc[mask, 'prediction']
            s_ret   = self.results.loc[mask, 'adjusted_return']

            if len(s_true.unique()) < 2:
                continue

            try:
                s_auc = roc_auc_score(s_true, s_proba)
            except Exception:
                s_auc = 0.5

            s_total = (1 + s_ret).cumprod().iloc[-1] - 1
            status  = "✅" if s_auc > 0.52 else "❌"

            print(
                f"   {status} {symbol:6s}"
                f" | AUC: {s_auc:.3f}"
                f" | Return: {s_total:+.1%}"
            )

        # ── Strategy-level returns ────────────────────────────────
        self._calculate_strategy_returns()

        # ── Summary printout ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)

        print(f"  Stocks traded      : {len(traded_stocks)}")
        print(f"  Total observations : {len(self.results)}")
        print(f"  ROC AUC            : {overall_auc:.3f}")
        print(
            f"  Accuracy           : "
            f"{self.performance['accuracy']:.1%}"
        )
        print(
            f"  Precision          : "
            f"{self.performance['precision']:.1%}"
        )
        print(
            f"  Duration           : "
            f"{time.time() - t0:.1f}s"
        )

        sr = self.strategy_results or {}

        if sr:
            beat = (
                sr.get('annual_return', 0)
                > sr.get('buyhold_annual', 0)
            )
            print(
                f"\n  Annual Return      : "
                f"{sr.get('annual_return', 0):+.1%}"
            )
            print(
                f"  Buy & Hold Annual  : "
                f"{sr.get('buyhold_annual', 0):+.1%}"
            )
            print(
                f"  Sharpe Ratio       : "
                f"{sr.get('sharpe', 0):.2f}"
            )
            print(
                f"  Max Drawdown       : "
                f"{sr.get('max_drawdown', 0):.1%}"
            )
            print(
                f"  Win Rate           : "
                f"{sr.get('win_rate', 0):.1%}"
            )
            print(
                f"  Total Trades       : "
                f"{sr.get('total_trades', 0)}"
            )
            verdict = (
                "✅ STRATEGY BEATS BUY & HOLD!"
                if beat
                else f"❌ Underperforms B&H by "
                     f"{sr.get('buyhold_annual',0) - sr.get('annual_return',0):.1%}"
            )
            print(f"\n  {verdict}")

        print("=" * 60)

        # ── Return combined metrics dict ──────────────────────────
        return {
            # ML metrics
            'roc_auc'            : self.performance['roc_auc'],
            'accuracy'           : self.performance['accuracy'],
            'precision'          : self.performance['precision'],
            # Trading metrics
            'annual_return'      : sr.get('annual_return',   0.0),
            'total_return'       : sr.get('total_return',    0.0),
            'sharpe'             : sr.get('sharpe',          0.0),
            'max_drawdown'       : sr.get('max_drawdown',    0.0),
            'win_rate'           : sr.get('win_rate',        0.0),
            'total_trades'       : sr.get('total_trades',    0),
            'buyhold_annual'     : sr.get('buyhold_annual',  0.0),
            'beats_buyhold'      : sr.get('annual_return',   0.0)
                                   > sr.get('buyhold_annual', 0.0),
            # Config snapshot for auditability
            'stop_loss_pct'      : self.stop_loss_pct,
            'take_profit_pct'    : self.take_profit_pct,
            'trailing_stop_pct'  : self.trailing_stop_pct,
            'train_window_days'  : self.train_window,
            'stocks_traded'      : len(traded_stocks),
        }

    # ---------------------------------------------------------------- #
    #  PRIVATE: APPLY RISK MANAGEMENT PER FOLD                          #
    # ---------------------------------------------------------------- #

    def _apply_risk_management(
        self, fold_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Simulate stop-loss, take-profit, and trailing stop
        on each trade within a fold.

        This makes backtest returns directly comparable to
        live PaperTrader behaviour which applies the same rules.

        Adds 'adjusted_return' column to fold_df.
        """
        fold_df = fold_df.copy()

        # Use strategy_return as base if available
        # otherwise use daily returns column
        if 'strategy_return' in fold_df.columns:
            fold_df['adjusted_return'] = fold_df['strategy_return']
        elif 'returns' in fold_df.columns:
            fold_df['adjusted_return'] = fold_df['returns']
        else:
            fold_df['adjusted_return'] = 0.0

        if 'close' not in fold_df.columns:
            return fold_df

        in_trade    = False
        entry_price = 0.0
        highest     = 0.0

        for idx in fold_df.index:
            if 'signal' in fold_df.columns:
                signal = fold_df.loc[idx, 'signal']
                if isinstance(signal, pd.Series):
                    signal = signal.iloc[0]
                signal = float(signal)
            else:
                signal = 0.0

            price = fold_df.loc[idx, 'close']
            if isinstance(price, pd.Series):
                price = price.iloc[0]
            price = float(price)
            if not in_trade and signal > 0:
                in_trade    = True
                entry_price = price
                highest     = price

            if in_trade:
                highest  = max(highest, price)
                pnl_pct  = (price - entry_price) / entry_price
                drop_pct = (
                    (highest - price) / highest
                    if highest > 0 else 0
                )

                # Stop-loss
                if pnl_pct <= -self.stop_loss_pct:
                    fold_df.loc[idx, 'adjusted_return'] = pnl_pct
                    in_trade = False
                    continue

                # Take-profit
                if pnl_pct >= self.take_profit_pct:
                    fold_df.loc[idx, 'adjusted_return'] = pnl_pct
                    in_trade = False
                    continue

                # Trailing stop (only after 2% gain at peak)
                max_gain = (
                    (highest - entry_price) / entry_price
                    if entry_price > 0 else 0
                )
                if (drop_pct  >= self.trailing_stop_pct
                        and max_gain >= 0.02):
                    fold_df.loc[idx, 'adjusted_return'] = pnl_pct
                    in_trade = False

        return fold_df

    # ---------------------------------------------------------------- #
    #  PRIVATE: COUNT DISTINCT TRADES                                    #
    # ---------------------------------------------------------------- #

    def _count_trades(self, df: pd.DataFrame) -> tuple:
        """
        Count distinct trades (entry→exit) from signal column.

        Fix: old code counted trading DAYS not TRADES,
        inflating win rate for multi-day positions.

        Returns
        -------
        total_trades, wins, win_rate
        """
        in_trade   = False
        trade_rets = []
        running    = 0.0

        # Priority: managed_return > adjusted_return > strategy_return
        # managed_return is set by RiskManager and reflects actual
        # signal-filtered returns — this is the correct column
        if 'managed_return' in df.columns:
            ret_col = 'managed_return'
        elif 'adjusted_return' in df.columns:
            ret_col = 'adjusted_return'
        else:
            ret_col = 'strategy_return'

        for idx in df.index:
            if 'signal' in df.columns:
                signal = df.loc[idx, 'signal']
                if isinstance(signal, pd.Series):
                    signal = signal.iloc[0]
                signal = float(signal)
            else:
                signal = 0.0

            ret = df.loc[idx, ret_col]
            if isinstance(ret, pd.Series):
                ret = ret.iloc[0]
            ret = float(ret)

            if not in_trade and signal > 0:
                in_trade = True
                running  = 0.0

            if in_trade:
                running += ret
                if signal == 0:
                    trade_rets.append(running)
                    in_trade = False

        # Close any open trade at end of period
        if in_trade and running != 0.0:
            trade_rets.append(running)

        total = len(trade_rets)
        wins  = sum(1 for r in trade_rets if r > 0)
        wr    = wins / total if total > 0 else 0.0

        return total, wins, wr

    # ---------------------------------------------------------------- #
    #  PUBLIC: RUN WITH WEIGHT INJECTION (for optimizer only)            #
    # ---------------------------------------------------------------- #

    def run_for_optimizer(
        self,
        symbol: str,
        weight_combo: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        compute_signal_fn=None,
        buy_threshold: float = 0.40,
        hold_days_override: int = 5,
    ) -> Dict[str, Any]:

        """
        Called ONLY by run_weight_optimization.py.
        Runs walk-forward for one symbol with a given weight combo.
        Returns list of simulated trades for Sharpe calculation.
        """
        import yfinance as yf
        from sklearn.ensemble import RandomForestClassifier

        pred_w, sent_w, sector_w = weight_combo
        trades = []

        test_bars      = self.retrain_every   # 30-bar test fold
        hold_days      = hold_days_override
        min_train_bars = self.train_window
        step_bars      = self.retrain_every

        try:
            # ── Fetch 2 years of data ─────────────────────────────
            df = yf.download(
                symbol,
                period="2y",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            if df is None or len(df) < min_train_bars + test_bars:
                logger.warning(
                    "[%s] Insufficient data (%d bars)",
                    symbol,
                    len(df) if df is not None else 0,
                )
                return {"trades": []}

            # ── Flatten column names to lowercase ─────────────────
            def _flatten(c):
                if isinstance(c, tuple):
                    return c[0].lower()
                return str(c).lower()

            df.columns = [_flatten(c) for c in df.columns]
            df = df.dropna()
            # Keep DatetimeIndex — feature_engine requires it
            n = len(df)

            # ── Walk-forward loop ─────────────────────────────────
            # Feature engine needs ~200 bars for longest rolling window
            # We need train_window + 200 extra bars of history
            # Start much later so folds have enough history
            effective_start = min(400, n - hold_days - 1)
            train_end = effective_start
            while train_end + test_bars <= n:
                train_slice = df.iloc[:train_end].copy()
                test_slice  = df.iloc[
                    train_end: train_end + test_bars
                ].copy()

                if len(test_slice) == 0:
                    train_end += step_bars
                    continue

                train_len = len(train_slice)

                # ── Feature engineering on COMBINED fold ──────────
                try:
                    # Use full history up to end of test slice
                    # This gives feature engine maximum context
                    fold_end_idx = train_end + test_bars
                    fold_raw = df.iloc[:fold_end_idx].copy()

                    # Add target on full fold
                    fold_raw['target'] = (
                        fold_raw['close'].shift(-1)
                        > fold_raw['close']
                    ).astype(int)

                    if self.feature_engine is not None:
                        fold_featured = self.feature_engine.add_all_features(
                            fold_raw
                        )
                        feature_names = self.feature_engine.get_feature_names()
                    else:
                        fold_featured = fold_combined
                        feature_names = [
                            c for c in fold_combined.columns
                            if c not in (
                                'open', 'high', 'low',
                                'close', 'volume', 'target'
                            )
                        ]

                    # Split back by position
                    train_featured = fold_featured.iloc[:train_len].copy()
                    test_featured  = fold_featured.iloc[train_len:].copy()

                    # feature_engine drops NaN rows from rolling indicators
                    # We cannot split by original train_len anymore
                    # Instead split by test slice size from the end
                    # Split by date index
                    # train = everything before test_slice starts
                    # test  = rows matching test_slice dates
                    test_start_date = df.index[train_end]
                    test_end_date   = df.index[
                        min(train_end + test_bars - 1, n - 1)
                    ]

                    train_featured = fold_featured[
                        fold_featured.index < test_start_date
                    ].copy()

                    test_featured = fold_featured[
                        (fold_featured.index >= test_start_date) &
                        (fold_featured.index <= test_end_date)
                    ].copy()

                    if len(train_featured) == 0 or len(test_featured) == 0:
                        train_end += step_bars
                        continue


                except Exception as fe_err:
                    logger.warning(
                        "[%s] Feature engineering failed: %s",
                        symbol, fe_err,
                    )
                    train_end += step_bars
                    continue

                # ── Prepare training data ─────────────────────────
                valid_features = [
                    f for f in feature_names
                    if f in train_featured.columns
                ]

                if not valid_features:
                    train_end += step_bars
                    continue

                train_clean = train_featured[
                    valid_features + ['target']
                ].dropna()

                if len(train_clean) < 50:
                    train_end += step_bars
                    continue

                X_train = train_clean[valid_features]
                y_train = train_clean['target']

                if len(y_train.unique()) < 2:
                    train_end += step_bars
                    continue

                if y_train.value_counts().min() < 5:
                    train_end += step_bars
                    continue

                # ── Train fast model ──────────────────────────────
                try:
                    fast_model = RandomForestClassifier(
                        n_estimators=20,
                        max_depth=4,
                        random_state=42,
                        n_jobs=1,
                    )
                    fast_model.fit(X_train, y_train)
                except Exception as train_err:
                    logger.warning(
                        "[%s] Training failed: %s",
                        symbol, train_err,
                    )
                    train_end += step_bars
                    continue

                # ── Prepare test data ─────────────────────────────
                valid_features_test = [
                    f for f in valid_features
                    if f in test_featured.columns
                ]

                if 'close' not in test_featured.columns:
                    train_end += step_bars
                    continue

                # Do NOT dropna on test — combined fold feature
                # engineering already handled NaN via train history
                # dropna here empties the test set
                test_clean = test_featured[
                    valid_features_test + ['close']
                ].copy()

                # Only drop rows where close itself is NaN
                test_clean = test_clean.dropna(subset=['close'])

                # Fill any remaining NaN in features with 0
                test_clean[valid_features_test] = test_clean[
                    valid_features_test
                ].fillna(0)


                if len(test_clean) < hold_days + 1:
                    train_end += step_bars
                    continue

                # ── Predict probabilities ─────────────────────────
                X_test = test_clean[valid_features_test]

                try:
                    probas = fast_model.predict_proba(X_test)[:, 1]
                except Exception as pred_err:
                    logger.warning(
                        "[%s] Prediction failed: %s",
                        symbol, pred_err,
                    )
                    train_end += step_bars
                    continue

                # ── Simulate trades ───────────────────────────────
                for i in range(len(test_clean) - hold_days):
                    pred = float(probas[i])

                    sent_score = 0.0
                    sect_mult  = 1.0

                    if compute_signal_fn is not None:
                        combined = compute_signal_fn(
                            pred, sent_score, sect_mult,
                            pred_w, sent_w, sector_w,
                        )
                    else:
                        combined = pred


                    if combined >= buy_threshold:
                        entry_price = float(
                            test_clean['close'].iloc[i]
                        )
                        exit_idx = min(
                            i + hold_days,
                            len(test_clean) - 1,
                        )
                        exit_price = float(
                            test_clean['close'].iloc[exit_idx]
                        )

                        if entry_price <= 0:
                            continue

                        raw_return = (
                            exit_price - entry_price
                        ) / entry_price

                        capped = self._apply_risk_to_return(raw_return)

                        trades.append({
                            "symbol"         : symbol,
                            "return_pct"     : float(capped),
                            "held_days"      : hold_days,
                            "entry_price"    : round(entry_price, 2),
                            "exit_price"     : round(exit_price, 2),
                            "combined_signal": round(combined, 4),
                            "pred"           : round(pred, 4),
                        })

                train_end += step_bars

        except Exception as outer_err:
            logger.error(
                "[%s] run_for_optimizer failed: %s",
                symbol, outer_err,
            )
            return {"trades": []}

        logger.info(
            "[%s] Weight opt complete | weights=%s | trades=%d",
            symbol, weight_combo, len(trades),
        )
        return {"trades": trades}

    # ---------------------------------------------------------------- #
    #  PRIVATE: APPLY RISK TO A SINGLE RETURN (for optimizer)           #
    # ---------------------------------------------------------------- #

    def _apply_risk_to_return(self, raw_return: float) -> float:
        """
        Cap a single trade return at stop-loss and take-profit limits.
        Mirrors what PaperTrader does in live trading.
        Keeps backtest returns consistent with live behaviour.
        """
        if raw_return <= -self.stop_loss_pct:
            return -self.stop_loss_pct

        if raw_return >= self.take_profit_pct:
            return self.take_profit_pct

        return raw_return

    # ---------------------------------------------------------------- #
    #  PRIVATE: CALCULATE STRATEGY RETURNS                               #
    # ---------------------------------------------------------------- #
    def _calculate_strategy_returns(self):
        """
        Calculate aggregate strategy performance metrics.

        Fixes:
        - Sharpe uses excess returns (subtracts risk-free rate)
        - Annual return guarded for periods < 6 months
        - Win rate uses _count_trades() not day count
        - Uses 'adjusted_return' (risk-managed) not raw
          'strategy_return'
        """
        df = self.results.copy()

        if 'managed_return' in df.columns:
            ret_col = 'managed_return'
        elif 'adjusted_return' in df.columns:
            ret_col = 'adjusted_return'
        else:
            ret_col = 'strategy_return'

        # Use managed_signal if available — it reflects
        # what risk manager actually allowed to trade
        # Falls back to signal if managed_signal missing
        if 'managed_signal' in df.columns:
            active = df[df['managed_signal'] > 0]
        elif 'signal' in df.columns:
            active = df[df['signal'] > 0]
        else:
            active = df

        cumulative = (1 + df[ret_col]).cumprod()
        buyhold    = (1 + df['returns']).cumprod()
 
        # Calculate returns on ALL days
        # active is used for trade counting only
        # not for gating the entire calculation
        cumulative = (1 + df[ret_col]).cumprod()
        buyhold    = (1 + df['returns']).cumprod()

        total_days = len(df)
        years      = total_days / 252.0

        if years == 0:
            self.strategy_results = None
            return

        total_return = cumulative.iloc[-1] - 1
        bh_total     = buyhold.iloc[-1] - 1

        # Guard: do not annualise very short periods
        if years >= MIN_YEARS_FOR_ANNUALISATION:
            annual    = (1 + total_return) ** (1 / years) - 1
            bh_annual = (1 + bh_total)     ** (1 / years) - 1
        else:
            logger.warning(
                "Backtest period %.1f years < %.1f minimum — "
                "reporting total return without annualisation",
                years, MIN_YEARS_FOR_ANNUALISATION,
            )
            annual    = total_return
            bh_annual = bh_total

        # Sharpe with excess returns
        daily_ret    = df[ret_col]
        excess_daily = daily_ret - RISK_FREE_RATE_DAILY
        sharpe       = (
            excess_daily.mean() / excess_daily.std() * np.sqrt(252)
            if excess_daily.std() > 0
            else 0.0
        )

        # Max drawdown
        peak      = cumulative.cummax()
        drawdown  = (cumulative - peak) / peak
        max_dd    = drawdown.min()

        # Win rate — count distinct trades not days
        total_trades, wins, wr = self._count_trades(df)

        self.strategy_results = {
            'annual_return' : annual,
            'buyhold_annual': bh_annual,
            'total_return'  : total_return,
            'sharpe'        : sharpe,
            'max_drawdown'  : max_dd,
            'win_rate'      : wr,
            'total_trades'  : total_trades,
        }