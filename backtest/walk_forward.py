# backtest/walk_forward.py

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from models.technical_model import TechnicalPredictor
from models.regime_detector import RegimeDetector
from models.ensemble import EnsembleStrategy
from models.stock_selector import StockSelector
import logging

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Walk forward validation with stock selection,
    regime detection and ensemble strategy.
    """

    def __init__(self,
                 train_window_days=180,
                 retrain_frequency_days=30,
                 top_features=25,
                 min_auc=0.53):

        self.train_window = train_window_days
        self.retrain_every = retrain_frequency_days
        self.top_features = top_features

        self.results = None
        self.performance = {}
        self.strategy_results = None
        self.regime_detector = RegimeDetector()
        self.ensemble = EnsembleStrategy()
        self.stock_selector = StockSelector(
            min_auc=min_auc,
            top_features=top_features
        )

    def run_single_stock(self, df, feature_names, symbol,
                         selected_features=None,
                         target='target'):
        """Run walk forward backtest for one stock."""

        print(f"\n   --- {symbol} ---")

        df = self.regime_detector.detect(df)

        df = df.sort_index().copy()
        all_dates = sorted(list(df.index.unique()))

        split_dates = []
        current_split = self.train_window

        while current_split < len(all_dates):
            split_dates.append(all_dates[current_split])
            current_split += self.retrain_every

        if len(split_dates) == 0:
            return None

        all_predictions = []

        if selected_features is None:
            selected_features = feature_names

        for i, split_date in enumerate(split_dates):

            fold_num = i + 1
            total_folds = len(split_dates)

            train_mask = df.index < split_date
            test_end = split_date + pd.Timedelta(days=30)
            test_mask = (
                (df.index >= split_date)
                & (df.index < test_end)
            )

            X_train = df.loc[train_mask, selected_features]
            y_train = df.loc[train_mask, target]
            X_test = df.loc[test_mask, selected_features]
            y_test = df.loc[test_mask, target]

            if len(X_test) == 0:
                continue
            if len(y_train.unique()) < 2:
                continue
            if len(y_test.unique()) < 2:
                continue

            model = TechnicalPredictor()
            model.train(X_train, y_train)
            predictions = model.predict(X_test)

            fold_results = df.loc[test_mask].copy()
            fold_results['prediction'] = predictions
            fold_results['fold'] = i
            fold_results['stock'] = symbol

            all_predictions.append(fold_results)

            fold_auc = roc_auc_score(y_test, predictions)
            print(
                f"      Fold {fold_num}/{total_folds}"
                f" | AUC: {fold_auc:.3f}"
            )

        if len(all_predictions) == 0:
            return None

        stock_results = pd.concat(all_predictions)
        stock_results = self.ensemble.generate_signals(
            stock_results
        )

        return stock_results

    def run(self, full_df, feature_names, target='target'):
        """Run complete backtest with stock selection."""

        # Step 1: Select tradeable stocks
        selected = self.stock_selector.select(
            full_df, feature_names
        )

        if len(selected) == 0:
            print("\nNo tradeable stocks found")
            return {}

        # Step 2: Run backtest only on selected stocks
        print("\n" + "="*60)
        print("BACKTESTING SELECTED STOCKS ONLY")
        print("="*60)

        all_stock_results = []

        for symbol, info in selected.items():
            stock_df = full_df[
                full_df['symbol'] == symbol
            ].copy()

            min_rows = self.train_window + 30
            if len(stock_df) < min_rows:
                continue

            result = self.run_single_stock(
                stock_df,
                feature_names,
                symbol,
                selected_features=info['features'],
                target=target
            )

            if result is not None:
                all_stock_results.append(result)

        if len(all_stock_results) == 0:
            print("\nNo valid results generated")
            return {}

        self.results = pd.concat(
            all_stock_results
        ).sort_index()

        # Overall prediction metrics
        y_true = self.results['target']
        y_proba = self.results['prediction']

        self.performance = {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'accuracy': accuracy_score(
                y_true, y_proba > 0.5
            ),
            'precision': precision_score(
                y_true, y_proba > 0.5,
                zero_division=0
            ),
        }

        # Per stock results
        print("\n" + "="*60)
        print("PER STOCK RESULTS (SELECTED STOCKS ONLY)")
        print("="*60)

        traded_stocks = self.results['stock'].unique()

        for symbol in traded_stocks:
            sm = self.results['stock'] == symbol

            if sm.sum() == 0:
                continue

            s_true = self.results.loc[sm, 'target']
            s_proba = self.results.loc[sm, 'prediction']
            s_ret = self.results.loc[sm, 'strategy_return']

            if len(s_true.unique()) < 2:
                continue

            s_auc = roc_auc_score(s_true, s_proba)
            s_total = (1 + s_ret).cumprod().iloc[-1] - 1

            status = "✅" if s_auc > 0.52 else "❌"
            print(
                f"   {status} {symbol:6s}"
                f" | AUC: {s_auc:.3f}"
                f" | Return: {s_total:+.1%}"
            )

        # Calculate strategy returns
        self._calculate_strategy_returns()

        # Final summary
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)

        total = len(self.results)
        auc = self.performance['roc_auc']
        acc = self.performance['accuracy']
        prec = self.performance['precision']

        print(f"Stocks traded: {len(traded_stocks)}")
        print(f"Total observations: {total}")
        print(f"Overall ROC AUC: {auc:.3f}")
        print(f"Accuracy: {acc:.1%}")
        print(f"Precision: {prec:.1%}")

        if self.strategy_results is not None:
            sr = self.strategy_results

            print(
                f"\nStrategy Total Return:"
                f" {sr['total_return']:+.1%}"
            )
            print(
                f"Strategy Annual Return:"
                f" {sr['annual_return']:+.1%}"
            )
            print(
                f"Buy & Hold Annual Return:"
                f" {sr['buyhold_annual']:+.1%}"
            )
            print(f"Strategy Sharpe Ratio: {sr['sharpe']:.2f}")
            print(f"Max Drawdown: {sr['max_drawdown']:.1%}")
            print(f"Win Rate: {sr['win_rate']:.1%}")
            print(
                f"Active Trading Days:"
                f" {sr['total_trades']}"
            )

            beat = sr['annual_return'] > sr['buyhold_annual']
            if beat:
                print("\n   ✅ STRATEGY BEATS BUY & HOLD!")
            else:
                diff = (
                    sr['buyhold_annual'] - sr['annual_return']
                )
                print(
                    f"\n   ❌ Underperforms B&H by {diff:.1%}"
                )

        print("="*60)

        return self.performance

    def _calculate_strategy_returns(self):
        """Calculate real dollar returns from signals."""

        df = self.results.copy()
        active = df[df['signal'] > 0]

        if len(active) == 0:
            self.strategy_results = None
            return

        cumulative = (1 + df['strategy_return']).cumprod()
        buyhold = (1 + df['returns']).cumprod()

        total_days = len(df)
        years = total_days / 252.0

        if years == 0:
            self.strategy_results = None
            return

        total_return = cumulative.iloc[-1] - 1
        annual = (1 + total_return) ** (1 / years) - 1

        bh_total = buyhold.iloc[-1] - 1
        bh_annual = (1 + bh_total) ** (1 / years) - 1

        daily = df['strategy_return']
        if daily.std() > 0:
            sharpe = (
                daily.mean() / daily.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        wins = len(active[active['strategy_return'] > 0])
        total_trades = len(active)
        wr = wins / total_trades if total_trades > 0 else 0

        self.strategy_results = {
            'annual_return': annual,
            'buyhold_annual': bh_annual,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': wr,
            'total_trades': total_trades,
            'total_return': total_return,
        }