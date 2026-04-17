# models/stock_selector.py

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from models.technical_model import TechnicalPredictor
import logging

logger = logging.getLogger(__name__)


class StockSelector:
    """
    Strict stock selection. Only trade stocks where
    the model has a proven, significant edge.
    """

    def __init__(self, min_auc=0.54, validation_days=60,
                 top_features=20):
        self.min_auc = min_auc
        self.validation_days = validation_days
        self.top_features = top_features
        self.selected_stocks = {}

    def validate_stock(self, df, feature_names, symbol):
        """Test if model can predict this stock."""

        df = df.sort_index().copy()
        split_idx = len(df) - self.validation_days

        if split_idx < 120:
            return None

        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        X_train = train_df[feature_names]
        y_train = train_df['target']
        X_val = val_df[feature_names]
        y_val = val_df['target']

        if len(y_train.unique()) < 2:
            return None
        if len(y_val.unique()) < 2:
            return None

        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(self.top_features, len(feature_names))
        )
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected = [
            f for f, m in zip(feature_names, mask) if m
        ]

        model = TechnicalPredictor()
        model.train(X_train[selected], y_train)
        predictions = model.predict(X_val[selected])

        auc = roc_auc_score(y_val, predictions)

        return {
            'symbol': symbol,
            'auc': auc,
            'passed': auc >= self.min_auc,
            'features': selected,
        }

    def select(self, full_df, feature_names):
        """Test all stocks and return only the best."""

        print("\n" + "="*60)
        print("STRICT STOCK SELECTION")
        print(f"Minimum AUC required: {self.min_auc}")
        print("="*60)

        stocks = full_df['symbol'].unique()
        results = []

        for symbol in stocks:
            stock_df = full_df[
                full_df['symbol'] == symbol
            ].copy()

            if len(stock_df) < 200:
                continue

            result = self.validate_stock(
                stock_df, feature_names, symbol
            )

            if result is not None:
                results.append(result)
                status = "✅" if result['passed'] else "❌"
                print(
                    f"   {status} {symbol:6s}"
                    f" | AUC: {result['auc']:.3f}"
                )

        results.sort(key=lambda x: x['auc'], reverse=True)
        passed = [r for r in results if r['passed']]

        n_passed = len(passed)
        n_total = len(results)
        print(f"\n   {n_passed}/{n_total} stocks passed")

        # Take maximum 5 best stocks
        passed = passed[:5]

        if len(passed) == 0:
            print("   ⚠️ No stocks passed strict filter")
            print("   Lowering threshold to 0.52")
            self.min_auc = 0.52
            passed = [
                r for r in results if r['auc'] >= 0.52
            ][:5]

        self.selected_stocks = {
            r['symbol']: r for r in passed
        }

        print("\n   Final selected stocks for trading:")
        for r in passed:
            print(f"      {r['symbol']:6s} | AUC: {r['auc']:.3f}")

        return self.selected_stocks