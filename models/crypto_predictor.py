# models/crypto_predictor.py

import pandas as pd
import numpy as np
from data.crypto_data import CryptoDataFetcher
from data.feature_engine import FeatureEngine
from models.technical_model import TechnicalPredictor
from models.regime_detector import RegimeDetector
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class CryptoPredictor:
    """
    Prediction model for crypto markets.
    Uses daily candles for Coinbase compatibility.
    """

    def __init__(self, top_features=20):
        self.top_features = top_features
        self.models = {}
        self.features = {}
        self.regime_detector = RegimeDetector()

    def train_and_validate(self, symbol, df,
                           feature_names,
                           validation_days=60):
        """Train model for one crypto pair."""

        print(f"\n   Training model for {symbol}...")

        df = df.sort_index().copy()
        split_idx = len(df) - validation_days

        if split_idx < 100:
            print(f"   Not enough data for {symbol}")
            return None

        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]

        X_train = train[feature_names]
        y_train = train['target']
        X_val = val[feature_names]
        y_val = val['target']

        if len(y_train.unique()) < 2:
            return None
        if len(y_val.unique()) < 2:
            return None

        min_class = y_train.value_counts().min()
        if min_class < 10:
            print(f"   Not enough samples for {symbol}")
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

        print(f"   {symbol} validation AUC: {auc:.3f}")

        if auc > 0.52:
            self.models[symbol] = model
            self.features[symbol] = selected
            print(f"   ✅ {symbol} model accepted")
            return auc
        else:
            print(f"   ❌ {symbol} model rejected")
            return None

    def run_full_pipeline(self, watchlist,
                          lookback_days=365):
        """Run complete crypto pipeline."""

        print("\n" + "="*60)
        print("CRYPTO PREDICTION PIPELINE")
        print("="*60)

        fetcher = CryptoDataFetcher(
            watchlist=watchlist,
            timeframe='1d',
            lookback_days=lookback_days
        )

        if fetcher.exchange is None:
            print("   ❌ No exchange available")
            return {}

        all_data = fetcher.fetch_all()

        if len(all_data) == 0:
            print("   No crypto data fetched")
            return {}

        engine = FeatureEngine()
        results = {}

        for symbol, raw_df in all_data.items():
            try:
                df = engine.add_all_features(raw_df)
                feature_names = engine.get_feature_names()

                if len(df) < 100:
                    print(
                        f"   Skipping {symbol}:"
                        f" not enough data"
                    )
                    continue

                df = self.regime_detector.detect(df)

                auc = self.train_and_validate(
                    symbol, df, feature_names
                )

                if auc is not None:
                    selected = self.features[symbol]
                    latest = df.iloc[-1:]
                    pred = self.models[symbol].predict(
                        latest[selected]
                    )[0]
                    regime = latest['regime'].iloc[0]

                    results[symbol] = {
                        'auc': auc,
                        'prediction': pred,
                        'regime': regime,
                        'price': latest['close'].iloc[0],
                    }

            except Exception as e:
                logger.warning(
                    f"Error processing {symbol}: {e}"
                )

        return results