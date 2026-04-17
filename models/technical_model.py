# models/technical_model.py

"""
Multi-model ensemble predictor.
XGBoost + LightGBM + Random Forest + LSTM
Four models vote together for maximum accuracy.
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from models.lstm_model import LSTMPredictor
import logging

logger = logging.getLogger(__name__)


class TechnicalPredictor:
    """
    4-model ensemble predictor.
    XGBoost + LightGBM + Random Forest + LSTM
    All parallel processing disabled for Python 3.14
    compatibility.
    """

    def __init__(self, use_lstm=True):
        self.models = {}
        self.feature_names = []
        self.trained = False
        self.use_lstm = use_lstm

    def train(self, X, y):
        """Train all four models."""

        logger.info(
            f"Training ensemble on {len(X)} samples"
        )

        self.feature_names = list(X.columns)

        min_class = y.value_counts().min()
        n_folds = min(5, min_class)
        n_folds = max(2, n_folds)

        # ==========================================
        # MODEL 1: XGBoost
        # ==========================================
        try:
            xgb_base = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                eval_metric='logloss'
            )

            if min_class >= 5:
                self.models['xgboost'] = (
                    CalibratedClassifierCV(
                        xgb_base,
                        cv=n_folds,
                        method='isotonic'
                    )
                )
            else:
                self.models['xgboost'] = xgb_base

            self.models['xgboost'].fit(X, y)

        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")

        # ==========================================
        # MODEL 2: LightGBM
        # ==========================================
        try:
            lgb_base = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42,
                n_jobs=1,
                verbose=-1
            )

            if min_class >= 5:
                self.models['lightgbm'] = (
                    CalibratedClassifierCV(
                        lgb_base,
                        cv=n_folds,
                        method='isotonic'
                    )
                )
            else:
                self.models['lightgbm'] = lgb_base

            self.models['lightgbm'].fit(X, y)

        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")

        # ==========================================
        # MODEL 3: Random Forest
        # n_jobs=1 is critical for Python 3.14
        # parallel processing causes crashes
        # ==========================================
        try:
            rf_base = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=1
            )

            if min_class >= 5:
                self.models['random_forest'] = (
                    CalibratedClassifierCV(
                        rf_base,
                        cv=n_folds,
                        method='isotonic'
                    )
                )
            else:
                self.models['random_forest'] = rf_base

            self.models['random_forest'].fit(X, y)

        except Exception as e:
            logger.warning(
                f"Random Forest training failed: {e}"
            )

        # ==========================================
        # MODEL 4: LSTM Neural Network
        # ==========================================
        if self.use_lstm and len(X) >= 50:
            try:
                lstm = LSTMPredictor(
                    sequence_length=20,
                    hidden_size=64,
                    num_layers=2,
                    epochs=30,
                    learning_rate=0.001,
                    batch_size=32
                )
                lstm.train(X, y)

                if lstm.trained:
                    self.models['lstm'] = lstm
                    logger.info(
                        "LSTM model added to ensemble"
                    )
                else:
                    logger.info(
                        "LSTM skipped (not enough data)"
                    )

            except Exception as e:
                logger.warning(
                    f"LSTM training failed: {e}"
                )
        else:
            logger.info(
                "LSTM skipped (data too small)"
            )

        self.trained = True

        n_models = len(self.models)
        logger.info(
            f"Ensemble training complete ({n_models} models)"
        )

        return self

    def predict(self, X):
        """
        Average predictions from all models.
        Weighted ensemble: tree models get full weight,
        LSTM gets 0.8 weight.
        """

        if not self.trained:
            raise Exception("Model not trained yet")

        all_predictions = []
        weights = []

        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    proba = model.predict(X)
                    if len(proba) > 0:
                        all_predictions.append(proba)
                        weights.append(0.8)
                else:
                    proba = model.predict_proba(X)[:, 1]
                    all_predictions.append(proba)
                    weights.append(1.0)

            except Exception as e:
                logger.warning(
                    f"Model {name} prediction failed: {e}"
                )

        if len(all_predictions) == 0:
            return np.full(len(X), 0.5)

        total_weight = sum(weights)
        ensemble_pred = np.zeros(len(X))

        for pred, weight in zip(all_predictions, weights):
            if len(pred) == len(X):
                ensemble_pred += pred * weight
            elif len(pred) > 0:
                ensemble_pred += (
                    pred[-len(X):] * weight
                )

        ensemble_pred /= total_weight

        return ensemble_pred

    def predict_with_agreement(self, X):
        """
        Return prediction AND how many models agree.
        Higher agreement = higher confidence signal.
        """

        if not self.trained:
            raise Exception("Model not trained yet")

        all_predictions = []

        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    proba = model.predict(X)
                else:
                    proba = model.predict_proba(X)[:, 1]

                if len(proba) > 0:
                    all_predictions.append(proba)

            except Exception:
                pass

        if len(all_predictions) == 0:
            return np.full(len(X), 0.5), 0

        ensemble_pred = np.mean(
            all_predictions, axis=0
        )

        bullish_count = sum(
            1 for p in all_predictions
            if np.mean(p) > 0.5
        )
        agreement = bullish_count / len(all_predictions)

        return ensemble_pred, agreement

    def get_feature_importance(self):
        """
        Return averaged feature importance
        across all tree models.
        LSTM is excluded (no feature importance).
        """

        if not self.trained:
            return []

        all_importances = []

        for name, model in self.models.items():

            if name == 'lstm':
                continue

            try:
                if hasattr(
                    model, 'calibrated_classifiers_'
                ):
                    for cal in (
                        model.calibrated_classifiers_
                    ):
                        est = cal.estimator
                        if hasattr(
                            est, 'feature_importances_'
                        ):
                            all_importances.append(
                                est.feature_importances_
                            )

                elif hasattr(
                    model, 'feature_importances_'
                ):
                    all_importances.append(
                        model.feature_importances_
                    )

            except Exception:
                pass

        if len(all_importances) == 0:
            return []

        importance = np.mean(all_importances, axis=0)

        return sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )