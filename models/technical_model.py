# models/technical_model.py

"""
Multi-model ensemble predictor.
XGBoost + LightGBM + Random Forest + CatBoost + LSTM
Five models vote together for maximum accuracy.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from models.lstm_model import LSTMPredictor
import logging

logger = logging.getLogger(__name__)

# Overfit thresholds (kept in sync with model_validator.py)
OVERFIT_WARN_GAP  = 0.10
OVERFIT_HARD_STOP = 0.20


class TechnicalPredictor:
    """
    5-model ensemble predictor.
    XGBoost + LightGBM + Random Forest + CatBoost + LSTM
    All parallel processing disabled for Python 3.14
    compatibility.
    """

    # Maximum features fed to tree models.
    # Rule of thumb: n_samples / 8-10.
    # With ~180 samples, 20 features keeps p/n ≈ 0.14 — safe for tree ensembles.
    MAX_FEATURES = 20

    def __init__(self, use_lstm=True):
        self.models           = {}
        self.feature_names    = []
        self.selected_features= []   # post-selection feature names
        self.feature_selector = None  # fitted SelectKBest
        self.trained          = False
        self.use_lstm         = use_lstm
        # Set after train() — AUC on last 20% of training data
        self.train_auc    : float = 0.5
        self.val_auc      : float = 0.5
        self.overfit_gap  : float = 0.0
        self.overfit_flagged: bool = False

    def train(self, X, y):
        """Train all models with built-in overfitting guard."""

        logger.info(
            f"Training ensemble on {len(X)} samples"
        )

        self.feature_names = list(X.columns)

        # ── Temporal validation split (last 20% = val set) ────────────────────
        # Must be TEMPORAL not random — future-leak otherwise
        val_size  = max(30, int(len(X) * 0.20))
        X_val_raw = X.iloc[-val_size:]
        y_val_raw = y.iloc[-val_size:]
        X_tr      = X.iloc[:-val_size]
        y_tr      = y.iloc[:-val_size]

        # Fall back to full data if val split leaves too few train rows
        if len(X_tr) < 50:
            X_tr, y_tr = X, y
            X_val_raw, y_val_raw = X, y

        # ── Feature selection (p >> n guard) ──────────────────────────────────
        # With 167 features on ~144 training rows, p/n ≈ 1.16 — guaranteed overfit.
        # Reduce to MAX_FEATURES using mutual information so p/n < 0.15.
        # Selector is fitted on train only (no data leak).
        n_features = X_tr.shape[1]
        k = min(self.MAX_FEATURES, n_features)
        if n_features > k:
            try:
                selector = SelectKBest(mutual_info_classif, k=k)
                selector.fit(X_tr, y_tr)
                selected_mask   = selector.get_support()
                selected_cols   = [c for c, s in zip(X_tr.columns, selected_mask) if s]
                X_tr            = X_tr[selected_cols]
                X_val_raw       = X_val_raw[selected_cols]
                self.feature_selector  = selector
                self.selected_features = selected_cols
                logger.info(
                    f'Feature selection: {n_features} → {k} features '
                    f'(p/n ratio: {n_features/max(len(X_tr),1):.2f} → {k/max(len(X_tr),1):.2f})'
                )
            except Exception as e:
                logger.warning(f'Feature selection failed ({e}) — using all {n_features} features')
                self.feature_selector  = None
                self.selected_features = list(X_tr.columns)
        else:
            self.selected_features = list(X_tr.columns)

        min_class = y_tr.value_counts().min()
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

            self.models['xgboost'].fit(X_tr, y_tr)

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

            self.models['lightgbm'].fit(X_tr, y_tr)

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

            self.models['random_forest'].fit(X_tr, y_tr)

        except Exception as e:
            logger.warning(
                f"Random Forest training failed: {e}"
            )
        # ==========================================
        # MODEL 4: CatBoost
        # Excellent with categorical + numerical features
        # ==========================================
        try:
            cat_model = CatBoostClassifier(
                iterations    = 200,
                depth         = 4,
                learning_rate = 0.01,
                random_seed   = 42,
                verbose       = 0,
                thread_count  = 1,
            )
            cat_model.fit(X_tr, y_tr)
            self.models['catboost'] = cat_model
            logger.info("CatBoost model trained successfully")
        except Exception as e:
            logger.warning(f"CatBoost training failed: {e}")

        # ==========================================
        # MODEL 5: LSTM Neural Network
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

        # ── Overfitting guard ─────────────────────────────────────────────────
        # Evaluate on BOTH the training tail and the held-out validation set
        # to catch models that memorise rather than generalise.
        try:
            train_preds = self.predict(X_tr)
            val_preds   = self.predict(X_val_raw)

            if y_tr.nunique() >= 2:
                self.train_auc = float(roc_auc_score(y_tr, train_preds))
            if y_val_raw.nunique() >= 2:
                self.val_auc   = float(roc_auc_score(y_val_raw, val_preds))

            self.overfit_gap = self.train_auc - self.val_auc

            if self.overfit_gap > OVERFIT_HARD_STOP:
                self.overfit_flagged = True
                logger.error(
                    f'OVERFIT HARD STOP: train_AUC={self.train_auc:.3f} '
                    f'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f} '
                    f'— this fold will NOT generate live signals'
                )
            elif self.overfit_gap > OVERFIT_WARN_GAP:
                logger.warning(
                    f'Overfit warning: train_AUC={self.train_auc:.3f} '
                    f'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}'
                )
            else:
                logger.info(
                    f'Overfit check OK: train_AUC={self.train_auc:.3f} '
                    f'val_AUC={self.val_auc:.3f} gap={self.overfit_gap:.3f}'
                )
        except Exception as e:
            logger.debug(f'Overfit check failed (non-critical): {e}')

        return self

    def _apply_feature_selection(self, X):
        """Apply the same feature selection used during training."""
        if self.selected_features and isinstance(X, pd.DataFrame):
            # Keep only the columns the model was trained on
            available = [c for c in self.selected_features if c in X.columns]
            if len(available) == len(self.selected_features):
                return X[self.selected_features]
            elif len(available) > 0:
                logger.warning(
                    f'Feature mismatch: expected {len(self.selected_features)}, '
                    f'found {len(available)} — using available subset'
                )
                return X[available]
        return X

    def predict(self, X):
        """
        Average predictions from all models.
        Weighted ensemble: tree models get full weight,
        LSTM gets 0.8 weight.
        """

        if not self.trained:
            raise Exception("Model not trained yet")

        X = self._apply_feature_selection(X)

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

        X = self._apply_feature_selection(X)

        all_predictions = []

        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    proba = model.predict(X)
                else:
                    proba = model.predict_proba(X)[:, 1]

                if len(proba) > 0:
                    all_predictions.append(proba)

            except Exception as e:
                logger.debug(f'Model {name} agreement prediction failed: {e}')

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

            except Exception as e:
                logger.debug(f'Feature importance failed for {name}: {e}')

        if len(all_importances) == 0:
            return []

        importance = np.mean(all_importances, axis=0)

        return sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )