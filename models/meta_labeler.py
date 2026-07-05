# models/meta_labeler.py
"""
AlphaEdge — Meta-Labeling Engine
Concept: Lopez de Prado, "Advances in Financial ML" (Chapter 3)
         Also used by: Two Sigma, AQR, Renaissance Technologies

Idea in one line:
    Train a SECOND model to predict whether the FIRST model is correct.

Architecture:
    Model 1 (Primary): Trained on raw features → predicts BUY/HOLD probability
    Model 2 (Meta):    Trained on [raw features + primary prediction + primary confidence]
                       → predicts P(primary model is correct)

Only pass a BUY signal when:
    primary_pred > BUY_THRESHOLD         (primary model confident)
    AND meta_confidence > META_THRESHOLD (meta model agrees it's a real signal)

This dramatically reduces false positives without reducing win size.

Usage:
    from models.meta_labeler import MetaLabeler

    labeler = MetaLabeler()
    labeler.fit(X_train, y_primary, primary_probs_train)
    meta_conf = labeler.predict_confidence(X_live, primary_prob_live)
    if meta_conf > 0.55:
        execute_buy()
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Threshold: only pass signals where meta model is > this confident
META_CONFIDENCE_THRESHOLD = 0.55

# Meta model uses LightGBM by default (fast, handles correlated features well)
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKL_AVAILABLE = True
except ImportError:
    _SKL_AVAILABLE = False


class MetaLabeler:
    """
    Secondary model that learns when the primary model is right.

    The key insight: primary model errors are NOT random.
    They're clustered in specific market conditions (high volatility,
    low volume, regime transitions). The meta model learns these clusters.

    Parameters
    ----------
    threshold : float
        Minimum meta confidence to pass a BUY signal. Default 0.55.
        Higher = fewer but higher-quality signals.
    model_type : str
        'lgb' (LightGBM, default) or 'logistic' (fallback).
    """

    def __init__(
        self,
        threshold:  float = META_CONFIDENCE_THRESHOLD,
        model_type: str   = 'lgb',
    ):
        self.threshold  = threshold
        self.model_type = model_type if (_LGB_AVAILABLE or model_type == 'logistic') else 'logistic'
        self._model     = None
        self._scaler    = StandardScaler() if _SKL_AVAILABLE else None
        self._is_fitted = False
        self._feature_names: list = []

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X:              pd.DataFrame,
        y_true:         pd.Series,
        primary_probs:  pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> 'MetaLabeler':
        """
        Train the meta model.

        Parameters
        ----------
        X             : Feature matrix (same features used by primary model)
        y_true        : Ground truth labels (1=up, 0=down) for the training period
        primary_probs : Primary model's predicted probabilities for X
        sample_weights: Optional. Weight recent samples higher.
        """
        if len(X) < 50:
            logger.warning(f'MetaLabeler.fit: only {len(X)} samples — skipping (need ≥50)')
            return self

        # Meta target: 1 if primary model was correct, 0 if it was wrong
        primary_pred  = (primary_probs > 0.5).astype(int)
        meta_y        = (primary_pred == y_true.values).astype(int)

        if meta_y.sum() < 10:
            logger.warning('MetaLabeler: fewer than 10 correct primary predictions — meta model will be weak')

        # Build meta feature matrix: original features + primary signal info
        X_meta = self._build_meta_features(X, primary_probs)
        self._feature_names = list(X_meta.columns)

        logger.info(
            f'MetaLabeler.fit: {len(X_meta)} samples | '
            f'{meta_y.mean():.1%} primary accuracy | '
            f'{X_meta.shape[1]} meta features'
        )

        if self.model_type == 'lgb' and _LGB_AVAILABLE:
            self._fit_lgb(X_meta, meta_y, sample_weights)
        else:
            self._fit_logistic(X_meta, meta_y, sample_weights)

        self._is_fitted = True
        return self

    def _fit_lgb(self, X_meta, meta_y, weights):
        params = {
            'objective'       : 'binary',
            'metric'          : 'binary_logloss',
            'num_leaves'      : 15,          # small = less overfit
            'learning_rate'   : 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq'    : 5,
            'min_child_samples': 10,
            'n_estimators'    : 200,
            'verbose'         : -1,
        }
        self._model = lgb.LGBMClassifier(**params)
        self._model.fit(
            X_meta, meta_y,
            sample_weight=weights.values if weights is not None else None,
        )
        logger.info('MetaLabeler: LightGBM meta model trained')

    def _fit_logistic(self, X_meta, meta_y, weights):
        if not _SKL_AVAILABLE:
            raise ImportError('scikit-learn required for logistic meta model')
        X_scaled = self._scaler.fit_transform(X_meta)
        self._model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        self._model.fit(
            X_scaled, meta_y,
            sample_weight=weights.values if weights is not None else None,
        )
        logger.info('MetaLabeler: Logistic meta model trained')

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_confidence(
        self,
        X:            pd.DataFrame,
        primary_prob: float,
    ) -> float:
        """
        Given live features and primary model probability,
        return P(primary model is correct) in [0, 1].

        Returns 0.0 (block signal) if not fitted.
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        try:
            X_meta = self._build_meta_features(
                X.iloc[[-1]] if len(X) > 1 else X,
                pd.Series([primary_prob]),
            )
            # Align columns
            for col in self._feature_names:
                if col not in X_meta.columns:
                    X_meta[col] = 0.0
            X_meta = X_meta[self._feature_names]

            if self.model_type == 'lgb' and _LGB_AVAILABLE:
                prob = self._model.predict_proba(X_meta)[0, 1]
            else:
                X_scaled = self._scaler.transform(X_meta)
                prob = self._model.predict_proba(X_scaled)[0, 1]

            return float(prob)

        except Exception as e:
            logger.warning(f'MetaLabeler.predict_confidence failed: {e}')
            return 0.0

    def should_trade(self, X: pd.DataFrame, primary_prob: float) -> Tuple[bool, float]:
        """
        Returns (trade_approved: bool, meta_confidence: float).
        Use this as the final gate before executing a BUY.
        """
        if not self._is_fitted:
            return True, 1.0   # pass-through if not fitted (safe default)

        conf = self.predict_confidence(X, primary_prob)
        approved = conf >= self.threshold

        logger.info(
            f'MetaLabeler: primary_prob={primary_prob:.3f} '
            f'meta_conf={conf:.3f} '
            f'{"✅ APPROVED" if approved else "❌ FILTERED"}'
        )
        return approved, conf

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save meta model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model'        : self._model,
                'scaler'       : self._scaler,
                'feature_names': self._feature_names,
                'threshold'    : self.threshold,
                'model_type'   : self.model_type,
                'is_fitted'    : self._is_fitted,
            }, f)
        logger.info(f'MetaLabeler saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'MetaLabeler':
        """Load meta model from disk. Returns unfitted instance on failure."""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            obj                 = cls(threshold=state['threshold'], model_type=state['model_type'])
            obj._model          = state['model']
            obj._scaler         = state['scaler']
            obj._feature_names  = state['feature_names']
            obj._is_fitted      = state['is_fitted']
            logger.info(f'MetaLabeler loaded from {path}')
            return obj
        except Exception as e:
            logger.warning(f'MetaLabeler.load failed ({e}) — returning unfitted instance')
            return cls()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_meta_features(
        X:             pd.DataFrame,
        primary_probs: pd.Series,
    ) -> pd.DataFrame:
        """
        Augment feature matrix with primary model signal info.
        The meta model sees:
          1. All original features (market context)
          2. Primary model probability (how confident was it?)
          3. Primary model binary decision (did it say BUY?)
          4. Primary probability deciles (non-linear threshold info)
        """
        X_meta = X.copy().reset_index(drop=True)
        probs  = primary_probs.reset_index(drop=True)

        X_meta['_meta_primary_prob']  = probs
        X_meta['_meta_primary_pred']  = (probs > 0.5).astype(int)
        X_meta['_meta_prob_sq']       = probs ** 2          # non-linear
        X_meta['_meta_prob_centered'] = probs - 0.5        # how far from neutral
        X_meta['_meta_high_conf']     = (probs > 0.7).astype(int)
        X_meta['_meta_low_conf']      = (probs < 0.55).astype(int)

        return X_meta

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances (LGB only)."""
        if not self._is_fitted or self.model_type != 'lgb':
            return None
        try:
            return pd.Series(
                self._model.feature_importances_,
                index=self._feature_names,
            ).sort_values(ascending=False)
        except Exception:
            return None
