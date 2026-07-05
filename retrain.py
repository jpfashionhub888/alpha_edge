# retrain.py
"""
Weekly Model Retraining Script

Clears the model cache and retrains all models on fresh data.
Run every Sunday at 11 PM ET via cron:
  0 23 * * 0 cd /root/alpha_edge && /root/alpha_edge/venv/bin/python retrain.py >> /root/alpha_edge/logs/retrain.log 2>&1

Or run manually:
  python retrain.py
"""

import os
import time
import logging
import shutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

CACHE_DIRS = [
    'model_cache',
    'cache/models',
]
WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX',
    'SPY', 'QQQ', 'IWM',
    'JPM', 'V', 'GS', 'BAC',
    'JNJ', 'PFE', 'UNH', 'LLY',
    'WMT', 'COST', 'HD',
    'XOM', 'CVX',
    'CRM', 'SNOW', 'NET', 'CRWD',
    'SOFI', 'PLTR',
]
# Run HyperOpt every N retrains (0 = every time, 4 = every 4 weeks)
HYPEROPT_EVERY_N_RETRAINS = 4
# Minimum MetaLabeler training samples needed
META_MIN_SAMPLES = 60

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
CACHE_DIRS = [
    'model_cache',
    'cache/models',
]
WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX',
    'SPY', 'QQQ', 'IWM',
    'JPM', 'V', 'GS', 'BAC',
    'JNJ', 'PFE', 'UNH', 'LLY',
    'WMT', 'COST', 'HD',
    'XOM', 'CVX',
    'CRM', 'SNOW', 'NET', 'CRWD',
    'SOFI', 'PLTR',
]
# ─────────────────────────────────────────────────────────────────────


def clear_cache():
    """Delete all cached models to force fresh training."""
    cleared = 0
    for cache_dir in CACHE_DIRS:
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.endswith(('.joblib', '.pkl', '.json')):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                        cleared += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {f}: {e}")
    logger.info(f"Cache cleared: {cleared} files deleted")
    return cleared


def retrain_all():
    """Fetch fresh data and retrain all models."""
    start = time.time()

    print("\n" + "🔄" * 25)
    print(f"ALPHAEDGE WEEKLY RETRAIN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔄" * 25)

    # ── Step 1: Clear cache ───────────────────────────────────────
    print("\n── 1. Clearing model cache ───────────────────────────────")
    cleared = clear_cache()
    print(f"  Cleared {cleared} cached model files")

    # ── Step 2: Fetch fresh data ──────────────────────────────────
    print("\n── 2. Fetching fresh data ────────────────────────────────")
    try:
        from data.stock_data import StockDataFetcher
        fetcher  = StockDataFetcher(watchlist=WATCHLIST, lookback_days=730)
        all_data = fetcher.fetch_all()
        print(f"  Fetched {len(all_data)} symbols")
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return False

    # ── Step 3: Retrain models ────────────────────────────────────
    print("\n── 3. Retraining models ──────────────────────────────────")
    try:
        from data.feature_engine    import FeatureEngine
        from models.technical_model import TechnicalPredictor
        from models.regime_detector import RegimeDetector
        from model_cache            import save_models
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        import pandas as pd

        engine   = FeatureEngine()
        detector = RegimeDetector()
        success  = 0
        failed   = 0

        for symbol, raw_df in all_data.items():
            try:
                df            = engine.add_all_features(raw_df)
                feature_names = engine.get_feature_names()
                df            = detector.detect(df)

                if len(df) < 100:
                    continue

                split   = len(df) - 30
                train   = df.iloc[max(0, split-180):split]
                if len(train) < 100:
                    train = df.iloc[:split]

                X_train = train[feature_names]
                y_train = train['target']

                if len(y_train.unique()) < 2:
                    continue
                if y_train.value_counts().min() < 10:
                    continue

                # Feature selection
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(20, len(feature_names))
                )
                selector.fit(X_train, y_train)
                selected = [f for f, m in zip(feature_names, selector.get_support()) if m]

                # Train
                model = TechnicalPredictor(use_lstm=False)
                model.train(X_train[selected], y_train)

                # Save to cache
                save_models(symbol, {
                    'xgboost'          : model.models.get('xgboost'),
                    'lightgbm'         : model.models.get('lightgbm'),
                    'random_forest'    : model.models.get('random_forest'),
                    'catboost'         : model.models.get('catboost'),
                    'selected_features': selected,
                })
                success += 1
                print(f"  ✅ {symbol} retrained")

            except Exception as e:
                logger.warning(f"  ❌ {symbol} failed: {e}")
                failed += 1

        print(f"\n  Retrained: {success} models | Failed: {failed}")

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return False

    # ── Step 3b: Train MetaLabelers ───────────────────────────────
    print("\n── 3b. Training Meta-Labelers ──────────────────────────")
    try:
        from models.meta_labeler import MetaLabeler
        from models.technical_model import TechnicalPredictor
        from model_cache import ModelCache, save_models
        import pandas as pd, numpy as np

        meta_cache    = ModelCache()
        meta_success  = 0
        meta_skipped  = 0

        for symbol, raw_df in all_data.items():
            try:
                # Re-engineer features (reuse engine from step 3)
                df_full = engine.add_all_features(raw_df)
                df_full = detector.detect(df_full)

                # Load primary model from cache to get probabilities
                from model_cache import load_models
                cached = load_models(symbol)
                if not cached:
                    meta_skipped += 1
                    continue

                feat  = cached.get('selected_features', [])
                if not feat or len(df_full) < META_MIN_SAMPLES + 30:
                    meta_skipped += 1
                    continue

                # Reconstruct primary model
                primary = TechnicalPredictor(use_lstm=False)
                primary.models = {
                    'xgboost'      : cached.get('xgboost'),
                    'lightgbm'     : cached.get('lightgbm'),
                    'random_forest': cached.get('random_forest'),
                    'catboost'     : cached.get('catboost'),
                }
                primary.feature_names = feat
                primary.trained = True

                # Use last META_MIN_SAMPLES+30 rows for meta training
                meta_window = df_full.iloc[-(META_MIN_SAMPLES + 30):].copy()
                avail_feat  = [f for f in feat if f in meta_window.columns]
                if len(avail_feat) < 5:
                    meta_skipped += 1
                    continue

                X_meta = meta_window[avail_feat]
                probs  = pd.Series(
                    primary.predict(X_meta), index=meta_window.index
                )

                # y_meta = 1 if primary model was right (price went up AND pred > 0.5)
                future_return = meta_window['close'].pct_change(1).shift(-1)
                y_meta = ((probs > 0.5) == (future_return > 0)).astype(int)
                y_meta = y_meta.dropna()
                X_meta = X_meta.loc[y_meta.index]
                probs  = probs.loc[y_meta.index]

                if len(y_meta) < META_MIN_SAMPLES or y_meta.nunique() < 2:
                    meta_skipped += 1
                    continue

                labeler = MetaLabeler(threshold=0.55)
                labeler.fit(X_meta, y_meta, probs)
                meta_cache.save_meta_labeler(symbol, labeler)
                meta_success += 1
                print(f'  ✅ {symbol} MetaLabeler trained')

            except Exception as e:
                logger.warning(f'  MetaLabeler failed for {symbol}: {e}')
                meta_skipped += 1

        print(f'\n  MetaLabelers: {meta_success} trained | {meta_skipped} skipped')

    except ImportError as e:
        logger.warning(f'MetaLabeler not available: {e}')

    # ── Step 4: Send Telegram notification ───────────────────────
    print("\n── 4. Sending notification ─────────────────────────────")
    try:
        from monitoring.telegram_bot import TelegramBot
        elapsed = time.time() - start
        bot = TelegramBot()
        bot.send_message(
            f"🔄 Weekly Retrain Complete\n"
            f"✅ {success} primary models retrained\n"
            f"🧠 {meta_success} MetaLabelers trained\n"
            f"❌ {failed} failed\n"
            f"⏱ {elapsed/60:.1f} minutes\n"
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    except Exception as e:
        logger.warning(f'Telegram notification failed: {e}')

    elapsed = time.time() - start
    print(f'\n✅ Retrain complete in {elapsed/60:.1f} minutes')
    print(f'   Models + MetaLabelers saved — ready for Monday scan\n')
    return True


if __name__ == '__main__':
    retrain_all()
