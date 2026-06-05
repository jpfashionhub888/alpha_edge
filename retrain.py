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

    # ── Step 4: Send Telegram notification ───────────────────────
    print("\n── 4. Sending notification ───────────────────────────────")
    try:
        from monitoring.telegram_bot import TelegramBot
        elapsed = time.time() - start
        bot = TelegramBot()
        bot.send_message(
            f"🔄 Weekly Retrain Complete\n"
            f"✅ {success} models retrained\n"
            f"❌ {failed} failed\n"
            f"⏱ {elapsed/60:.1f} minutes\n"
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")

    elapsed = time.time() - start
    print(f"\n✅ Retrain complete in {elapsed/60:.1f} minutes")
    print(f"   Models saved to cache — ready for Monday scan\n")
    return True


if __name__ == '__main__':
    retrain_all()
