# model_cache.py
# ALPHAEDGE - Model Cache System
# Saves trained models to disk
# Loads instantly on next run
# Auto-retrains every 30 days

import os
import json
import joblib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_DIR      = 'model_cache'
CACHE_INFO     = 'model_cache/cache_info.json'
RETRAIN_DAYS   = 30  # Retrain every 30 days


def get_cache_path(symbol, model_name):
    """Get file path for a cached model."""
    safe_symbol = symbol.replace('/', '_').replace('.', '_')
    return os.path.join(
        CACHE_DIR,
        f"{safe_symbol}_{model_name}.joblib"
    )


def load_cache_info():
    """Load cache metadata."""
    if not os.path.exists(CACHE_INFO):
        return {}
    try:
        with open(CACHE_INFO, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache_info(info):
    """Save cache metadata."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_INFO, 'w') as f:
        json.dump(info, f, indent=2)


def is_cache_valid(symbol):
    """Check if cached models are still fresh."""
    info = load_cache_info()

    if symbol not in info:
        return False

    cached_date = info[symbol].get('trained_at')
    if not cached_date:
        return False

    trained_at  = datetime.fromisoformat(cached_date)
    expiry      = trained_at + timedelta(days=RETRAIN_DAYS)

    if datetime.now() > expiry:
        logger.info(f"Cache expired for {symbol}")
        return False

    # Check all model files exist
    for model_name in ['xgboost', 'lightgbm', 'random_forest', 'catboost']:
        path = get_cache_path(symbol, model_name)
        if not os.path.exists(path):
            return False

    return True


def save_models(symbol, models_dict):
    """
    Save trained models to disk.

    models_dict should contain:
    {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'rf':  rf_model,
        'scaler': scaler,
        'selector': selector,
        'selected_features': [...],
    }
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name, obj in models_dict.items():
        if name == 'selected_features':
            # Save feature list as JSON
            path = get_cache_path(symbol, 'features')
            with open(path, 'w') as f:
                json.dump(obj, f)
        else:
            path = get_cache_path(symbol, name)
            try:
                joblib.dump(obj, path)
            except Exception as e:
                logger.warning(f"Could not save {name} for {symbol}: {e}")

    # Update cache info
    info = load_cache_info()
    info[symbol] = {
        'trained_at' : datetime.now().isoformat(),
        'expires_at' : (
            datetime.now() + timedelta(days=RETRAIN_DAYS)
        ).isoformat(),
    }
    save_cache_info(info)
    logger.info(f"Models cached for {symbol}")


def load_models(symbol):
    """
    Load cached models from disk.
    Returns None if cache is invalid.
    """
    if not is_cache_valid(symbol):
        return None

    try:
        models = {}

        model_names = ['xgboost', 'lightgbm', 'random_forest', 'catboost']
        for model_name in model_names:
            path = get_cache_path(symbol, model_name)
            if os.path.exists(path):
                models[model_name] = joblib.load(path)
            else:
                logger.warning(f"Cache file missing: {path}")
                return None

        # Load feature list
        feat_path = get_cache_path(symbol, 'features')
        if os.path.exists(feat_path):
            with open(feat_path, 'r') as f:
                models['selected_features'] = json.load(f)

        logger.info(f"Models loaded from cache for {symbol}")
        return models

    except Exception as e:
        logger.warning(f"Cache load failed for {symbol}: {e}")
        return None


def clear_cache(symbol=None):
    """Clear cache for one symbol or all symbols."""
    if symbol:
        for name in ['xgb', 'lgb', 'rf', 'scaler', 'selector', 'features']:
            path = get_cache_path(symbol, name)
            if os.path.exists(path):
                os.remove(path)
        info = load_cache_info()
        info.pop(symbol, None)
        save_cache_info(info)
        print(f"Cache cleared for {symbol}")
    else:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        print("All cache cleared")


def get_cache_status():
    """Print cache status for all symbols."""
    info = load_cache_info()

    if not info:
        print("No cached models found")
        return

    print(f"\n{'='*55}")
    print(f"  MODEL CACHE STATUS")
    print(f"{'='*55}")
    print(f"  Cache directory: {CACHE_DIR}/")
    print(f"  Retrain interval: {RETRAIN_DAYS} days")
    print(f"  Symbols cached: {len(info)}")
    print(f"\n  {'Symbol':<20} {'Trained':<12} {'Expires':<12} Status")
    print(f"  {'-'*55}")

    now = datetime.now()
    for symbol, data in info.items():
        trained = data.get('trained_at', '')[:10]
        expires = data.get('expires_at', '')[:10]
        valid   = is_cache_valid(symbol)
        status  = "VALID" if valid else "EXPIRED"
        print(f"  {symbol:<20} {trained:<12} {expires:<12} {status}")

    print(f"{'='*55}\n")


if __name__ == '__main__':
    get_cache_status()
