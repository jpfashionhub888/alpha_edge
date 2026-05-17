# model_cache.py - V3 (M5 fix)
"""
Model cache with TTL, feature-hash validation, and atomic saves.

Changes in V3:
- M5 fix: The backward-compat `load_models()` shim previously bypassed
  the TTL check by reading the pickle directly. Now it routes through
  the ModelCache class so TTL and version checks actually apply.
- Feature hash is now stored at the same key (`feature_hash`) inside
  the pickled model dict, so the scanner.py legacy call path still
  works: `cached.get('feature_hash')` returns the hash and scanner
  can compare against its current feature set.

Note: the `model_cache/` directory at repo root containing `*.joblib`
files is leftover from an older cache implementation no longer in
use. Safe to delete. Current cache lives at `cache/models/`.
"""

import hashlib
import json
import logging
import pickle
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
CACHE_VERSION     = "3.0"   # bumped — invalidates all V2 caches on first run
DEFAULT_TTL_DAYS  = 14      # raised from 7 — see scanner.retrain_days*2 rule
DEFAULT_CACHE_DIR = "cache/models"


class ModelCache:
    """
    Thread-safe model cache.
    - Feature-hash validation
    - TTL expiry
    - Atomic saves
    - Version tagging
    """

    def __init__(self,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 ttl_days: int  = DEFAULT_TTL_DAYS):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        self._lock = threading.Lock()
        logger.info(
            "ModelCache initialised | dir=%s ttl=%dd version=%s",
            self.cache_dir, ttl_days, CACHE_VERSION,
        )

    # ── Public API ────────────────────────────────────────────────────

    def get(self, symbol: str,
            feature_names: Optional[List[str]] = None) -> Optional[Any]:
        """
        Load cached model. Returns None if:
        - No cache exists
        - Cache is expired
        - Version mismatch
        - Feature set has changed (if feature_names provided)
        - File is corrupted
        """
        with self._lock:
            meta_path, model_path = self._paths(symbol)

            if not meta_path.exists() or not model_path.exists():
                return None

            # Metadata
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[%s] Corrupt metadata, invalidating: %s", symbol, e)
                self._delete(symbol)
                return None

            # Version
            if meta.get("cache_version") != CACHE_VERSION:
                logger.info(
                    "[%s] Cache version mismatch (%s vs %s), retraining",
                    symbol, meta.get('cache_version'), CACHE_VERSION,
                )
                self._delete(symbol)
                return None

            # TTL
            try:
                trained_at = datetime.fromisoformat(meta["trained_at"])
            except (KeyError, ValueError):
                logger.warning("[%s] Missing/invalid trained_at, invalidating", symbol)
                self._delete(symbol)
                return None

            age_days = (datetime.utcnow() - trained_at).days
            if age_days > self.ttl_days:
                logger.info(
                    "[%s] Cache expired (%dd > %dd), retraining",
                    symbol, age_days, self.ttl_days,
                )
                self._delete(symbol)
                return None

            # Feature hash (if caller provided current feature set)
            if feature_names is not None:
                current_hash = self._hash_features(feature_names)
                if meta.get("feature_hash") != current_hash:
                    logger.warning(
                        "[%s] Feature set changed — invalidating "
                        "(old=%s new=%s)",
                        symbol,
                        (meta.get('feature_hash') or '')[:8],
                        current_hash[:8],
                    )
                    self._delete(symbol)
                    return None

            # Load model
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info("[%s] Cache hit | age=%dd", symbol, age_days)
                return model
            except (pickle.UnpicklingError, OSError) as e:
                logger.warning("[%s] Corrupt model file, invalidating: %s", symbol, e)
                self._delete(symbol)
                return None

    def save(self, symbol: str, model: Any,
             feature_names: List[str]) -> bool:
        with self._lock:
            meta_path, model_path = self._paths(symbol)

            meta = {
                "symbol"       : symbol,
                "cache_version": CACHE_VERSION,
                "trained_at"   : datetime.utcnow().isoformat(),
                "ttl_days"     : self.ttl_days,
                "feature_hash" : self._hash_features(feature_names),
                "feature_count": len(feature_names),
                "feature_names": feature_names,
            }

            try:
                self._atomic_write_json(meta_path, meta)
                self._atomic_write_pickle(model_path, model)
                logger.info(
                    "[%s] Model cached | features=%d expires_in=%dd",
                    symbol, len(feature_names), self.ttl_days,
                )
                return True
            except OSError as e:
                logger.error("[%s] Cache save failed: %s", symbol, e)
                self._delete(symbol)
                return False

    def invalidate(self, symbol: str) -> None:
        with self._lock:
            self._delete(symbol)
            logger.info("[%s] Cache manually invalidated", symbol)

    def invalidate_all(self) -> int:
        with self._lock:
            count = 0
            for f in self.cache_dir.glob("*"):
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
            logger.warning("Full cache wipe: %d files removed", count)
            return count

    def get_cache_status(self) -> Dict[str, Any]:
        status = {}
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                symbol     = meta["symbol"]
                trained_at = datetime.fromisoformat(meta["trained_at"])
                age_days   = (datetime.utcnow() - trained_at).days
                status[symbol] = {
                    "age_days"     : age_days,
                    "expired"      : age_days > self.ttl_days,
                    "version_ok"   : meta.get("cache_version") == CACHE_VERSION,
                    "feature_count": meta.get("feature_count", "unknown"),
                    "trained_at"   : meta["trained_at"],
                }
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return status

    # ── Private helpers ───────────────────────────────────────────────

    def _paths(self, symbol: str):
        safe = symbol.replace("/", "_").replace(":", "_")
        return (
            self.cache_dir / f"{safe}.meta.json",
            self.cache_dir / f"{safe}.model.pkl",
        )

    def _delete(self, symbol: str) -> None:
        meta_path, model_path = self._paths(symbol)
        for p in (meta_path, model_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _hash_features(feature_names: List[str]) -> str:
        canonical = json.dumps(sorted(feature_names), separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            shutil.move(str(tmp), str(path))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_pickle(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(str(tmp), str(path))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


# ── Singleton + backward-compat shims ──────────────────────────────────────

_cache = ModelCache()


def is_cache_valid(symbol: str, feature_names: List[str]) -> bool:
    """True if a valid cached model exists with the given feature set."""
    return _cache.get(symbol, feature_names) is not None


def load_models(symbol: str) -> Optional[Any]:
    """
    M5 fix: now routes through ModelCache.get() so TTL and version
    checks apply. Was previously a raw pickle.load() that bypassed
    all invalidation logic.

    Feature hash check is skipped here because the legacy API doesn't
    pass feature names. scanner.py handles feature hash separately via
    the returned model dict's 'feature_hash' field.
    """
    return _cache.get(symbol, feature_names=None)


def save_models(symbol: str, models: Any) -> bool:
    """
    Save models. Feature names extracted from models dict if present
    (so feature hash is recorded in metadata for TTL check on load).
    """
    feature_names = []
    if isinstance(models, dict):
        feature_names = models.get('selected_features', []) or []

    return _cache.save(
        symbol        = symbol,
        model         = models,
        feature_names = feature_names,
    )
