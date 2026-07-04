# model_cache.py - Fixed V2
# Fixes:
# 1. Feature hash validation (stale model = wrong features = silent corruption)
# 2. Model age TTL (default 7 days - configurable)
# 3. Atomic writes (crash during save = corrupted cache)
# 4. Explicit versioning field in metadata
# 5. Thread-safe file locking for concurrent scanner runs

import os
import json
import pickle
import hashlib
import logging
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# Bumped 3.1: force retrain after feature_engine macro-feature removal
# (gld_level/vix_level/dxy_level/tlt_level removed → old models crash on predict)
CACHE_VERSION = "3.1"          # Bump this manually when model architecture changes
DEFAULT_TTL_DAYS = 7           # Max age before forced retrain
DEFAULT_CACHE_DIR = "cache/models"


class ModelCache:
    """
    Thread-safe model cache with:
    - Feature-hash validation (detects feature_engine.py changes)
    - TTL expiry (no model runs forever without retraining)
    - Atomic saves (crash-safe)
    - Version tagging (architecture changes invalidate old caches)
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        self._lock = threading.Lock()

        logger.info(
            f"ModelCache initialized | dir={self.cache_dir} "
            f"ttl={ttl_days}d version={CACHE_VERSION}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(
        self,
        symbol: str,
        feature_names: List[str],
        required_trained_through: str = None,
    ) -> Optional[Any]:
        """
        Load cached model for symbol.

        Returns None if:
        - No cache exists
        - Cache is expired (> ttl_days old)
        - Feature set has changed since training
        - Cache version mismatch
        - File is corrupted
        - Cache was trained after required_trained_through (Fix 1.4: walk-forward date gate)
        """
        with self._lock:
            meta_path, model_path = self._paths(symbol)

            if not meta_path.exists() or not model_path.exists():
                logger.debug(f"[{symbol}] No cache found")
                return None

            # ── Load metadata ──────────────────────────────────────────────
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[{symbol}] Corrupt metadata, invalidating: {e}")
                self._delete(symbol)
                return None

            # ── Version check ──────────────────────────────────────────────
            if meta.get("cache_version") != CACHE_VERSION:
                logger.info(
                    f"[{symbol}] Cache version mismatch "
                    f"({meta.get('cache_version')} vs {CACHE_VERSION}), retraining"
                )
                self._delete(symbol)
                return None

            # ── TTL check ─────────────────────────────────────────────────
            trained_at = datetime.fromisoformat(meta["trained_at"])
            age_days = (datetime.utcnow() - trained_at).days
            if age_days > self.ttl_days:
                logger.info(
                    f"[{symbol}] Cache expired ({age_days}d > {self.ttl_days}d), retraining"
                )
                self._delete(symbol)
                return None

            # ── Fix 1.4: Walk-forward date gate ────────────────────────────
            # Reject cache if it was trained AFTER the required cutoff date.
            # This prevents a model trained on 2024 data from being loaded
            # in a walk-forward window that ends in 2023 (data leakage).
            if required_trained_through:
                cache_through = meta.get('trained_through', '1900-01-01')
                if cache_through > required_trained_through:
                    logger.warning(
                        f"[{symbol}] Cache trained through {cache_through} "
                        f"but window ends {required_trained_through} — REJECTING (Fix 1.4)"
                    )
                    return None

            # ── Feature hash check ───────────────────────────────────────
            current_hash = self._hash_features(feature_names)
            if meta.get("feature_hash") != current_hash:
                logger.warning(
                    f"[{symbol}] Feature set changed since training — "
                    f"cached model is INVALID, forcing retrain. "
                    f"Old hash={meta.get('feature_hash')[:8]} "
                    f"New hash={current_hash[:8]}"
                )
                self._delete(symbol)
                return None

            # ── Load model ────────────────────────────────────────────────
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info(
                    f"[{symbol}] Cache hit | age={age_days}d "
                    f"features={len(feature_names)}"
                )
                return model
            except (pickle.UnpicklingError, OSError) as e:
                logger.warning(f"[{symbol}] Corrupt model file, invalidating: {e}")
                self._delete(symbol)
                return None

    def save(
        self,
        symbol: str,
        model: Any,
        feature_names: List[str],
        trained_through: str = None,
    ) -> bool:
        """
        Atomically save model + metadata.

        trained_through: ISO date of last training bar (Fix 1.4).
          e.g. '2024-06-15' — used by walk-forward backtester to
          reject caches that were trained on future data.

        Uses temp file + rename so a crash during save
        never leaves a half-written corrupted cache.
        """
        with self._lock:
            meta_path, model_path = self._paths(symbol)

            meta = {
                "symbol": symbol,
                "cache_version": CACHE_VERSION,
                "trained_at": datetime.utcnow().isoformat(),
                "trained_through": trained_through or datetime.utcnow().strftime('%Y-%m-%d'),
                "ttl_days": self.ttl_days,
                "feature_hash": self._hash_features(feature_names),
                "feature_count": len(feature_names),
                "feature_names": feature_names,  # stored for debugging
            }

            try:
                # ── Atomic metadata write ──────────────────────────────
                self._atomic_write_json(meta_path, meta)

                # ── Atomic model write ─────────────────────────────────
                self._atomic_write_pickle(model_path, model)

                logger.info(
                    f"[{symbol}] Model cached | "
                    f"features={len(feature_names)} "
                    f"expires_in={self.ttl_days}d"
                )
                return True

            except OSError as e:
                logger.error(f"[{symbol}] Cache save failed: {e}")
                # Clean up partial writes
                self._delete(symbol)
                return False

    def invalidate(self, symbol: str) -> None:
        """Manually invalidate cache for a symbol (e.g., after regime change)."""
        with self._lock:
            self._delete(symbol)
            logger.info(f"[{symbol}] Cache manually invalidated")

    def invalidate_all(self) -> int:
        """Wipe entire cache. Returns count of files removed."""
        with self._lock:
            count = 0
            for f in self.cache_dir.glob("*"):
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
            logger.warning(f"Full cache wipe: {count} files removed")
            return count

    def get_cache_status(self) -> Dict[str, Any]:
        """Return summary of all cached models — useful for dashboard."""
        status = {}
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                symbol = meta["symbol"]
                trained_at = datetime.fromisoformat(meta["trained_at"])
                age_days = (datetime.utcnow() - trained_at).days
                status[symbol] = {
                    "age_days": age_days,
                    "expired": age_days > self.ttl_days,
                    "version_ok": meta.get("cache_version") == CACHE_VERSION,
                    "feature_count": meta.get("feature_count", "unknown"),
                    "trained_at": meta["trained_at"],
                }
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return status

    # ── Private helpers ────────────────────────────────────────────────────────

    def _paths(self, symbol: str):
        safe = symbol.replace("/", "_").replace(":", "_")
        return (
            self.cache_dir / f"{safe}.meta.json",
            self.cache_dir / f"{safe}.model.pkl",
        )

    def _delete(self, symbol: str) -> None:
        """Delete both files for a symbol. Called inside lock."""
        meta_path, model_path = self._paths(symbol)
        for p in (meta_path, model_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _hash_features(feature_names: List[str]) -> str:
        """
        SHA-256 of sorted feature names.
        Sorting ensures order changes don't invalidate cache unnecessarily.
        """
        canonical = json.dumps(sorted(feature_names), separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON atomically using temp file + rename."""
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
        """Write pickle atomically using temp file + rename."""
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.move(str(tmp), str(path))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
# ── Backward-compatible function wrappers ──────────────────────────────────
# scanner.py uses these older function-based API calls
# They wrap the new ModelCache class internally

_cache = ModelCache()


def is_cache_valid(symbol: str, feature_names: List[str]) -> bool:
    """
    Check if a valid cached model exists for this symbol
    with the current feature set.

    Returns True if cache hit, False if miss/expired/stale.
    """
    model = _cache.get(symbol, feature_names)
    return model is not None


def load_models(symbol: str, feature_names: Optional[List[str]] = None) -> Optional[Any]:
    """
    Load cached model for symbol.
    Returns None if cache miss, version mismatch, TTL expired, or feature mismatch.

    Fix: no longer bypasses version and TTL checks — doing so caused
    stale models (trained on old feature sets like gld_level, vix_level)
    to be silently loaded, crashing prediction with KeyError on missing columns.

    feature_names: current feature list. If provided, feature hash is
      validated and None is returned on mismatch (forces retrain).
    """
    meta_path, model_path = _cache._paths(symbol)

    # ── Check metadata first ──────────────────────────────────────────
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Version check — reject caches from incompatible training runs
            if meta.get("cache_version") != CACHE_VERSION:
                logger.info(
                    f"[{symbol}] Cache version mismatch "
                    f"({meta.get('cache_version')} vs {CACHE_VERSION}), forcing retrain"
                )
                _cache._delete(symbol)
                return None

            # TTL check — reject caches older than allowed window
            try:
                trained_at = datetime.fromisoformat(meta["trained_at"])
                age_days = (datetime.utcnow() - trained_at).days
                if age_days > _cache.ttl_days:
                    logger.info(
                        f"[{symbol}] Cache expired ({age_days}d > {_cache.ttl_days}d), retraining"
                    )
                    _cache._delete(symbol)
                    return None
            except Exception:
                pass  # Non-critical: if date parse fails, proceed

            # Feature hash check (when caller provides current feature list)
            if feature_names:
                current_hash = ModelCache._hash_features(feature_names)
                cached_hash  = meta.get("feature_hash", "")
                if cached_hash and cached_hash != current_hash:
                    logger.warning(
                        f"[{symbol}] Feature set changed since training — "
                        f"cached={cached_hash[:8]} current={current_hash[:8]} — forcing retrain"
                    )
                    _cache._delete(symbol)
                    return None

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[{symbol}] Corrupt metadata ({e}), invalidating")
            _cache._delete(symbol)
            return None

    if not model_path.exists():
        return None

    try:
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"[{symbol}] load_models failed: {e}")
        _cache._delete(symbol)
        return None


def save_models(symbol: str, models: Any, trained_through: str = None,
                feature_names: Optional[List[str]] = None) -> bool:
    """
    Save models to cache for symbol.

    models: can be a dict or a single model object
    trained_through: ISO date string of last training bar (Fix 1.4).
      Pass str(train.index[-1].date()) from the walk-forward loop.
    feature_names: list of features the model was trained on.
      Used for hash-based staleness detection on next load.
    """
    return _cache.save(
        symbol=symbol,
        model=models,
        feature_names=feature_names or [],
        trained_through=trained_through,
    )