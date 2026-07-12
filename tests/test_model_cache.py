"""Tests for model_cache.py — TTL, feature hash, atomic write, version gate."""
import json
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_cache import ModelCache, CACHE_VERSION
except ImportError as e:
    pytest.skip(f"model_cache not importable: {e}", allow_module_level=True)


class FakeModel:
    """Minimal sklearn-like stub that can be pickled."""
    def predict(self, X):
        return [0] * len(X)


FEATURES = ["close", "rsi", "macd", "vol"]


@pytest.fixture()
def cache(tmp_path):
    return ModelCache(cache_dir=str(tmp_path), ttl_days=7)


# ── get() / save() round-trip ────────────────────────────────────────────────

def test_save_and_load(cache):
    model = FakeModel()
    cache.save("AAPL", model, FEATURES)
    loaded = cache.get("AAPL", FEATURES)
    assert loaded is not None, "Cached model should load back"


def test_miss_on_empty(cache):
    assert cache.get("TSLA", FEATURES) is None, "Empty cache must return None"


# ── TTL expiry ────────────────────────────────────────────────────────────────

def test_ttl_expiry(cache):
    model = FakeModel()
    cache.save("NVDA", model, FEATURES)

    # Forge the trained_at timestamp to be 8 days ago
    meta_path, _ = cache._paths("NVDA")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["trained_at"] = (datetime.utcnow() - timedelta(days=8)).isoformat()
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    assert cache.get("NVDA", FEATURES) is None, "Expired cache must return None"


# ── Feature hash invalidation ─────────────────────────────────────────────────

def test_feature_hash_mismatch(cache):
    model = FakeModel()
    cache.save("MSFT", model, FEATURES)
    different_features = FEATURES + ["new_feature"]
    assert cache.get("MSFT", different_features) is None, "Feature mismatch must return None"


def test_same_features_pass(cache):
    model = FakeModel()
    cache.save("MSFT", model, FEATURES)
    assert cache.get("MSFT", FEATURES) is not None, "Same features must hit cache"


# ── Cache version gate ────────────────────────────────────────────────────────

def test_version_mismatch(cache):
    model = FakeModel()
    cache.save("GOOGL", model, FEATURES)

    meta_path, _ = cache._paths("GOOGL")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["cache_version"] = "0.0"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    assert cache.get("GOOGL", FEATURES) is None, "Version mismatch must invalidate"


# ── Atomic write (corrupt file doesn't crash) ─────────────────────────────────

def test_corrupt_metadata_returns_none(cache):
    model = FakeModel()
    cache.save("AMZN", model, FEATURES)
    meta_path, _ = cache._paths("AMZN")
    meta_path.write_text("NOT_JSON")
    assert cache.get("AMZN", FEATURES) is None, "Corrupt metadata must return None"
