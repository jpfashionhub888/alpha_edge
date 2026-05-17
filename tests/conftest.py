# tests/conftest.py
"""
pytest configuration.

Adds the repo root to sys.path so tests can `from execution... import ...`
without needing pip install -e or PYTHONPATH set externally.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
