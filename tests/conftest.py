# tests/conftest.py
"""
Shared pytest fixtures for AlphaEdge test suite.
All external dependencies (broker, Telegram, yfinance) are mocked here
so tests run fully offline with no API keys required.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── OHLCV Fixtures ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, start: str = '2023-01-01',
                seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq='B')  # business days
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    high  = close * (1 + rng.uniform(0.001, 0.015, n))
    low   = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        'open'  : open_,
        'high'  : high,
        'low'   : low,
        'close' : close,
        'volume': vol,
    }, index=dates)
    df.index.name = 'date'
    return df


@pytest.fixture
def ohlcv_df():
    """300-bar OHLCV DataFrame — enough for all indicators."""
    return _make_ohlcv(300)


@pytest.fixture
def short_ohlcv_df():
    """50-bar OHLCV DataFrame — used to test insufficient-data guards."""
    return _make_ohlcv(50)


# ── Paper Trader Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_log(tmp_path):
    """Temporary log file path for PaperTrader state."""
    return str(tmp_path / 'paper_trades_test.json')


@pytest.fixture
def paper_trader(tmp_log):
    """Fresh PaperTrader with $10,000 capital and isolated state file."""
    from execution.paper_trader import PaperTrader
    pt = PaperTrader(
        starting_capital=10_000.0,
        max_positions=5,
        risk_per_trade_pct=0.02,
        daily_loss_limit_pct=0.05,
        log_file=tmp_log,
    )
    return pt


# ── Broker / API Mocks ────────────────────────────────────────────────────────

@pytest.fixture
def mock_telegram():
    """Mock Telegram bot — records calls, never sends real messages."""
    bot = MagicMock()
    bot.send_message.return_value = True
    bot.alert_buy_signal.return_value = True
    bot.alert_stop_loss.return_value = True
    bot.alert_take_profit.return_value = True
    return bot


@pytest.fixture
def mock_alpaca_broker():
    """Mock Alpaca broker with standard responses."""
    broker = MagicMock()
    broker.get_account.return_value = {
        'portfolio_value': 10_000.0,
        'cash': 10_000.0,
    }
    broker.get_positions.return_value = {}
    broker.buy.return_value = True
    broker.sell.return_value = True
    return broker


@pytest.fixture
def mock_circuit_breaker():
    """Circuit breaker that always passes (for testing trading logic)."""
    cb = MagicMock()
    cb.check.return_value = (True, 'OK')
    return cb


# ── Settings Override ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """
    Patch config.settings so tests never need real API keys.
    autouse=True applies this to every test automatically.
    """
    monkeypatch.setenv('ALPACA_API_KEY',    'test-key-alpaca')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'test-secret-alpaca')
    monkeypatch.setenv('GATEIO_API_KEY',    'test-key-gateio')
    monkeypatch.setenv('GATEIO_SECRET',     'test-secret-gateio')
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','test:token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID',  '123456789')
    monkeypatch.setenv('GROQ_API_KEY',      'test-groq-key')
    monkeypatch.setenv('WEBHOOK_SECRET',    'test-webhook-secret')


# ── Trade History Factories ───────────────────────────────────────────────────

def make_trade(action='SELL', symbol='AAPL', pnl=50.0, pnl_pct=0.05,
               days_ago=1, reason='take_profit'):
    """Factory for a single trade dict."""
    return {
        'action'   : action,
        'symbol'   : symbol,
        'shares'   : 10,
        'price'    : 150.0,
        'fill_price': 150.0,
        'pnl'      : pnl,
        'pnl_pct'  : pnl_pct,
        'date'     : (datetime.now() - timedelta(days=days_ago)).isoformat(),
        'reason'   : reason,
    }


@pytest.fixture
def sample_trade_history():
    """A realistic mix of wins and losses for testing analytics."""
    return [
        make_trade('SELL', 'AAPL',  pnl= 80.0, pnl_pct= 0.053, days_ago=1, reason='take_profit'),
        make_trade('SELL', 'MSFT',  pnl=-35.0, pnl_pct=-0.023, days_ago=2, reason='stop_loss'),
        make_trade('SELL', 'NVDA',  pnl=120.0, pnl_pct= 0.080, days_ago=3, reason='take_profit'),
        make_trade('SELL', 'AMD',   pnl=-50.0, pnl_pct=-0.033, days_ago=4, reason='stop_loss'),
        make_trade('SELL', 'GOOGL', pnl= 45.0, pnl_pct= 0.030, days_ago=5, reason='trailing_stop'),
        make_trade('PARTIAL_SELL', 'TSLA', pnl=25.0, pnl_pct=0.050,
                   days_ago=3, reason='partial_exit_5pct'),
    ]
