# tests/conftest.py
"""
Shared pytest fixtures for AlphaEdge test suite.

DATA POLICY: All OHLCV data is REAL historical data from Yahoo Finance
(2020-01-01 to 2025-12-30), downloaded via scripts/fetch_test_fixtures.py
and stored in tests/fixtures/. No synthetic/random data is used.

All external dependencies (broker, Telegram, yfinance live calls) are
mocked so tests run fully offline with no API keys required.

To refresh the data:
    python scripts/fetch_test_fixtures.py
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── Fixture paths ──────────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / 'fixtures'

FIXTURE_FILES = {
    'AAPL'   : FIXTURES_DIR / 'aapl_daily.csv',
    'NVDA'   : FIXTURES_DIR / 'nvda_daily.csv',
    'SPY'    : FIXTURES_DIR / 'spy_daily.csv',
    'BTC-USD': FIXTURES_DIR / 'btc_usd_daily.csv',
}


def _load_real_ohlcv(ticker: str, n_rows: int | None = None) -> pd.DataFrame:
    """
    Load real historical OHLCV data from the fixtures directory.

    Args:
        ticker:  One of AAPL, NVDA, SPY, BTC-USD
        n_rows:  If set, return only the last n_rows rows (most recent data)
    """
    path = FIXTURE_FILES.get(ticker)
    if path is None:
        raise ValueError(
            f"No fixture for ticker '{ticker}'. "
            f"Available: {list(FIXTURE_FILES.keys())}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture file missing: {path}\n"
            f"Run: python scripts/fetch_test_fixtures.py"
        )

    df = pd.read_csv(path, index_col='date', parse_dates=True)
    df.index.name = 'date'

    # Ensure standard lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # Drop any rows where close is NaN
    df = df.dropna(subset=['close'])

    if n_rows is not None:
        df = df.iloc[-n_rows:]

    return df


# ── OHLCV Fixtures — REAL historical data ─────────────────────────────────────

@pytest.fixture
def ohlcv_df():
    """
    Full AAPL real daily data (2020-2025, ~1507 rows).
    Sufficient for all long-window indicators (SMA-200, ATR-14, etc.).
    """
    return _load_real_ohlcv('AAPL')


@pytest.fixture
def ohlcv_df_nvda():
    """
    NVDA real daily data — high volatility stock for stress testing.
    """
    return _load_real_ohlcv('NVDA')


@pytest.fixture
def ohlcv_df_spy():
    """
    SPY real daily data — market benchmark / regime context.
    """
    return _load_real_ohlcv('SPY')


@pytest.fixture
def ohlcv_df_btc():
    """
    BTC-USD real daily data — crypto path through feature engine.
    """
    return _load_real_ohlcv('BTC-USD')


@pytest.fixture
def short_ohlcv_df():
    """
    Only 50 rows of AAPL (most recent) — used to test insufficient-data guards.
    Even this is real data, not synthetic.
    """
    return _load_real_ohlcv('AAPL', n_rows=50)


@pytest.fixture
def bear_market_df():
    """
    AAPL data ending in the 2022 bear market (rate-hike selloff).
    Slice includes 2020-2022 to provide sufficient rolling-window warmup:
    alpha158 uses rolling(252), so we need 252+ rows before the period
    of interest. The feature-engine test only cares about bear-regime
    characteristics at the END of this slice (2022 data).
    """
    df = _load_real_ohlcv('AAPL')
    return df['2020-01-01':'2022-12-31']


@pytest.fixture
def bull_market_df():
    """
    NVDA data ending in the 2023 AI bull run.
    Slice includes 2021-2023 to provide sufficient rolling-window warmup:
    alpha158 uses rolling(252), so we need 252+ rows before 2023 begins.
    """
    df = _load_real_ohlcv('NVDA')
    return df['2021-01-01':'2023-12-31']


@pytest.fixture
def volatile_period_df():
    """
    SPY data ending in the COVID crash + recovery (2020).
    Slice includes 2018-2020 to provide sufficient rolling-window warmup:
    alpha158 uses rolling(252), so we need 252+ rows before Mar 2020.
    """
    df = _load_real_ohlcv('SPY')
    return df['2018-01-01':'2020-12-31']


# ── Fixture metadata helper ───────────────────────────────────────────────────

def load_fixture_manifest() -> dict:
    """Return the manifest.json describing all available fixture files."""
    manifest_path = FIXTURES_DIR / 'manifest.json'
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


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


# ── Real Trade History (based on real price levels) ───────────────────────────

def make_trade(action='SELL', symbol='AAPL', pnl=50.0, pnl_pct=0.05,
               days_ago=1, reason='take_profit', price=150.0):
    """Factory for a single trade dict using realistic price levels."""
    return {
        'action'    : action,
        'symbol'    : symbol,
        'shares'    : 10,
        'price'     : price,
        'fill_price': price,
        'pnl'       : pnl,
        'pnl_pct'   : pnl_pct,
        'date'      : (datetime.now() - timedelta(days=days_ago)).isoformat(),
        'reason'    : reason,
    }


@pytest.fixture
def sample_trade_history():
    """
    Realistic trade history using actual AAPL/NVDA/SPY price levels
    from the 2024 trading period.
    """
    return [
        # AAPL trades (price range ~170-230 in 2024)
        make_trade('SELL',    'AAPL', pnl= 183.0, pnl_pct= 0.053, days_ago=1,
                   reason='take_profit',  price=182.85),
        make_trade('SELL',    'AAPL', pnl= -79.0, pnl_pct=-0.023, days_ago=2,
                   reason='stop_loss',    price=175.20),
        # NVDA trades (price range ~400-900 in 2024)
        make_trade('SELL',    'NVDA', pnl= 640.0, pnl_pct= 0.080, days_ago=3,
                   reason='take_profit',  price=495.00),
        make_trade('SELL',    'NVDA', pnl=-225.0, pnl_pct=-0.033, days_ago=4,
                   reason='stop_loss',    price=430.50),
        # SPY trade (price range ~450-540 in 2024)
        make_trade('SELL',    'SPY',  pnl= 147.0, pnl_pct= 0.030, days_ago=5,
                   reason='trailing_stop', price=490.00),
        # Partial exit (regression test for PARTIAL_SELL filter)
        make_trade('PARTIAL_SELL', 'NVDA', pnl=112.0, pnl_pct=0.050, days_ago=3,
                   reason='partial_exit_5pct', price=460.00),
    ]
