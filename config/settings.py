# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv('config/secrets.env')

# ============================================================
# SYSTEM SETTINGS
# ============================================================

SYSTEM_NAME = "AlphaEdge"
VERSION = "0.1.0"
MODE = "paper"  # "paper" or "live"

# ============================================================
# MARKET SETTINGS
# ============================================================

# Stocks to track
STOCK_WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX', 'SPY',
    'QQQ', 'JPM', 'V', 'JNJ', 'WMT'
]

# Crypto to track
CRYPTO_WATCHLIST = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT',
    'ADA/USDT', 'DOT/USDT'
]

# Timeframes
STOCK_TIMEFRAME = '1d'       # Daily candles for stocks
CRYPTO_TIMEFRAME = '4h'      # 4-hour candles for crypto
LOOKBACK_DAYS = 365           # 1 year of historical data

# ============================================================
# API KEYS (loaded from secrets.env)
# ============================================================

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL

NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# ============================================================
# RISK MANAGEMENT SETTINGS
# ============================================================

MAX_RISK_PER_TRADE = 0.02      # Risk 2% of capital per trade
MAX_PORTFOLIO_RISK = 0.06       # Max 6% total portfolio risk
MAX_POSITION_SIZE = 0.15        # Max 15% of capital in one position
MAX_DAILY_LOSS = 0.03           # Stop trading if down 3% in a day
MAX_DRAWDOWN = 0.10             # Emergency stop if down 10% from peak
MAX_OPEN_POSITIONS = 5          # Maximum concurrent positions

# ============================================================
# MODEL SETTINGS
# ============================================================

PREDICTION_THRESHOLD = 0.6      # Only trade if model confidence > 60%
ENSEMBLE_WEIGHTS = {
    'technical': 0.4,
    'sentiment': 0.3,
    'momentum': 0.3
}

RETRAIN_INTERVAL_DAYS = 30      # Retrain models every 30 days

# ============================================================
# TRADING SETTINGS
# ============================================================

STOP_LOSS_PCT = 0.05            # 5% stop loss
TAKE_PROFIT_PCT = 0.10          # 10% take profit (2:1 reward/risk)
TRAILING_STOP_PCT = 0.03        # 3% trailing stop

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading.log'