# config/settings.py
# Dynamic Settings Loader V2

import os
import yaml
from dotenv import load_dotenv

# Load env keys from secrets.env
load_dotenv('config/secrets.env')
if os.path.exists('alpha_edge/config/secrets.env'):
    load_dotenv('alpha_edge/config/secrets.env')

# Load settings from settings.yaml
settings_path = 'config/settings.yaml'
if not os.path.exists(settings_path) and os.path.exists('alpha_edge/config/settings.yaml'):
    settings_path = 'alpha_edge/config/settings.yaml'

yaml_config = {}
if os.path.exists(settings_path):
    try:
        with open(settings_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
    except Exception as e:
        import sys
        print(f"Error loading settings.yaml: {e}", file=sys.stderr)

# ============================================================
# SYSTEM SETTINGS
# ============================================================
SYSTEM_NAME = yaml_config.get('system', {}).get('name', 'AlphaEdge')
VERSION = yaml_config.get('system', {}).get('version', '0.2.0')
MODE = yaml_config.get('system', {}).get('mode', 'paper')

# ============================================================
# MARKET SETTINGS
# ============================================================
STOCK_WATCHLIST = yaml_config.get('watchlist', {}).get('stocks', [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX', 'SPY',
    'QQQ', 'JPM', 'V', 'JNJ', 'WMT'
])
CRYPTO_WATCHLIST = yaml_config.get('watchlist', {}).get('crypto', [
    'BTC/USD', 'ETH/USD', 'SOL/USD'
])

STOCK_TIMEFRAME = '1d'
CRYPTO_TIMEFRAME = '4h'
LOOKBACK_DAYS = 365

# ============================================================
# SIGNAL THRESHOLDS
# ============================================================
BUY_THRESHOLD = yaml_config.get('signal_thresholds', {}).get('buy_threshold', 0.63)
PREDICTION_THRESHOLD = BUY_THRESHOLD  # Alias for backward compatibility
VOLUME_SPIKE_MIN = yaml_config.get('signal_thresholds', {}).get('volume_spike_min', 1.3)
MIN_RISK_REWARD = yaml_config.get('signal_thresholds', {}).get('min_risk_reward', 2.0)
MTF_BLOCK_THRESHOLD = yaml_config.get('signal_thresholds', {}).get('mtf_block_threshold', 0.0)
MTF_WEIGHT_IN_SIGNAL = yaml_config.get('signal_thresholds', {}).get('mtf_weight_in_signal', 0.15)

# ============================================================
# BACKTEST CONFIG
# ============================================================
TRAIN_WINDOW_DAYS = yaml_config.get('backtest_config', {}).get('train_window_days', 180)
RETRAIN_FREQUENCY_DAYS = yaml_config.get('backtest_config', {}).get('retrain_frequency_days', 30)
TOP_FEATURES = yaml_config.get('backtest_config', {}).get('top_features', 20)
MIN_AUC = yaml_config.get('backtest_config', {}).get('min_auc', 0.52)
USE_ATR_STOPS = yaml_config.get('backtest_config', {}).get('use_atr_stops', True)
RANDOM_SEED = yaml_config.get('backtest_config', {}).get('random_seed', 42)

# ============================================================
# API KEYS (loaded from secrets.env)
# ============================================================
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Telegram Alerts
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Groq AI
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Crypto Coinbase skeleton compatibility keys
CRYPTO_EXCHANGE = 'coinbasepro'
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY', '')
COINBASE_PASSPHRASE = os.getenv('COINBASE_PASSPHRASE', '')
COINBASE_SANDBOX = True

# ============================================================
# RISK MANAGEMENT SETTINGS
# ============================================================
MAX_RISK_PER_TRADE = yaml_config.get('risk_management', {}).get('max_risk_per_trade', 0.02)
MAX_PORTFOLIO_RISK = yaml_config.get('risk_management', {}).get('max_portfolio_risk', 0.06)
MAX_POSITION_SIZE = yaml_config.get('risk_management', {}).get('max_position_size', 0.15)
MAX_DAILY_LOSS = yaml_config.get('risk_management', {}).get('max_daily_loss', 0.02)
MAX_DRAWDOWN = yaml_config.get('risk_management', {}).get('max_drawdown', 0.10)
MAX_OPEN_POSITIONS = yaml_config.get('risk_management', {}).get('max_open_positions', 5)
MAX_PER_SECTOR = yaml_config.get('risk_management', {}).get('max_per_sector', 2)
COOLOFF_DAYS = yaml_config.get('risk_management', {}).get('cooloff_days', 5)

# Dynamic stops parameters
ATR_STOP_MULT = yaml_config.get('risk_management', {}).get('atr_stop_mult', 1.0)
ATR_TARGET_MULT = yaml_config.get('risk_management', {}).get('atr_target_mult', 2.5)
TRAILING_STOP_MULTIPLIER = yaml_config.get('risk_management', {}).get('trailing_stop_multiplier', 0.8)

# Kelly Criterion Settings
KELLY_POSITION_SIZING = yaml_config.get('risk_management', {}).get('kelly_position_sizing', True)
KELLY_MULTIPLIER = yaml_config.get('risk_management', {}).get('kelly_multiplier', 0.5)
KELLY_REWARD_RISK_RATIO = yaml_config.get('risk_management', {}).get('kelly_reward_risk_ratio', 2.5)

# Backward-compatible fixed stops percentages
STOP_LOSS_PCT = yaml_config.get('risk_management', {}).get('stop_loss_pct', 0.05)
TAKE_PROFIT_PCT = yaml_config.get('risk_management', {}).get('take_profit_pct', 0.10)
TRAILING_STOP_PCT = yaml_config.get('risk_management', {}).get('trailing_stop_pct', 0.035)

# ============================================================
# EXECUTION COSTS
# ============================================================
SLIPPAGE_PCT = yaml_config.get('execution_costs', {}).get('slippage_pct', 0.0005)
COMMISSION = yaml_config.get('execution_costs', {}).get('commission', 1.0)

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = yaml_config.get('logging', {}).get('log_level', 'INFO')
LOG_FILE = yaml_config.get('logging', {}).get('log_file', 'logs/trading.log')