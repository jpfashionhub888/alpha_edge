# main.py
"""
AlphaEdge Main Orchestrator V5

Responsibilities (thin orchestrator only):
    - Initialise all components
    - Call StockScanner for signals
    - Act on signals via PaperTrader
    - Manage positions (stop-loss / take-profit)
    - Write dashboard JSON (atomic)
    - Send Telegram alerts
    - Run weekly analytics + critic review

Does NOT:
    - Contain any scan logic       (scanner.py)
    - Contain feature engineering  (feature_engine.py)
    - Contain model training       (technical_model.py)
    - Contain position sizing      (paper_trader.py)

Fixes applied (V5):
    - Scan logic extracted to scanner.py (testable, single-responsibility)
    - add_all_features() now called inside each window (look-ahead fix)
    - Earnings calendar fetched concurrently (was blocking 2-5 min)
    - Crypto open_position() now passes ATR proxy (was passing nothing)
    - Top-level exception handler with Telegram alert on crash
    - Signal weights promoted to named constants in scanner.py
    - MTF lambda replaced with named function (scanner.py)
    - Crypto prices fall back to entry price for stop-loss check
    - Insider boost now affects sizing_combined only (not signal)
    - Critic agent gated behind weekly check (was firing every scan)
    - Phase timing logged throughout
    - sector_scores save uses safe key access
    - P&L calculation uses trader.get_portfolio_value()
"""

import json
import logging
import os

# ── Load secrets from config/secrets.env ──────────────────────────────────
def _load_secrets():
    secrets_path = os.path.join(
        os.path.dirname(__file__), 'config', 'secrets.env'
    )
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

_load_secrets()

import tempfile
import time
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from critic_agent import CriticAgent
from execution.paper_trader import PaperTrader
from market_regime import MarketRegimeDetector
from models.sector_rotation import SectorRotation
from monitoring.telegram_bot import TelegramBot
from performance_analytics import PerformanceAnalytics
from risk_circuit_breaker import RiskCircuitBreaker
from scanner import StockScanner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  CONFIGURATION                                                       #
# ------------------------------------------------------------------ #

STOCK_WATCHLIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX',
    'SPY', 'QQQ', 'IWM', 'DIA',
    'JPM', 'V', 'GS', 'BAC', 'MS',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY', 'MRK',
    'WMT', 'COST', 'HD', 'MCD',
    'XOM', 'CVX', 'OXY',
    'SOFI', 'PLTR', 'RIVN', 'HOOD', 'MARA',
    'CRM', 'SNOW', 'NET', 'DDOG', 'CRWD',
]

CRYPTO_WATCHLIST = ['BTC/USD', 'ETH/USD', 'SOL/USD']

STARTING_CAPITAL = 10_000.0


# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def atomic_json_write(filepath: str, data: object) -> None:
    """
    Write JSON atomically: write to temp file then os.replace().
    Prevents corrupted JSON if the process is killed mid-write.
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(filepath) or '.', suffix='.tmp'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ------------------------------------------------------------------ #
#  INNER SCAN FUNCTION                                                 #
# ------------------------------------------------------------------ #

def _run_scan(telegram: TelegramBot) -> None:
    """
    Full scan logic. Separated from run_daily_scan() so the
    top-level wrapper can catch all exceptions and alert Telegram.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "🚀" * 25)
    print(f"  ALPHA EDGE V5 — {now}")
    print("  4-Model Ensemble + Sector Rotation + LSTM")
    print("🚀" * 25)

    scan_start = time.time()

    # ── Init trader ───────────────────────────────────────────────
    trader = PaperTrader(starting_capital=STARTING_CAPITAL)
    trader.load_state()

    # ================================================================
    # PHASE 0: SECTOR + MARKET REGIME + CIRCUIT BREAKER
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 0: MARKET CONTEXT")
    print("=" * 60)
    phase_start = time.time()

    sector_analyzer = SectorRotation()
    sector_scores   = sector_analyzer.analyze()

    regime_filter = MarketRegimeDetector()
    try:
        import yfinance as yf
        _spy = yf.download(
            'SPY',
            period='1y',
            interval='1d',
            progress=False,
            auto_adjust=True,
        )
        _spy.columns = [
            c[0].lower() if isinstance(c, tuple) else c.lower()
            for c in _spy.columns
        ]
        market_regime = regime_filter.detect(_spy)
        # Add can_trade key for backward compatibility
        market_regime['can_trade'] = market_regime.get('tradeable', True)
        market_regime['reason'] = (
            f"regime={market_regime.get('regime','unknown')} "
            f"confidence={market_regime.get('confidence',0):.2f}"
        )
    except Exception as _re:
        logger.warning(
            f"Regime detection failed: {_re} — defaulting to unknown"
        )
        market_regime = {
            'regime'    : 'unknown',
            'confidence': 0.0,
            'tradeable' : True,
            'can_trade' : True,
            'signals'   : {},
            'reason'    : 'regime detection failed',
        }

    # Portfolio value for circuit breaker
    # Portfolio value for circuit breaker
    portfolio_value = trader.get_portfolio_value(
        {s: p.get('entry_price', 0)
         for s, p in trader.positions.items()}
    )

    circuit_breaker   = RiskCircuitBreaker()
    circuit_triggered = circuit_breaker.check(
        current_value    = portfolio_value,
        starting_capital = trader.starting_capital,
        telegram         = telegram,
    )

    if circuit_triggered:
        market_regime['can_trade'] = False
        print("  🛑 Trading suspended by circuit breaker!")

    if not market_regime['can_trade']:
        print(f"\n  CASH MODE ACTIVATED!")
        print(f"  Reason: {market_regime.get('reason', 'unknown')}")

    logger.info("Phase 0: %.1fs", time.time() - phase_start)

    # ================================================================
    # PHASE 1 + 2: SCANNER
    # ================================================================
    phase_start = time.time()

    scanner = StockScanner(
        stock_watchlist  = STOCK_WATCHLIST,
        crypto_watchlist = CRYPTO_WATCHLIST,
        sector_analyzer  = sector_analyzer,
        market_regime    = market_regime,
        open_positions   = trader.positions,
    )

    # Earnings calendar — concurrent fetch
    earnings_soon = scanner.fetch_earnings_calendar()

    # Stock signals
    stock_signals  = scanner.scan_stocks()

    # Crypto signals
    crypto_signals = scanner.scan_crypto()

    logger.info(
        "Phase 1+2 (scan): %.1fs", time.time() - phase_start
    )

    # ================================================================
    # PHASE 3: EXECUTE STOCK SIGNALS
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: EXECUTE SIGNALS")
    print("=" * 60)
    phase_start = time.time()

    dashboard_signals = {}

    for symbol, data in sorted(
        stock_signals.items(),
        key=lambda x: x[1]['prediction'],
        reverse=True,
    ):
        # Always record every signal for dashboard
        dashboard_signals[symbol] = {
            'prediction': data['prediction'],
            'regime'    : data['regime'],
            'price'     : data['price'],
            'sentiment' : data['sentiment'],
            'sector'    : data['sector'],
            'signal'    : data['signal'],
            'combined'  : data['combined'],
        }

        action = data.get('action', 'SKIP')

        # Update dashboard signal if filtered
        if action not in ('OPEN', 'SKIP'):
            dashboard_signals[symbol]['signal'] = action
            continue

        if action != 'OPEN':
            continue

        # Open position using insider-adjusted sizing score
        atr    = data.get('atr')
        opened = trader.open_position(
            symbol,
            data['price'],
            data['sizing_combined'],   # insider-adjusted, not raw combined
            reason=data['regime'],
            atr=atr,
        )

        if opened:
            telegram.alert_buy_signal(
                symbol,
                data['price'],
                data['prediction'],
                data['regime'],
                data['sentiment'],
            )

    logger.info(
        "Phase 3 (execute stocks): %.1fs", time.time() - phase_start
    )

    # ================================================================
    # PHASE 4: EXECUTE CRYPTO SIGNALS
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: CRYPTO SIGNALS")
    print("=" * 60)
    phase_start = time.time()

    for symbol, data in crypto_signals.items():
        dashboard_signals[symbol] = {
            'prediction': data['prediction'],
            'regime'    : data['regime'],
            'price'     : data['price'],
            'sentiment' : 0.0,
            'sector'    : 'Crypto',
            'signal'    : data['signal'],
            'combined'  : data['combined'],
        }

        if data.get('action') != 'OPEN':
            continue

        # ATR proxy already computed in scanner.scan_crypto()
        opened = trader.open_position(
            symbol,
            data['price'],
            data['sizing_combined'],
            reason=data['regime'],
            atr=data['atr'],          # 5% price proxy — never None
        )

        if opened:
            telegram.alert_buy_signal(
                symbol,
                data['price'],
                data['prediction'],
                data['regime'],
                0.0,
            )

    logger.info(
        "Phase 4 (execute crypto): %.1fs", time.time() - phase_start
    )

    # ================================================================
    # PHASE 5: POSITION MANAGEMENT
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: POSITION MANAGEMENT")
    print("=" * 60)
    phase_start = time.time()

    # Build current_prices with fallback for missing symbols
    # Fix: open crypto positions without live prices now use
    # entry_price so stop-loss still evaluates correctly
    current_prices = {}

    for s, d in stock_signals.items():
        if d.get('price', 0) > 0:
            current_prices[s] = d['price']

    for s, d in crypto_signals.items():
        if d.get('price', 0) > 0:
            current_prices[s] = d['price']

    # Safety net: open positions with no live price → entry price
    for sym, pos in trader.positions.items():
        if sym not in current_prices:
            current_prices[sym] = pos.get('entry_price', 0)
            logger.warning(
                "No live price for %s — using entry price "
                "for stop-loss check", sym
            )

    if trader.positions:
        print("\n  Checking stop-loss / take-profit...")
        for symbol in list(trader.positions.keys()):
            if symbol not in current_prices:
                continue

            pos_before = trader.positions.get(symbol, {})
            entry      = pos_before.get('entry_price', 0)

            trader.update_position(symbol, current_prices[symbol])

            # Position closed by update_position?
            if symbol not in trader.positions:
                exit_price = current_prices[symbol]
                pnl        = (
                    (exit_price - entry)
                    * pos_before.get('shares', 0)
                )
                if pnl < 0:
                    telegram.alert_stop_loss(symbol, exit_price, pnl)
                else:
                    telegram.alert_take_profit(symbol, exit_price, pnl)
    else:
        print("\n  No open positions to manage")

    logger.info(
        "Phase 5 (position mgmt): %.1fs", time.time() - phase_start
    )

    # ================================================================
    # PHASE 6: SAVE DASHBOARD DATA
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: SAVE DASHBOARD DATA")
    print("=" * 60)
    phase_start = time.time()

    atomic_json_write('logs/latest_signals.json', dashboard_signals)
    atomic_json_write('logs/earnings.json', earnings_soon)

    # Safe sector scores save — guards against missing keys
    safe_sector_scores = {}
    for sector, d in sector_scores.items():
        try:
            safe_sector_scores[sector] = {
                'score'       : float(d.get('score', 0.0)),
                'flow'        : d.get('flow', 'neutral'),
                'momentum_21d': float(d.get('momentum_21d', 0.0)),
            }
        except Exception as e:
            logger.warning(
                "Skipping sector %s in save: %s", sector, e
            )

    atomic_json_write('logs/sectors.json', safe_sector_scores)
    print("  💾 All data saved (atomic)")

    logger.info(
        "Phase 6 (save): %.1fs", time.time() - phase_start
    )

    # ================================================================
    # PHASE 7: PORTFOLIO SUMMARY + TELEGRAM
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 7: PORTFOLIO SUMMARY")
    print("=" * 60)

    trader.get_summary(current_prices)

    # Update current price on each position for state save
    for symbol, pos in trader.positions.items():
        pos['current_price'] = current_prices.get(
            symbol, pos.get('entry_price', 0)
        )
    trader.save_state()

    # Use trader's own method — single source of truth for P&L
    total_value = trader.get_portfolio_value(current_prices)
    total_pnl   = total_value - trader.starting_capital
    total_pct   = total_pnl / trader.starting_capital

    # Build positions dict for Telegram (with live P&L)
    positions_with_pnl = {}
    for symbol, pos in trader.positions.items():
        curr  = current_prices.get(symbol, pos.get('entry_price', 0))
        entry = pos.get('entry_price', 0)
        shares= pos.get('shares', 0)
        positions_with_pnl[symbol] = {
            **pos,
            'current_price': curr,
            'pnl'          : (curr - entry) * shares,
            'pnl_pct'      : (
                (curr - entry) / entry if entry > 0 else 0
            ),
        }

    telegram.alert_daily_summary(
        total_value, total_pnl, total_pct,
        positions_with_pnl, dashboard_signals,
    )

    # ================================================================
    # PHASE 8: WEEKLY ANALYTICS + CRITIC (gated — not every scan)
    # ================================================================
    analytics = PerformanceAnalytics()
    if analytics.should_run_today():
        print("\n  📊 Sending Weekly Performance Report...")
        analytics.send_report(telegram)

        # Critic review runs alongside weekly report only
        # Fix: was firing every scan — now gated correctly
        critic = CriticAgent()
        critic.run_weekly_review(
            trade_history    = trader.trade_history,
            portfolio_value  = total_value,
            starting_capital = trader.starting_capital,
            telegram_bot     = telegram,
        )

    # ================================================================
    # DONE
    # ================================================================
    total_duration = time.time() - scan_start

    print("\n" + "✅" * 25)
    print("  ALPHA EDGE V5 SCAN COMPLETE")
    print("✅" * 25)
    print(f"\n  Scanned : {len(stock_signals)} stocks"
          f" + {len(crypto_signals)} crypto")
    print(f"  Earnings: {len(earnings_soon)} stocks this week")
    print(f"  Duration: {total_duration:.1f}s")
    print(f"  Models  : XGB + LGB + RF + CatBoost + LSTM per stock")
    print("\n  Dashboard → python run_dashboard.py")
    print("             → http://localhost:8050\n")


# ------------------------------------------------------------------ #
#  PUBLIC ENTRY POINT                                                  #
# ------------------------------------------------------------------ #

def run_daily_scan() -> None:
    """
    Top-level entry point with full exception handling.

    If anything inside _run_scan() fails, a Telegram alert
    is sent before re-raising so you always know the scan failed.
    Telegram is initialised here (not inside _run_scan) so it is
    available even if the crash happens before telegram is created.
    """
    telegram = TelegramBot()
    try:
        _run_scan(telegram)
    except Exception as e:
        logger.critical(
            "SCAN FAILED: %s", e, exc_info=True
        )
        try:
            telegram.send_message(
                f"🚨 ALPHA EDGE SCAN FAILED\n"
                f"Time : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Error: {str(e)[:500]}"
            )
        except Exception:
            pass   # never let telegram failure hide the real error
        raise


if __name__ == "__main__":
    run_daily_scan()