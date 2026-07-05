# run_validation.py
"""
AlphaEdge V5 — Post-HyperOpt Validation Script

Runs the Event-Driven Backtest Engine with current settings.yaml parameters
on all watchlist symbols, then sends a full Telegram report with:
  - Sharpe ratio
  - Win rate
  - Total return
  - Commission + slippage cost
  - Trade count
  - Safety verdict (READY / NOT READY for live)

Usage:
    python run_validation.py               # uses config/settings.yaml params
    python run_validation.py --days 365    # custom lookback
    python run_validation.py --no-telegram # silent mode

Runs automatically after each HyperOpt optimisation.
"""

import argparse
import logging
import time
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
VALIDATION_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'AMD', 'NFLX',
    'SPY', 'QQQ', 'IWM',
    'JPM', 'V', 'GS',
    'JNJ', 'PFE', 'UNH',
    'WMT', 'COST', 'HD',
    'XOM', 'CVX',
    'CRM', 'SNOW', 'NET', 'CRWD',
]
LOOKBACK_DAYS  = 365          # 1 year out-of-sample validation
INITIAL_CAPITAL = 50_000      # USD
SHARPE_THRESHOLD = 0.5        # Minimum acceptable Sharpe
WIN_RATE_THRESHOLD = 45.0        # Minimum acceptable win rate (already in % from summary())
MIN_TRADES = 10               # Minimum trades to be statistically meaningful
# ──────────────────────────────────────────────────────────────────────


def load_settings() -> dict:
    """Load current strategy params from settings.yaml."""
    defaults = {
        'buy_threshold' : 0.55,
        'atr_stop_mult' : 1.0,
        'atr_target_mult': 2.5,
        'kelly_multiplier': 0.5,
        'max_pos_pct'   : 0.10,
    }
    try:
        import yaml
        with open('config/settings.yaml') as f:
            cfg = yaml.safe_load(f) or {}
        thr  = cfg.get('signal_thresholds', {})
        risk = cfg.get('risk_management', {})
        hyp  = cfg.get('hyperopt', {})
        return {
            'buy_threshold'   : float(thr.get('buy_threshold',    hyp.get('buy_threshold',    defaults['buy_threshold']))),
            'atr_stop_mult'   : float(risk.get('atr_stop_mult',   hyp.get('atr_stop_mult',   defaults['atr_stop_mult']))),
            'atr_target_mult' : float(risk.get('atr_target_mult', hyp.get('atr_target_mult', defaults['atr_target_mult']))),
            'kelly_multiplier': float(risk.get('kelly_multiplier', hyp.get('kelly_multiplier', defaults['kelly_multiplier']))),
            'max_pos_pct'     : float(risk.get('max_position_size', defaults['max_pos_pct'])),
        }
    except Exception as e:
        logger.warning(f'Could not load settings.yaml: {e} — using defaults')
        return defaults


def fetch_validation_data(symbols: list, lookback_days: int) -> dict:
    """Fetch OHLCV data for all symbols via yfinance."""
    start = (datetime.utcnow() - timedelta(days=lookback_days + 60)).strftime('%Y-%m-%d')
    data  = {}
    print(f'  Fetching {len(symbols)} symbols from yfinance...')
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(start=start, auto_adjust=True)
            if df.empty or len(df) < 60:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            data[sym] = df
        except Exception as e:
            logger.debug(f'{sym}: fetch failed — {e}')
    print(f'  Fetched {len(data)} symbols')
    return data


def run_validation(lookback_days: int = LOOKBACK_DAYS,
                   send_telegram: bool = True) -> dict:
    """
    Run full validation pipeline:
    1. Fetch data
    2. Run EventDrivenBacktest with current settings params
    3. Compute aggregate stats
    4. Send Telegram report
    5. Return results dict
    """
    start_ts = time.time()

    print('\n' + '=' * 60)
    print(f'  ALPHAEDGE V5 — VALIDATION RUN')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)

    # ── 1. Load settings ──────────────────────────────────────────
    params = load_settings()
    print(f'\n  Params loaded:')
    for k, v in params.items():
        print(f'    {k}: {v}')

    # ── 2. Fetch data ─────────────────────────────────────────────
    print(f'\n  Fetching {lookback_days}d of validation data...')
    all_data = fetch_validation_data(VALIDATION_SYMBOLS, lookback_days)
    if not all_data:
        logger.error('No data fetched — aborting validation')
        return {}

    # ── 3. Run event-driven backtest ──────────────────────────────
    print('\n  Running Event-Driven Backtest...')
    try:
        from backtest.event_engine import EventDrivenBacktest, SimpleAlphaEdgeStrategy

        strategy = SimpleAlphaEdgeStrategy(
            buy_threshold = params['buy_threshold'],
            max_pos_pct   = params['max_pos_pct'],
        )
        engine = EventDrivenBacktest(
            strategy        = strategy,
            initial_capital = INITIAL_CAPITAL,
            commission      = 0.005,
            slippage_pct    = 0.0005,
        )
        result = engine.run(all_data)
        summary = result.summary()
    except Exception as e:
        logger.error(f'Backtest failed: {e}')
        import traceback; traceback.print_exc()
        return {}

    # ── 4. Safety verdict ─────────────────────────────────────────
    sharpe    = float(summary.get('sharpe_ratio', 0) or 0)
    win_rate  = float(summary.get('win_rate', 0) or 0)
    n_trades  = int(summary.get('total_trades', 0) or 0)
    ret_pct   = float(summary.get('total_return_pct', 0) or 0)
    commission = float(summary.get('total_commission', 0) or 0)
    slippage  = float(summary.get('total_slippage', 0) or 0)

    ready = (
        sharpe   >= SHARPE_THRESHOLD     and
        win_rate >= WIN_RATE_THRESHOLD   and
        n_trades >= MIN_TRADES
    )
    verdict = '✅ READY FOR LIVE' if ready else '⚠️ NOT READY — paper trade more'

    # ── 5. Print results ──────────────────────────────────────────
    elapsed = time.time() - start_ts
    print(f'\n  ══════════════ RESULTS ══════════════')
    print(f'  Symbols validated : {len(all_data)}')
    print(f'  Total trades      : {n_trades}')
    print(f'  Total return      : {ret_pct:+.2f}%')
    print(f'  Sharpe ratio      : {sharpe:.3f}')
    print(f'  Win rate          : {win_rate:.1f}%')
    print(f'  Costs (C+S)       : ${commission + slippage:.2f}')
    print(f'  Verdict           : {verdict}')
    print(f'  Runtime           : {elapsed:.1f}s')
    print(f'  ═════════════════════════════════════')

    # ── 6. Telegram report ────────────────────────────────────────
    if send_telegram:
        try:
            from monitoring.telegram_bot import TelegramBot
            bot = TelegramBot()
            status_emoji = '✅' if ready else '⚠️'
            msg = (
                f'📊 AlphaEdge V5 — Validation Report\n'
                f'📅 {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
                f'━━━━━━━━━━━━━━━━━━━━\n'
                f'📈 Return     : {ret_pct:+.2f}%\n'
                f'📉 Sharpe     : {sharpe:.3f}\n'
                f'🎯 Win Rate   : {win_rate:.1f}%\n'
                f'🔁 Trades     : {n_trades}\n'
                f'💸 C+S costs  : ${commission + slippage:.2f}\n'
                f'🔑 Params     : buy={params["buy_threshold"]:.2f} '
                f'stop={params["atr_stop_mult"]:.1f}x '
                f'target={params["atr_target_mult"]:.1f}x\n'
                f'━━━━━━━━━━━━━━━━━━━━\n'
                f'{status_emoji} Verdict: {verdict}'
            )
            bot.send_message(msg)
            print('  Telegram report sent ✅')
        except Exception as e:
            logger.warning(f'Telegram send failed: {e}')

    return {
        'sharpe'    : sharpe,
        'win_rate'  : win_rate,
        'n_trades'  : n_trades,
        'ret_pct'   : ret_pct,
        'costs'     : commission + slippage,
        'ready'     : ready,
        'verdict'   : verdict,
        'params'    : params,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaEdge V5 Validation')
    parser.add_argument('--days',         type=int,  default=LOOKBACK_DAYS,
                        help='Lookback window in days (default: 365)')
    parser.add_argument('--no-telegram',  action='store_true',
                        help='Skip Telegram notification')
    args = parser.parse_args()

    results = run_validation(
        lookback_days  = args.days,
        send_telegram  = not args.no_telegram,
    )
    # Exit code 1 if not ready (useful for CI/CD checks)
    sys.exit(0 if results.get('ready', False) else 1)
