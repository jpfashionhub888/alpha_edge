#!/usr/bin/env python3
"""
backtesting/run_backtest.py
Phase 1 backtest runner — earnings revision momentum signal.

This is the entry point for running the SUE signal backtest.
Fetches data, runs the event-driven engine, and prints a performance
tearsheet with Gate 1 + Gate 2 metrics.

Usage (from /root/alpha_edge):
    python3 backtesting/run_backtest.py

    # With custom parameters:
    python3 backtesting/run_backtest.py --symbols AAPL MSFT GOOGL --start 2021-01-01

Options:
    --symbols  : space-separated list of symbols (default: 40-stock liquid universe)
    --start    : start date YYYY-MM-DD (default: 2021-01-01)
    --end      : end date YYYY-MM-DD (default: today)
    --capital  : initial capital USD (default: 100000)
    --top-n    : max simultaneous positions (default: 10)
    --decay    : earnings decay days (default: 20)
    --no-cache : disable data cache (re-download everything)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    datefmt= '%H:%M:%S',
)
logger = logging.getLogger('backtest')

# Suppress noisy loggers
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description='AlphaEdge Phase 1 Backtest')
    p.add_argument('--symbols', nargs='+', default=None,
                   help='Symbols to include (default: liquid 40-stock universe)')
    p.add_argument('--start',   default='2021-01-01',
                   help='Start date (default: 2021-01-01)')
    p.add_argument('--end',     default=str(date.today()),
                   help='End date (default: today)')
    p.add_argument('--capital', type=float, default=100_000,
                   help='Initial capital USD (default: 100000)')
    p.add_argument('--top-n',   type=int, default=10,
                   help='Max simultaneous positions (default: 10)')
    p.add_argument('--decay',   type=int, default=20,
                   help='Earnings signal decay days (default: 20)')
    p.add_argument('--no-cache', action='store_true',
                   help='Disable data cache')
    return p.parse_args()


DEFAULT_SYMBOLS = [
    # Large-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CRM',
    # Financials
    'JPM', 'BAC', 'V', 'MA', 'GS',
    # Healthcare
    'JNJ', 'UNH', 'ABBV', 'LLY', 'MRK', 'TMO',
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'MCD', 'COST', 'HD',
    # Energy / Industrials
    'CVX', 'CAT', 'HON', 'RTX',
    # Semiconductors
    'TXN', 'QCOM', 'AMD',
    # Other
    'NEE', 'ACN', 'IBM', 'AMGN', 'PM', 'ABT',
]


def main():
    args = parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    logger.info('=' * 55)
    logger.info('AlphaEdge Phase 1 — Earnings Revision Backtest')
    logger.info('=' * 55)
    logger.info(f'Symbols       : {len(symbols)}')
    logger.info(f'Period        : {args.start} → {args.end}')
    logger.info(f'Capital       : ${args.capital:,.0f}')
    logger.info(f'Max positions : {args.top_n}')
    logger.info(f'Earnings decay: {args.decay} days')

    from backtesting.signals.library.earnings_revision import EarningsRevisionBacktest

    bt = EarningsRevisionBacktest(
        initial_capital   = args.capital,
        max_positions     = args.top_n,
        decay_days        = args.decay,
        lookback_quarters = 8,
    )

    # Override cache if requested
    if args.no_cache:
        bt.loader.cache_enabled = False

    result = bt.run(
        symbols    = symbols,
        start_date = args.start,
        end_date   = args.end,
    )

    # Save equity curve
    out_path = Path('backtesting/results')
    out_path.mkdir(parents=True, exist_ok=True)
    eq_path = out_path / f'equity_curve_{args.start[:4]}.csv'
    result.equity_curve.to_csv(eq_path)
    logger.info(f'Equity curve saved → {eq_path}')

    if len(result.trades) > 0:
        trades_path = out_path / f'trades_{args.start[:4]}.csv'
        result.trades.to_csv(trades_path, index=False)
        logger.info(f'Trade log saved → {trades_path}')

    # Gate checks
    from backtesting.analysis.metrics import (
        returns_from_equity, sharpe_ratio, max_drawdown, performance_summary
    )
    eq      = result.equity_curve['equity']
    ret     = returns_from_equity(eq)
    sharpe  = sharpe_ratio(ret)
    mdd     = max_drawdown(eq)

    print('\n' + '─' * 55)
    print('Phase 1 Gate 2 Check (Cost Survival):')
    print(f'  Sharpe (net of fills): {sharpe:.2f}   [need > 1.0]')
    print(f'  Max Drawdown:          {mdd:.1%}  [need > -25%]')
    gate2 = sharpe > 1.0 and mdd > -0.25
    print(f'  Gate 2 result:         {"PASS ✓" if gate2 else "FAIL ✗"}')
    if not gate2:
        print()
        print('  If Sharpe < 1.0, possible causes:')
        print('  1. Signal IC is low → need more symbols or longer history')
        print('  2. Transaction costs too high → reduce rebalance frequency')
        print('  3. Earnings data quality → check yfinance vs Polygon coverage')
    print('─' * 55)


if __name__ == '__main__':
    main()
