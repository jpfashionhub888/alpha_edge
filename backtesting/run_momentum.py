#!/usr/bin/env python3
"""
backtesting/run_momentum.py
Phase 1 backtest -- 12-1 price momentum signal with SPY regime filter.

Usage (from /root/alpha_edge):
    python3 backtesting/run_momentum.py
    python3 backtesting/run_momentum.py --start 2018-01-01 --top-n 10
    python3 backtesting/run_momentum.py --no-regime   # disable SPY filter

Options:
    --start      : start date YYYY-MM-DD (default: 2018-01-01)
    --end        : end date (default: today)
    --capital    : initial capital USD (default: 100000)
    --top-n      : max simultaneous positions (default: 10)
    --no-regime  : disable SPY 200d MA regime filter
    --no-cache   : re-download all data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    datefmt= '%H:%M:%S',
)
logger = logging.getLogger('momentum_backtest')
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CRM',
    'JPM',  'BAC',  'V',     'MA',   'GS',
    'JNJ',  'UNH',  'ABBV',  'LLY',  'MRK',  'TMO',
    'PG',   'KO',   'PEP',   'WMT',  'MCD',  'COST', 'HD',
    'CVX',  'CAT',  'HON',   'RTX',
    'TXN',  'QCOM', 'AMD',
    'NEE',  'ACN',  'IBM',   'AMGN', 'PM',   'ABT',
]


def parse_args():
    p = argparse.ArgumentParser(description='AlphaEdge Momentum Backtest')
    p.add_argument('--start',     default='2018-01-01')
    p.add_argument('--end',       default=str(date.today()))
    p.add_argument('--capital',   type=float, default=100_000)
    p.add_argument('--top-n',     type=int,   default=10)
    p.add_argument('--no-regime', action='store_true')
    p.add_argument('--no-cache',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    logger.info('=' * 58)
    logger.info('AlphaEdge -- 12-1 Momentum Backtest')
    logger.info('=' * 58)
    logger.info('Symbols    : %d', len(SYMBOLS))
    logger.info('Period     : %s to %s', args.start, args.end)
    logger.info('Capital    : $%,.0f', args.capital)
    logger.info('Top-N      : %d', args.top_n)
    logger.info('Regime filter: %s', 'OFF' if args.no_regime else 'SPY 200d MA')

    from backtesting.signals.library.price_momentum import MomentumBacktest

    bt = MomentumBacktest(
        initial_capital = args.capital,
        max_positions   = args.top_n,
    )
    if args.no_cache:
        bt.loader.cache_enabled = False

    result = bt.run(
        symbols          = SYMBOLS,
        start_date       = args.start,
        end_date         = args.end,
        spy_regime_filter= not args.no_regime,
    )

    out = Path('backtesting/results')
    out.mkdir(parents=True, exist_ok=True)
    eq_path = out / f'equity_momentum_{args.start[:4]}.csv'
    result.equity_curve.to_csv(eq_path)
    logger.info('Equity curve -> %s', eq_path)

    if len(result.trades) > 0:
        t_path = out / f'trades_momentum_{args.start[:4]}.csv'
        result.trades.to_csv(t_path, index=False)
        logger.info('Trade log   -> %s', t_path)

    from backtesting.analysis.metrics import returns_from_equity, sharpe_ratio, max_drawdown
    eq     = result.equity_curve['equity']
    ret    = returns_from_equity(eq)
    sharpe = sharpe_ratio(ret)
    mdd    = max_drawdown(eq)

    print('\n' + '-' * 58)
    print('Gate 2 Check (Cost Survival):')
    print(f'  Sharpe  : {sharpe:.2f}  [need > 1.0]')
    print(f'  Max MDD : {mdd:.1%}  [need > -25%]')
    gate2 = sharpe > 1.0 and mdd > -0.25
    print(f'  Result  : {"PASS" if gate2 else "FAIL"}')
    print('-' * 58)


if __name__ == '__main__':
    main()
