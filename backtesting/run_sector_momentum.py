#!/usr/bin/env python3
"""
backtesting/run_sector_momentum.py
Phase 1 backtest -- Sector ETF rotation momentum with SPY regime filter.

Usage (from /root/alpha_edge):
    python3 backtesting/run_sector_momentum.py
    python3 backtesting/run_sector_momentum.py --start 2010-01-01 --top-n 3
    python3 backtesting/run_sector_momentum.py --no-regime --top-n 5

Options:
    --start      : start date YYYY-MM-DD (default: 2010-01-01)
    --end        : end date (default: today)
    --capital    : initial capital USD (default: 100000)
    --top-n      : sectors to hold simultaneously (default: 3)
    --no-regime  : disable SPY 200d MA regime filter
    --no-vol-adj : disable volatility adjustment in scoring
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
logger = logging.getLogger('sector_backtest')
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

from backtesting.signals.library.sector_momentum import SECTOR_ETFS, SECTOR_NAMES


def parse_args():
    p = argparse.ArgumentParser(description='AlphaEdge Sector Momentum Backtest')
    p.add_argument('--start',      default='2010-01-01')
    p.add_argument('--end',        default=str(date.today()))
    p.add_argument('--capital',    type=float, default=100_000)
    p.add_argument('--top-n',      type=int,   default=3)
    p.add_argument('--no-regime',  action='store_true')
    p.add_argument('--no-vol-adj', action='store_true')
    p.add_argument('--no-cache',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    logger.info('=' * 60)
    logger.info('AlphaEdge -- Sector ETF Rotation Momentum')
    logger.info('=' * 60)
    logger.info('Sectors     : %s', ', '.join(SECTOR_ETFS))
    logger.info('Period      : %s to %s', args.start, args.end)
    logger.info('Capital     : $%s', '{:,.0f}'.format(args.capital))
    logger.info('Top-N       : %d sectors', args.top_n)
    logger.info('Regime gate : %s', 'OFF' if args.no_regime else 'SPY 200d MA')
    logger.info('Vol-adjust  : %s', 'OFF' if args.no_vol_adj else 'ON')

    from backtesting.signals.library.sector_momentum import SectorMomentumBacktest

    bt = SectorMomentumBacktest(
        initial_capital = args.capital,
        top_n           = args.top_n,
        vol_adjust      = not args.no_vol_adj,
    )
    if args.no_cache:
        bt.loader.cache_enabled = False

    result = bt.run(
        start_date        = args.start,
        end_date          = args.end,
        spy_regime_filter = not args.no_regime,
        print_holdings    = False,
    )

    out = Path('backtesting/results')
    out.mkdir(parents=True, exist_ok=True)

    eq_path = out / ('equity_sector_%s.csv' % args.start[:4])
    result.equity_curve.to_csv(eq_path)
    logger.info('Equity curve -> %s', eq_path)

    if len(result.trades) > 0:
        t_path = out / ('trades_sector_%s.csv' % args.start[:4])
        result.trades.to_csv(t_path, index=False)
        logger.info('Trade log   -> %s', t_path)

    from backtesting.analysis.metrics import returns_from_equity, sharpe_ratio, max_drawdown
    eq     = result.equity_curve['equity']
    ret    = returns_from_equity(eq)
    sharpe = sharpe_ratio(ret)
    mdd    = max_drawdown(eq)

    print('\n' + '-' * 60)
    print('Gate 2 Check (Cost Survival):')
    print('  Sharpe  : %.2f  [need > 1.0]' % sharpe)
    print('  Max MDD : %.1f%%  [need > -25%%]' % (mdd * 100))
    gate2 = sharpe > 1.0 and mdd > -0.25
    print('  Result  : %s' % ('PASS' if gate2 else 'FAIL'))
    print('-' * 60)


if __name__ == '__main__':
    main()
