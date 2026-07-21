#!/usr/bin/env python3
"""
backtesting/run_ml_backtest.py
Gate 2 backtest -- ML Ensemble Signal (21-day target, walk-forward).

Signal validated:
  IC=+0.058, t=+4.75, p<0.0001  (Gate 1 IC/t-stat PASS)
  pct_months_positive=52.7%     (Gate 1 pct_pos MARGINAL -- 2.3pp below 55% gate)
  Proceeding to Gate 2: Sharpe > 1.0, MDD > -25%

Usage (from /root/alpha_edge):
    python3 backtesting/run_ml_backtest.py
    python3 backtesting/run_ml_backtest.py --start 2023-01-01 --top-n 8
    python3 backtesting/run_ml_backtest.py --no-regime
    python3 backtesting/run_ml_backtest.py --no-cache

Options:
    --start      : backtest start date YYYY-MM-DD (default: 2022-01-01)
    --end        : end date (default: today)
    --capital    : initial capital USD (default: 100000)
    --top-n      : max simultaneous positions (default: 10)
    --no-regime  : disable SPY 200d MA regime filter
    --no-cache   : re-download all price data

Runtime note:
    First run is slow (trains walk-forward models monthly). Expect 20-40 min.
    Subsequent runs use cached price data and are faster.
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
logger = logging.getLogger('ml_backtest')
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)
# Suppress per-window model quality logs (too verbose in backtest mode)
logging.getLogger('models.technical_model').setLevel(logging.WARNING)

# Same 35-symbol universe used in IC diagnostic
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CRM',
    'JPM',  'BAC',  'V',     'MA',   'GS',
    'JNJ',  'UNH',  'ABBV',  'LLY',  'MRK',  'TMO',
    'PG',   'KO',   'PEP',   'WMT',  'MCD',  'COST', 'HD',
    'CVX',  'CAT',  'HON',
    'TXN',  'QCOM', 'AMD',
    'ACN',  'IBM',  'AMGN',
]


def parse_args():
    p = argparse.ArgumentParser(description='AlphaEdge ML Ensemble Backtest (Gate 2)')
    p.add_argument('--start',     default='2022-01-01',
                   help='Backtest start date (default: 2022-01-01). '
                        'Model needs ~2yr warmup so data loads from 2020-01-01 regardless.')
    p.add_argument('--end',       default=str(date.today()))
    p.add_argument('--capital',   type=float, default=100_000)
    p.add_argument('--top-n',     type=int,   default=10)
    p.add_argument('--no-regime', action='store_true',
                   help='Disable SPY 200d MA regime filter (always invested)')
    p.add_argument('--no-cache',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    print('\n' + '=' * 65)
    print('AlphaEdge -- ML Ensemble Backtest (Gate 2)')
    print('=' * 65)
    print(f'  Symbols      : {len(SYMBOLS)}')
    print(f'  Period       : {args.start} to {args.end}')
    print(f'  Capital      : ${args.capital:,.0f}')
    print(f'  Top-N        : {args.top_n}')
    print(f'  Regime filter: {"OFF" if args.no_regime else "SPY 200d MA"}')
    print(f'  Signal       : ML Ensemble (21-day target, walk-forward)')
    print('=' * 65 + '\n')

    from backtesting.signals.library.ml_signal import MLBacktest

    bt = MLBacktest(
        initial_capital = args.capital,
        max_positions   = args.top_n,
    )
    if args.no_cache:
        bt.loader.cache_enabled = False

    result = bt.run(
        symbols           = SYMBOLS,
        start_date        = args.start,
        end_date          = args.end,
        spy_regime_filter = not args.no_regime,
    )

    # Save outputs
    out = Path('backtesting/results')
    out.mkdir(parents=True, exist_ok=True)

    eq_path = out / f'equity_ml_{args.start[:4]}.csv'
    result.equity_curve.to_csv(eq_path)
    logger.info('Equity curve -> %s', eq_path)

    if len(result.trades) > 0:
        t_path = out / f'trades_ml_{args.start[:4]}.csv'
        result.trades.to_csv(t_path, index=False)
        logger.info('Trade log   -> %s', t_path)

    # Gate 2 check
    from backtesting.analysis.metrics import returns_from_equity, sharpe_ratio, max_drawdown
    eq     = result.equity_curve['equity']
    ret    = returns_from_equity(eq)
    sharpe = sharpe_ratio(ret)
    mdd    = max_drawdown(eq)

    gate2 = sharpe > 1.0 and mdd > -0.25

    print('\n' + '=' * 65)
    print('GATE 2 CHECK (Cost Survival)')
    print('=' * 65)
    print(f'  Sharpe  : {sharpe:+.2f}  [need > 1.0]  {"PASS" if sharpe > 1.0 else "FAIL"}')
    print(f'  Max MDD : {mdd:.1%}    [need > -25%] {"PASS" if mdd > -0.25 else "FAIL"}')
    print(f'  Result  : {"PASS -- proceed to Gate 3 (regime robustness)" if gate2 else "FAIL"}')
    print('=' * 65 + '\n')

    if not gate2:
        print('Next steps if FAIL:')
        print('  Sharpe FAIL : try score-weighted position sizing (--score-weight flag)')
        print('  MDD FAIL    : add volatility targeting overlay (like run_vol_targeted.py)')
        print('  Both FAIL   : signal IC exists but live alpha is being consumed by costs')
        print('                -- reduce top-n to 5, widen hold to 42 days')


if __name__ == '__main__':
    main()
