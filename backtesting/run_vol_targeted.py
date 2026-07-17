#!/usr/bin/env python3
"""
backtesting/run_vol_targeted.py
12-1 momentum with volatility targeting overlay.

Volatility targeting: scale position sizes so the portfolio targets
a fixed annualized volatility (default 15%). When markets are calm,
hold full position. When vol spikes, reduce exposure proportionally.

This improves Sharpe by smoothing the return distribution without
requiring any new signal or data source.

Usage (from /root/alpha_edge):
    python3 backtesting/run_vol_targeted.py
    python3 backtesting/run_vol_targeted.py --target-vol 0.12 --vol-window 21
    python3 backtesting/run_vol_targeted.py --no-vol-target  # baseline comparison
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
logger = logging.getLogger('vol_target_backtest')
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

import numpy as np
import pandas as pd

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
    p = argparse.ArgumentParser(description='Vol-Targeted Momentum Backtest')
    p.add_argument('--start',        default='2018-01-01')
    p.add_argument('--end',          default=str(date.today()))
    p.add_argument('--capital',      type=float, default=100_000)
    p.add_argument('--top-n',        type=int,   default=10)
    p.add_argument('--target-vol',   type=float, default=0.15,
                   help='Target annualized portfolio vol (default 0.15 = 15%%)')
    p.add_argument('--vol-window',   type=int,   default=21,
                   help='Realized vol estimation window in days (default 21)')
    p.add_argument('--max-leverage', type=float, default=1.0,
                   help='Cap on position scale (1.0 = long-only, no leverage)')
    p.add_argument('--no-vol-target', action='store_true',
                   help='Disable vol targeting (baseline comparison)')
    p.add_argument('--no-cache',     action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    logger.info('=' * 60)
    logger.info('AlphaEdge -- 12-1 Momentum + Volatility Targeting')
    logger.info('=' * 60)
    logger.info('Symbols      : %d', len(SYMBOLS))
    logger.info('Period       : %s to %s', args.start, args.end)
    logger.info('Top-N        : %d', args.top_n)
    logger.info('Target vol   : %s', 'OFF (baseline)' if args.no_vol_target else '%.0f%%' % (args.target_vol * 100))
    logger.info('Vol window   : %d days', args.vol_window)
    logger.info('Max leverage : %.1fx', args.max_leverage)

    from backtesting.data.loader         import DataLoader
    from backtesting.engine.event_driven import EventDrivenBacktester
    from backtesting.engine.fill_model   import FillModel
    from backtesting.engine.cost_model   import TransactionCostModel
    from backtesting.signals.library.price_momentum import MomentumSignal

    loader = DataLoader(cache=not args.no_cache)
    signal = MomentumSignal(lookback_days=252, skip_days=21, market_adjust=True)
    engine = EventDrivenBacktester(
        initial_capital = args.capital,
        fill_model      = FillModel(spread_bps=5.0, market_impact_factor=0.1),
        cost_model      = TransactionCostModel(),
    )

    logger.info('Loading price data...')
    fetch_start = '2017-01-01'
    all_data    = loader.get_ohlcv(SYMBOLS + ['SPY'], fetch_start, args.end)
    price_data  = {k: v for k, v in all_data.items() if k in SYMBOLS}
    spy_prices  = all_data.get('SPY')
    logger.info('Loaded %d symbols', len(price_data))

    # Pre-compute SPY realized vol for vol targeting
    spy_close   = spy_prices['close']
    spy_returns = spy_close.pct_change().fillna(0)

    target_vol   = args.target_vol
    vol_window   = args.vol_window
    max_leverage = args.max_leverage
    top_n        = args.top_n
    use_vol_tgt  = not args.no_vol_target

    def signal_fn(date, data):
        # Momentum signal: top-N stocks by 12-1 momentum
        scores = signal.compute(date, data)
        if not scores:
            return {}
        n   = min(top_n, len(scores))
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        base_weights = {sym: 1.0 / n for sym, _ in top}

        if not use_vol_tgt:
            return base_weights

        # Vol targeting: estimate recent portfolio vol from SPY as proxy
        past_spy_ret = spy_returns[spy_returns.index < date]
        if len(past_spy_ret) < vol_window + 5:
            return base_weights

        recent_vol = float(past_spy_ret.iloc[-vol_window:].std()) * np.sqrt(252)
        if recent_vol < 1e-6:
            return base_weights

        # Scale factor: target_vol / realized_vol, capped at max_leverage
        scale = min(target_vol / recent_vol, max_leverage)

        # Apply scale to all weights (reduces position sizes in high-vol periods)
        return {sym: w * scale for sym, w in base_weights.items()}

    logger.info('Running backtest %s to %s...', args.start, args.end)
    result = engine.run(
        price_data    = price_data,
        signal_fn     = signal_fn,
        start_date    = args.start,
        end_date      = args.end,
        max_positions = top_n,
        rebalance_freq= 'M',
    )

    out = Path('backtesting/results')
    out.mkdir(parents=True, exist_ok=True)
    suffix = 'baseline' if args.no_vol_target else 'vol_targeted'
    eq_path = out / ('equity_%s_%s.csv' % (suffix, args.start[:4]))
    result.equity_curve.to_csv(eq_path)
    logger.info('Equity curve -> %s', eq_path)
    if len(result.trades) > 0:
        t_path = out / ('trades_%s_%s.csv' % (suffix, args.start[:4]))
        result.trades.to_csv(t_path, index=False)

    from backtesting.analysis.metrics import (
        returns_from_equity, sharpe_ratio, max_drawdown,
        performance_summary, print_summary,
    )
    eq      = result.equity_curve['equity']
    ret     = returns_from_equity(eq)
    sharpe  = sharpe_ratio(ret)
    mdd     = max_drawdown(eq)
    summary = performance_summary(eq, result.trades)
    print_summary(summary)

    print('\n' + '-' * 60)
    print('Gate 2 Check:')
    print('  Sharpe  : %.2f  [need > 1.0]' % sharpe)
    print('  Max MDD : %.1f%%  [need > -25%%]' % (mdd * 100))
    gate2 = sharpe > 1.0 and mdd > -0.25
    print('  Result  : %s' % ('PASS' if gate2 else 'FAIL'))
    print('-' * 60)


if __name__ == '__main__':
    main()
