#!/usr/bin/env python3
"""
backtesting/diagnose_ic.py
IC analysis — does the SUE signal have positive predictive power?

Run from /root/alpha_edge:
    python3 backtesting/diagnose_ic.py

Computes rank IC at 5, 10, 20, 40 trading day horizons.
Gate 1 target: IC > 0.03, t-stat > 2.0
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from scipy import stats

SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CRM',
    'JPM',  'BAC',  'V',     'MA',   'GS',
    'JNJ',  'UNH',  'ABBV',  'LLY',  'MRK',  'TMO',
    'PG',   'KO',   'PEP',   'WMT',  'MCD',  'COST', 'HD',
    'CVX',  'CAT',  'HON',   'RTX',
    'TXN',  'QCOM', 'AMD',
    'NEE',  'ACN',  'IBM',   'AMGN', 'PM',   'ABT',
]

START = '2018-01-01'
END   = '2026-07-01'

print('\n' + '=' * 65)
print('IC Analysis -- SUE Signal')
print('=' * 65)

from backtesting.data.loader import DataLoader

loader = DataLoader(cache=True)

print('\nLoading price data...')
price_data = loader.get_ohlcv(SYMBOLS, START, END)
print(f'  {len(price_data)} symbols loaded')

print('Loading earnings data...')
earnings   = loader.get_earnings_history(list(price_data.keys()))
print(f'  {len(earnings)} symbols with earnings')

# Build event table: (symbol, ann_date, sue_score, fwd_5, fwd_10, fwd_20, fwd_40)
HORIZONS = [5, 10, 20, 40]
records   = []

for sym, earn_df in earnings.items():
    if sym not in price_data:
        continue
    prices = price_data[sym]['close']

    for ann_date in earn_df.index:
        # Get price data available at announcement (use next trading day open approx)
        past_earn = earn_df[earn_df.index <= ann_date]
        if len(past_earn) < 4:
            continue

        # Compute SUE at this announcement date
        row = earn_df.loc[ann_date]
        actual    = float(row.get('actual_eps', np.nan))
        estimated = float(row.get('estimated_eps', np.nan))

        if np.isnan(actual) or np.isnan(estimated):
            continue

        surprise = actual - estimated
        lookback  = past_earn['actual_eps'].dropna()
        if len(lookback) < 3:
            continue
        eps_std = float(lookback.std())
        if eps_std < 1e-6:
            eps_std = abs(estimated) * 0.1 if abs(estimated) > 0.01 else 0.1

        sue = surprise / eps_std

        # Forward returns starting the day after announcement
        future_prices = prices[prices.index > ann_date]
        if len(future_prices) < max(HORIZONS):
            continue

        entry_price = float(future_prices.iloc[0])
        fwd = {}
        for h in HORIZONS:
            if len(future_prices) > h:
                exit_price = float(future_prices.iloc[h])
                fwd[h] = (exit_price / entry_price - 1)
            else:
                fwd[h] = np.nan

        records.append({
            'symbol'  : sym,
            'ann_date': ann_date,
            'sue'     : sue,
            'fwd_5'   : fwd[5],
            'fwd_10'  : fwd[10],
            'fwd_20'  : fwd[20],
            'fwd_40'  : fwd[40],
        })

df = pd.DataFrame(records)
print(f'\n  Total events: {len(df)}')
print(f'  Date range:   {df.ann_date.min().date()} to {df.ann_date.max().date()}')
print(f'  Symbols:      {df.symbol.nunique()}')

# ── IC at each horizon ──────────────────────────────────────────────────────
print('\n[1] RANK IC BY HORIZON')
print(f'  {"Horizon":<10} {"IC":>8} {"t-stat":>8} {"p-value":>10} {"N":>6} {"Gate1?":>8}')
print('  ' + '-' * 55)

for h in HORIZONS:
    col = f'fwd_{h}'
    sub = df[['sue', col]].dropna()
    if len(sub) < 10:
        print(f'  {h}d{"":<7} {"N/A":>8}')
        continue
    ic, pval = stats.spearmanr(sub['sue'], sub[col])
    n   = len(sub)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'FAIL'
    print(f'  {h}d{"":<7} {ic:>8.4f} {tst:>8.2f} {pval:>10.4f} {n:>6} {gate:>8}')

# ── IC by year ──────────────────────────────────────────────────────────────
print('\n[2] 20-DAY IC BY YEAR')
df['year'] = df['ann_date'].dt.year
print(f'  {"Year":<8} {"IC":>8} {"N":>6}')
print('  ' + '-' * 25)
for yr, grp in df.groupby('year'):
    sub = grp[['sue', 'fwd_20']].dropna()
    if len(sub) < 5:
        continue
    ic, _ = stats.spearmanr(sub['sue'], sub['fwd_20'])
    print(f'  {yr:<8} {ic:>8.4f} {len(sub):>6}')

# ── SUE quintile returns ─────────────────────────────────────────────────────
print('\n[3] 20-DAY FORWARD RETURN BY SUE QUINTILE')
sub = df[['sue', 'fwd_20']].dropna()
sub = sub.copy()
sub['quintile'] = pd.qcut(sub['sue'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
qreturns = sub.groupby('quintile', observed=True)['fwd_20'].agg(['mean', 'median', 'count'])
qreturns.columns = ['mean_ret', 'median_ret', 'count']
print(f'  {"Quintile":<12} {"Mean Ret":>10} {"Median Ret":>12} {"N":>6}')
print('  ' + '-' * 45)
for q, row in qreturns.iterrows():
    print(f'  {str(q):<12} {row.mean_ret:>10.2%} {row.median_ret:>12.2%} {row.count:>6.0f}')

q5_mean = qreturns.loc['Q5(high)', 'mean_ret']
q1_mean = qreturns.loc['Q1(low)', 'mean_ret']
spread  = q5_mean - q1_mean
print(f'\n  Q5-Q1 spread (long-short): {spread:.2%}')
print(f'  Monotonic Q1-Q5?  ', end='')
means = qreturns['mean_ret'].values
monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
print('YES' if monotonic else 'NO')

# ── Large surprise filter ─────────────────────────────────────────────────────
print('\n[4] IC FILTERED TO LARGE SURPRISES (|SUE| > 1.0)')
big = df[df['sue'].abs() > 1.0]
sub = big[['sue', 'fwd_20']].dropna()
print(f'  Events with |SUE| > 1.0: {len(sub)} ({100*len(sub)/len(df):.0f}% of total)')
if len(sub) > 10:
    ic, pval = stats.spearmanr(sub['sue'], sub['fwd_20'])
    n   = len(sub)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    print(f'  IC={ic:.4f}  t-stat={tst:.2f}  p={pval:.4f}')

print('\n' + '=' * 65)
print('IC Diagnostic complete.')
print('=' * 65)
