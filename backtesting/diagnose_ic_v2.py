#!/usr/bin/env python3
"""
backtesting/diagnose_ic_v2.py
IC analysis with three targeted fixes:
  1. Entry delay: T+2 and T+3 instead of T+1
  2. Regime filter: only events when SPY > 200d MA
  3. Surprise filter: |surprise_pct| > 3%

Run from /root/alpha_edge:
    python3 backtesting/diagnose_ic_v2.py
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
FWD_DAYS = 20

print('\n' + '=' * 65)
print('IC Analysis v2 -- Entry Delay + Regime + Surprise Filter')
print('=' * 65)

from backtesting.data.loader import DataLoader
loader = DataLoader(cache=True)

print('\nLoading price data + SPY...')
all_syms   = SYMBOLS + ['SPY']
price_data = loader.get_ohlcv(all_syms, START, END)
print(f'  {len(price_data)} symbols')

spy_close  = price_data['SPY']['close']
spy_ma200  = spy_close.rolling(200).mean()

print('Loading earnings...')
earnings = loader.get_earnings_history(SYMBOLS)
print(f'  {len(earnings)} symbols with earnings')

# Build event table with various offsets
records = []

for sym, earn_df in earnings.items():
    if sym not in price_data:
        continue
    prices = price_data[sym]['close']

    for ann_date in earn_df.index:
        past_earn = earn_df[earn_df.index <= ann_date]
        if len(past_earn) < 4:
            continue

        row = earn_df.loc[ann_date]
        actual     = float(row.get('actual_eps', np.nan))
        estimated  = float(row.get('estimated_eps', np.nan))
        surp_pct   = float(row.get('surprise_pct', np.nan))

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

        # SPY regime: is SPY above 200d MA on announcement date?
        spy_dates_before = spy_close.index[spy_close.index <= ann_date]
        if len(spy_dates_before) < 200:
            regime_ok = None
        else:
            spy_val  = float(spy_close.loc[spy_dates_before[-1]])
            spy_ma   = float(spy_close.iloc[:len(spy_dates_before)].rolling(200).mean().iloc[-1])
            regime_ok = spy_val > spy_ma

        # Forward prices at T+1, T+2, T+3 entry
        future_prices = prices[prices.index > ann_date]
        if len(future_prices) < FWD_DAYS + 3:
            continue

        def fwd_ret(entry_offset):
            if len(future_prices) <= entry_offset + FWD_DAYS:
                return np.nan
            ep = float(future_prices.iloc[entry_offset])
            xp = float(future_prices.iloc[entry_offset + FWD_DAYS])
            return (xp / ep - 1)

        records.append({
            'symbol'    : sym,
            'ann_date'  : ann_date,
            'sue'       : sue,
            'surp_pct'  : surp_pct,
            'regime_ok' : regime_ok,
            'fwd_t1'    : fwd_ret(0),   # T+1 open (current approach)
            'fwd_t2'    : fwd_ret(1),   # T+2 open (1-day delay)
            'fwd_t3'    : fwd_ret(2),   # T+3 open (2-day delay)
        })

df = pd.DataFrame(records)
print(f'\n  Total events: {len(df)}')
print(f'  Regime OK (SPY > 200d): {df.regime_ok.sum()} / {df.regime_ok.notna().sum()}')
print(f'  |surprise_pct| > 3%%: {(df.surp_pct.abs() > 3).sum()}')

def ic_report(label, sub, signal_col, ret_col):
    sub2 = sub[[signal_col, ret_col]].dropna()
    if len(sub2) < 10:
        print(f'  {label:<40} N too small ({len(sub2)})')
        return
    ic, pval = stats.spearmanr(sub2[signal_col], sub2[ret_col])
    n   = len(sub2)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print(f'  {label:<40} IC={ic:+.4f}  t={tst:+.2f}  N={n}  {gate}')

print('\n[1] ENTRY DELAY COMPARISON (all events, SUE signal)')
ic_report('T+1 entry (current)', df, 'sue', 'fwd_t1')
ic_report('T+2 entry (1-day delay)', df, 'sue', 'fwd_t2')
ic_report('T+3 entry (2-day delay)', df, 'sue', 'fwd_t3')

print('\n[2] REGIME FILTER (SPY > 200d MA, T+2 entry)')
bull = df[df['regime_ok'] == True]
bear = df[df['regime_ok'] == False]
ic_report('Bull regime only', bull, 'sue', 'fwd_t2')
ic_report('Bear regime only', bear, 'sue', 'fwd_t2')

print('\n[3] SURPRISE FILTER (|surprise_pct| > 3%, T+2 entry, bull only)')
filt = df[(df['regime_ok'] == True) & (df['surp_pct'].abs() > 3)]
ic_report('Bull + |surp| > 3%', filt, 'sue', 'fwd_t2')
ic_report('Bull + |surp| > 3%, surp_pct signal', filt, 'surp_pct', 'fwd_t2')

print('\n[4] QUINTILE RETURNS (bull regime, T+2 entry, all surprises)')
sub = bull[['sue', 'fwd_t2']].dropna().copy()
if len(sub) > 50:
    sub['q'] = pd.qcut(sub['sue'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    qr = sub.groupby('q', observed=True)['fwd_t2'].agg(['mean', 'count'])
    print(f'  {"Q":<6} {"Mean 20d Ret":>14} {"N":>6}')
    for q, row in qr.iterrows():
        print(f'  {str(q):<6} {row["mean"]:>14.2%} {int(row["count"]):>6}')
    q5 = qr.loc['Q5', 'mean']
    q1 = qr.loc['Q1', 'mean']
    print(f'  Q5-Q1 spread: {q5-q1:.2%}')

print('\n[5] IC BY YEAR (bull regime, T+2 entry)')
bull2 = bull.copy()
bull2['year'] = bull2['ann_date'].dt.year
print(f'  {"Year":<8} {"IC":>8} {"t-stat":>8} {"N":>6}')
for yr, grp in bull2.groupby('year'):
    sub = grp[['sue', 'fwd_t2']].dropna()
    if len(sub) < 5:
        continue
    ic, pval = stats.spearmanr(sub['sue'], sub['fwd_t2'])
    n   = len(sub)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    print(f'  {yr:<8} {ic:>8.4f} {tst:>8.2f} {n:>6}')

print('\n' + '=' * 65)
print('IC v2 complete.')
print('=' * 65)
