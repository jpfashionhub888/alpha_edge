#!/usr/bin/env python3
"""
backtesting/diagnose_momentum_ic.py
Gate 1 IC validation for the 12-1 momentum signal.

Run from /root/alpha_edge:
    python3 backtesting/diagnose_momentum_ic.py
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

START = '2017-01-01'
END   = '2026-07-01'

print('\n' + '=' * 65)
print('IC Analysis -- 12-1 Momentum Signal')
print('=' * 65)

from backtesting.data.loader import DataLoader
loader = DataLoader(cache=True)

print('\nLoading price data...')
all_syms   = SYMBOLS + ['SPY']
price_data = loader.get_ohlcv(all_syms, START, END)
spy_close  = price_data['SPY']['close']
spy_ma200  = spy_close.rolling(200).mean()
price_data = {k: v for k, v in price_data.items() if k != 'SPY'}
print(f'  {len(price_data)} symbols, SPY loaded')

# Score stocks monthly and collect (score, fwd_return) pairs
LOOKBACK   = 252
SKIP       = 21
FWD_DAYS   = [21, 42, 63]   # 1m, 2m, 3m forward returns

# Build monthly scoring dates
all_dates  = spy_close.index
monthly    = all_dates[all_dates >= pd.Timestamp(START)]
monthly    = monthly[monthly.day <= 7]   # approx first week of each month

records = []
for ref_date in monthly:
    scores = {}
    for sym, df in price_data.items():
        past = df[df.index < ref_date]['close']
        if len(past) < LOOKBACK + SKIP + 20:
            continue
        n = len(past)
        p_start = float(past.iloc[n - LOOKBACK])
        p_end   = float(past.iloc[n - SKIP])
        if p_start <= 0:
            continue
        scores[sym] = p_end / p_start - 1.0

    if len(scores) < 10:
        continue

    # Market adjust
    mkt = float(np.mean(list(scores.values())))
    scores = {k: v - mkt for k, v in scores.items()}

    # Z-score
    vals = np.array(list(scores.values()))
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if sd < 1e-8:
        continue
    scores = {k: (v - mu) / sd for k, v in scores.items()}

    # SPY regime
    spy_past = spy_close[spy_close.index < ref_date]
    spy_ma   = spy_ma200[spy_ma200.index < ref_date]
    regime   = None
    if len(spy_past) > 200 and len(spy_ma) > 0:
        regime = float(spy_past.iloc[-1]) > float(spy_ma.iloc[-1])

    for sym, score in scores.items():
        df = price_data[sym]
        fut = df[df.index >= ref_date]['close']
        fwd = {}
        for h in FWD_DAYS:
            fwd[h] = float(fut.iloc[h] / fut.iloc[0] - 1) if len(fut) > h else np.nan
        records.append({
            'date'    : ref_date,
            'symbol'  : sym,
            'score'   : score,
            'regime'  : regime,
            'fwd_21'  : fwd[21],
            'fwd_42'  : fwd[42],
            'fwd_63'  : fwd[63],
        })

df = pd.DataFrame(records)
print(f'  {len(df)} symbol-month observations')
print(f'  {df.date.nunique()} scoring dates')
print(f'  Regime OK (bull): {(df.regime == True).sum()} / {df.regime.notna().sum()}')

def ic_line(label, sub, ret_col):
    sub2 = sub[['score', ret_col]].dropna()
    if len(sub2) < 20:
        print(f'  {label:<42} N too small')
        return None
    ic, pval = stats.spearmanr(sub2['score'], sub2[ret_col])
    n   = len(sub2)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print(f'  {label:<42} IC={ic:+.4f}  t={tst:+.2f}  N={n}  {gate}')
    return ic

print('\n[1] IC BY HORIZON (all events)')
ic_line('1-month forward (21d)', df, 'fwd_21')
ic_line('2-month forward (42d)', df, 'fwd_42')
ic_line('3-month forward (63d)', df, 'fwd_63')

print('\n[2] REGIME SPLIT (1-month forward)')
ic_line('Bull regime (SPY > 200d)', df[df.regime == True],  'fwd_21')
ic_line('Bear regime (SPY < 200d)', df[df.regime == False], 'fwd_21')

print('\n[3] IC BY YEAR (1-month forward, all events)')
df['year'] = df['date'].dt.year
print(f'  {"Year":<8} {"IC":>8} {"t-stat":>8} {"N":>6} {"Gate1?":>8}')
for yr, grp in df.groupby('year'):
    sub = grp[['score', 'fwd_21']].dropna()
    if len(sub) < 10:
        continue
    ic, _ = stats.spearmanr(sub['score'], sub['fwd_21'])
    n   = len(sub)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print(f'  {yr:<8} {ic:>8.4f} {tst:>8.2f} {n:>6} {gate:>8}')

print('\n[4] QUINTILE RETURNS (21d forward, all events)')
sub = df[['score', 'fwd_21']].dropna().copy()
if len(sub) > 50:
    sub['q'] = pd.qcut(sub['score'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
    qr = sub.groupby('q', observed=True)['fwd_21'].agg(['mean', 'count'])
    print(f'  {"Quintile":<12} {"Mean 1m Ret":>12} {"N":>6}')
    for q, row in qr.iterrows():
        print(f'  {str(q):<12} {row["mean"]:>12.2%} {int(row["count"]):>6}')
    q5 = qr.loc['Q5(high)', 'mean']
    q1 = qr.loc['Q1(low)', 'mean']
    print(f'  Q5-Q1 spread: {q5-q1:.2%}')
    vals = qr['mean'].values
    mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    print(f'  Monotonic?    {"YES" if mono else "NO"}')

print('\n[5] GATE 1 SUMMARY')
sub_all  = df[['score', 'fwd_21']].dropna()
ic_all, pval = stats.spearmanr(sub_all['score'], sub_all['fwd_21'])
n   = len(sub_all)
tst = ic_all * np.sqrt(n - 2) / np.sqrt(1 - ic_all**2 + 1e-10)
pct_pos = (sub_all.groupby(df.loc[sub_all.index, 'date'])
           .apply(lambda g: float(stats.spearmanr(g['score'], g['fwd_21'])[0]))
           .gt(0).mean()) if n > 0 else 0
print(f'  IC (overall): {ic_all:+.4f}')
print(f'  t-stat      : {tst:+.2f}')
print(f'  p-value     : {pval:.4f}')
pct_pos_ic = (df.groupby('date')[['score', 'fwd_21']].apply(
    lambda g: float(stats.spearmanr(g['score'].values, g['fwd_21'].dropna().reindex(g.index).values)[0]) if g['fwd_21'].notna().sum() > 3 else np.nan
).dropna().gt(0).mean())
print(f'  Pct months positive IC: {pct_pos_ic:.1%}  [need > 55%]')
gate1 = ic_all > 0.03 and tst > 2.0 and pct_pos_ic > 0.55
print(f'  Gate 1 result: {"PASS" if gate1 else "FAIL"}')

print('\n' + '=' * 65)
print('Momentum IC diagnostic complete.')
print('=' * 65)
