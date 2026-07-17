#!/usr/bin/env python3
"""
backtesting/diagnose_sector_ic.py
Gate 1 IC validation for sector ETF rotation momentum.

Computes rank IC at 21/42/63 day horizons across all 11 sector ETFs.
Much stronger IC expected vs individual large-cap stocks.

Run from /root/alpha_edge:
    python3 backtesting/diagnose_sector_ic.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from scipy import stats

from backtesting.signals.library.sector_momentum import SECTOR_ETFS, SECTOR_NAMES

START = '2007-01-01'
END   = '2026-07-01'

print('\n' + '=' * 65)
print('IC Analysis -- Sector Rotation Momentum (12-1, vol-adjusted)')
print('=' * 65)

from backtesting.data.loader import DataLoader
loader = DataLoader(cache=True)

print('\nLoading sector ETFs + SPY...')
all_syms   = SECTOR_ETFS + ['SPY']
price_data = loader.get_ohlcv(all_syms, START, END)
spy_close  = price_data['SPY']['close']
spy_ma200  = spy_close.rolling(200).mean()
sector_data = {k: v for k, v in price_data.items() if k in SECTOR_ETFS}
print('  Loaded: %s' % ', '.join(sorted(sector_data.keys())))

LOOKBACK = 252
SKIP     = 21
VOL_WIN  = 63
FWD_DAYS = [21, 42, 63]

# Build monthly scoring table
all_dates = spy_close.index
scoring_dates = all_dates[
    (all_dates >= pd.Timestamp('2009-01-01')) &
    (all_dates.day <= 7)
]

records = []
for ref_date in scoring_dates:
    scores = {}
    for ticker, df in sector_data.items():
        past = df[df.index < ref_date]['close']
        n    = len(past)
        if n < LOOKBACK + SKIP + VOL_WIN:
            continue
        p_start = float(past.iloc[n - LOOKBACK])
        p_end   = float(past.iloc[n - SKIP])
        if p_start <= 0:
            continue
        raw_ret = (p_end / p_start) - 1.0
        recent  = past.iloc[-VOL_WIN:]
        vol     = float(recent.pct_change().dropna().std()) * np.sqrt(252)
        if vol < 1e-6:
            vol = 0.01
        scores[ticker] = raw_ret / vol

    if len(scores) < 5:
        continue

    spy_past = spy_close[spy_close.index < ref_date]
    spy_ma   = spy_ma200[spy_ma200.index < ref_date]
    regime   = None
    if len(spy_past) > 0 and len(spy_ma) > 0:
        sv = float(spy_past.iloc[-1])
        mv = float(spy_ma.iloc[-1])
        if not np.isnan(mv):
            regime = sv > mv

    for ticker, score in scores.items():
        df  = sector_data[ticker]
        fut = df[df.index >= ref_date]['close']
        fwd = {}
        for h in FWD_DAYS:
            fwd[h] = float(fut.iloc[h] / fut.iloc[0] - 1) if len(fut) > h else np.nan
        records.append({
            'date'   : ref_date,
            'ticker' : ticker,
            'sector' : SECTOR_NAMES.get(ticker, ticker),
            'score'  : score,
            'regime' : regime,
            'fwd_21' : fwd[21],
            'fwd_42' : fwd[42],
            'fwd_63' : fwd[63],
        })

df = pd.DataFrame(records)
print('  %d sector-month observations' % len(df))
print('  %d scoring dates' % df.date.nunique())
print('  Regime OK (bull): %d / %d' % ((df.regime == True).sum(), df.regime.notna().sum()))

def ic_row(label, sub, ret_col):
    sub2 = sub[['score', ret_col]].dropna()
    n    = len(sub2)
    if n < 10:
        print('  %-42s N too small' % label)
        return None
    ic, pval = stats.spearmanr(sub2['score'], sub2[ret_col])
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  %-42s IC=%+.4f  t=%+.2f  N=%-5d  %s' % (label, ic, tst, n, gate))
    return ic

print('\n[1] IC BY HORIZON (all events, 2009-present)')
ic_row('1-month forward (21d)', df, 'fwd_21')
ic_row('2-month forward (42d)', df, 'fwd_42')
ic_row('3-month forward (63d)', df, 'fwd_63')

print('\n[2] REGIME SPLIT (1-month forward)')
ic_row('Bull regime (SPY > 200d MA)', df[df.regime == True],  'fwd_21')
ic_row('Bear regime (SPY < 200d MA)', df[df.regime == False], 'fwd_21')

print('\n[3] IC BY YEAR (1-month forward, all events)')
df['year'] = df['date'].dt.year
print('  %-8s %8s %8s %6s %-8s' % ('Year', 'IC', 't-stat', 'N', 'Gate1?'))
for yr, grp in df.groupby('year'):
    sub = grp[['score', 'fwd_21']].dropna()
    if len(sub) < 5:
        continue
    ic, _ = stats.spearmanr(sub['score'], sub['fwd_21'])
    n   = len(sub)
    tst = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  %-8d %8.4f %8.2f %6d %-8s' % (yr, ic, tst, n, gate))

print('\n[4] QUINTILE RETURNS (21d forward, all events)')
sub = df[['score', 'fwd_21']].dropna().copy()
if len(sub) > 20:
    sub['q'] = pd.qcut(sub['score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    qr = sub.groupby('q', observed=True)['fwd_21'].agg(['mean', 'count'])
    print('  %-10s %12s %6s' % ('Quintile', 'Mean 1m Ret', 'N'))
    for q, row in qr.iterrows():
        print('  %-10s %12.2f%% %6d' % (str(q), row['mean'] * 100, int(row['count'])))
    q5 = qr.loc['Q5', 'mean']
    q1 = qr.loc['Q1', 'mean']
    print('  Q5-Q1 spread: %.2f%%' % ((q5 - q1) * 100))
    vals = qr['mean'].values
    mono = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    print('  Monotonic?    %s' % ('YES' if mono else 'NO'))

print('\n[5] GATE 1 SUMMARY')
sub_all = df[['score', 'fwd_21', 'date']].dropna()
ic_all, pval = stats.spearmanr(sub_all['score'], sub_all['fwd_21'])
n   = len(sub_all)
tst = ic_all * np.sqrt(n - 2) / np.sqrt(1 - ic_all**2 + 1e-10)

monthly_ics = []
for dt, grp in sub_all.groupby('date'):
    sub_m = grp[['score', 'fwd_21']].dropna()
    if len(sub_m) >= 5:
        ic_m, _ = stats.spearmanr(sub_m['score'], sub_m['fwd_21'])
        monthly_ics.append(ic_m)
pct_pos = float(np.mean([1 if x > 0 else 0 for x in monthly_ics])) if monthly_ics else 0.0

print('  IC (overall)           : %+.4f  [need > 0.03]' % ic_all)
print('  t-stat                 : %+.2f   [need > 2.0]' % tst)
print('  p-value                : %.4f' % pval)
print('  Pct months positive IC : %.1f%%  [need > 55%%]' % (pct_pos * 100))
gate1 = ic_all > 0.03 and tst > 2.0 and pct_pos > 0.55
print('  Gate 1 result          : %s' % ('PASS' if gate1 else 'FAIL'))

print('\n' + '=' * 65)
print('Sector IC diagnostic complete.')
print('=' * 65)
