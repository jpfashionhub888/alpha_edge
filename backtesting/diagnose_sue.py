#!/usr/bin/env python3
"""
backtesting/diagnose_sue.py
Diagnose the SUE signal — check data quality and signal direction.

Run from /root/alpha_edge:
    python3 backtesting/diagnose_sue.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'NVDA']

print('\n' + '=' * 65)
print('SUE Signal Diagnostic')
print('=' * 65)

# ── 1. Inspect raw earnings_history from yfinance ────────────────────────────

print('\n[1] RAW EARNINGS HISTORY (yfinance)')
print('    Checking if index = announcement date or fiscal quarter end')
print()

for sym in SYMBOLS:
    ticker = yf.Ticker(sym)
    hist = ticker.earnings_history
    if hist is None or len(hist) == 0:
        print(f'  {sym}: NO DATA')
        continue

    hist = hist.copy()
    hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

    print(f'  {sym} ({len(hist)} rows):')
    print(f'    Columns : {list(hist.columns)}')
    print(f'    Index   : {list(hist.index[-4:])}')  # last 4 dates

    # Check if columns have expected data
    for col in ['epsactual', 'epsestimate', 'surprisepct']:
        if col in hist.columns:
            nulls = hist[col].isna().sum()
            print(f'    {col}: {len(hist) - nulls}/{len(hist)} non-null')
    print()

# ── 2. Check date type — announcement vs fiscal end ──────────────────────────

print('\n[2] DATE VALIDATION — Is this announcement date or fiscal quarter end?')
print('    Known AAPL Q1 2024 announcement: 2024-02-01')
print('    Known AAPL Q4 2023 fiscal end:   2023-12-30')
print()

ticker = yf.Ticker('AAPL')
hist = ticker.earnings_history
if hist is not None and len(hist) > 0:
    print('  AAPL last 6 earnings index dates:')
    for d in hist.index[-6:]:
        print(f'    {d}')
    print()
    print('  If dates cluster around Mar/Jun/Sep/Dec → FISCAL QUARTER END (BAD)')
    print('  If dates cluster around Jan/Feb/Apr/Jul/Oct → ANNOUNCEMENT DATE (GOOD)')
else:
    print('  AAPL: no earnings history available')

# ── 3. Check signal direction for a known good period ─────────────────────────

print('\n[3] SIGNAL DIRECTION CHECK')
print('    NVDA had massive positive earnings surprises in 2023-2024.')
print('    The signal should give NVDA a HIGH positive score after those reports.')
print()

ticker = yf.Ticker('NVDA')
hist = ticker.earnings_history
if hist is not None and len(hist) > 0:
    hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]
    col_map = {
        'epsactual': 'actual_eps',
        'epsestimate': 'estimated_eps',
        'epsdifference': 'surprise',
        'surprisepct': 'surprise_pct',
    }
    hist = hist.rename(columns={k: v for k, v in col_map.items() if k in hist.columns})
    print('  NVDA recent earnings:')
    print(hist[['actual_eps', 'estimated_eps', 'surprise_pct']].tail(8).to_string())
    print()

    # Compute SUE for most recent entry
    if 'actual_eps' in hist.columns and 'estimated_eps' in hist.columns:
        latest = hist.iloc[-1]
        actual = latest.get('actual_eps', np.nan)
        est    = latest.get('estimated_eps', np.nan)
        trailing = hist['actual_eps'].dropna().iloc[:-1]
        eps_std = trailing.std() if len(trailing) > 1 else 0.1
        if not np.isnan(actual) and not np.isnan(est) and eps_std > 0:
            sue = (actual - est) / eps_std
            print(f'  SUE for most recent NVDA: {sue:.3f}')
            print(f'    actual={actual}, estimated={est}, eps_std={eps_std:.3f}')
            if sue > 1.0:
                print('  → Signal direction: CORRECT (high SUE for NVDA)')
            elif sue < -1.0:
                print('  → Signal direction: INVERTED (low SUE for NVDA — BUG)')
            else:
                print('  → Signal is near zero for NVDA (data may be post-revision)')
        else:
            print(f'  Cannot compute SUE: actual={actual}, est={est}, std={eps_std:.3f}')
else:
    print('  NVDA: no earnings history')

# ── 4. Score distribution on a known date ─────────────────────────────────────

print('\n[4] SCORE DISTRIBUTION (2024-08-01, right after NVDA blowout Q2 2024)')
print()

from backtesting.data.loader import DataLoader
from backtesting.signals.library.earnings_revision import EarningsRevisionSignal

loader = DataLoader(cache=True)
signal = EarningsRevisionSignal(decay_days=30, lookback_quarters=8)

price_data = loader.get_ohlcv(SYMBOLS, '2022-01-01', '2024-09-01')
earnings   = loader.get_earnings_history(SYMBOLS)

test_date  = pd.Timestamp('2024-08-01')
data_before = {sym: df[df.index < test_date] for sym, df in price_data.items()}

scores = signal.compute(test_date, data_before, earnings)
if scores:
    print(f'  Scores on {test_date.date()} (NVDA blowout Q2 was ~2024-08-28, so none yet):')
    for sym, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f'    {sym:<8} {score:+.3f}')
else:
    print('  No scores computed for 2024-08-01 (no earnings within decay window)')

# Try a date right after NVDA's August 2024 earnings
test_date2 = pd.Timestamp('2024-09-01')
data_before2 = {sym: df[df.index < test_date2] for sym, df in price_data.items()}
scores2 = signal.compute(test_date2, data_before2, earnings)
if scores2:
    print(f'\n  Scores on {test_date2.date()} (right after NVDA Q2 2024 ~2024-08-28):')
    for sym, score in sorted(scores2.items(), key=lambda x: x[1], reverse=True):
        print(f'    {sym:<8} {score:+.3f}')
else:
    print(f'\n  No scores on {test_date2.date()} either')
    print('  → earnings_history dates may be fiscal quarter ends, not announcement dates')

# ── 5. Coverage summary ──────────────────────────────────────────────────────

print('\n[5] EARNINGS DATA COVERAGE SUMMARY')
print()
for sym, earn_df in earnings.items():
    n_rows = len(earn_df)
    has_estimate = 'estimated_eps' in earn_df.columns and earn_df['estimated_eps'].notna().sum()
    date_range = f"{earn_df.index[0].date()} → {earn_df.index[-1].date()}" if n_rows else 'N/A'
    print(f'  {sym:<8} rows={n_rows:<4} estimate_cols={has_estimate:<5} range={date_range}')

print('\n' + '=' * 65)
print('Diagnostic complete.')
print('=' * 65)
