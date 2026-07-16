#!/usr/bin/env python3
"""
backtesting/diagnose_sue.py
Diagnose the SUE signal - check earnings data quality and signal direction.

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

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'NVDA']

print('\n' + '=' * 65)
print('SUE Signal Diagnostic')
print('=' * 65)

# --- 1. Inspect raw earnings dates from yfinance ---------------------------

print('\n[1] RAW EARNINGS DATES (yfinance.get_earnings_dates)')
print('    Should show ANNOUNCEMENT dates (Jan/Feb/Apr/Jul/Oct)')
print()

for sym in SYMBOLS:
    ticker = yf.Ticker(sym)
    try:
        hist = ticker.get_earnings_dates(limit=24)
    except Exception as e:
        print(f'  {sym}: FAILED ({e})')
        continue

    if hist is None or len(hist) == 0:
        print(f'  {sym}: NO DATA')
        continue

    hist = hist.copy()
    if hasattr(hist.index, 'tz') and hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)

    # Filter to past earnings only (Reported EPS not null)
    if 'Reported EPS' in hist.columns:
        past = hist[hist['Reported EPS'].notna()]
    else:
        past = hist

    print(f'  {sym}: {len(past)} past quarters with actuals')
    print(f'    Columns : {list(hist.columns)}')
    if len(past) > 0:
        dates = sorted(past.index)
        print(f'    Range   : {dates[0].date()} to {dates[-1].date()}')
        print(f'    Last 4  : {[str(d.date()) for d in dates[-4:]]}')
    for col in ['EPS Estimate', 'Reported EPS', 'Surprise(%)']:
        if col in past.columns:
            nn = past[col].notna().sum()
            print(f'    {col}: {nn}/{len(past)} non-null')
    print()

# --- 2. Date type check ---------------------------------------------------

print('\n[2] DATE VALIDATION')
print('    Expected: dates cluster around Jan/Feb/Apr/Jul/Oct (announcements)')
print('    Bad sign: dates cluster around Mar/Jun/Sep/Dec (fiscal ends)')
print()

ticker = yf.Ticker('AAPL')
try:
    hist = ticker.get_earnings_dates(limit=12)
    if hist is not None and len(hist) > 0:
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        if 'Reported EPS' in hist.columns:
            past = hist[hist['Reported EPS'].notna()]
        else:
            past = hist
        months = [d.month for d in past.index]
        print(f'  AAPL announcement months: {sorted(set(months))}')
        print(f'  (Expected: 1,4,7,10 or nearby -- NOT 3,6,9,12)')
        print()
        print('  AAPL last 6 announcement dates:')
        for d in sorted(past.index)[-6:]:
            row = past.loc[d]
            actual = row.get('Reported EPS', 'N/A')
            est    = row.get('EPS Estimate', 'N/A')
            print(f'    {d.date()}  actual={actual}  estimate={est}')
except Exception as e:
    print(f'  AAPL check failed: {e}')

# --- 3. NVDA signal direction check --------------------------------------

print('\n[3] SIGNAL DIRECTION - NVDA (big surprises in 2023-2024)')
print()

ticker = yf.Ticker('NVDA')
try:
    hist = ticker.get_earnings_dates(limit=24)
    if hist is not None and len(hist) > 0:
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        if 'Reported EPS' in hist.columns:
            hist = hist[hist['Reported EPS'].notna()]
        print(hist[['EPS Estimate','Reported EPS','Surprise(%)']].tail(8).to_string())
        print()
        # Compute SUE for most recent
        if 'Reported EPS' in hist.columns and 'EPS Estimate' in hist.columns:
            latest = hist.iloc[-1]
            actual = latest['Reported EPS']
            est    = latest['EPS Estimate']
            trailing_std = hist['Reported EPS'].iloc[:-1].std()
            if pd.notna(actual) and pd.notna(est) and trailing_std > 0:
                sue = (actual - est) / trailing_std
                print(f'  Most recent NVDA SUE: {sue:.3f}')
                if sue > 1.0:
                    print('  Signal direction: CORRECT (high SUE for NVDA)')
                elif sue < -1.0:
                    print('  Signal direction: INVERTED (BUG)')
                else:
                    print('  Signal near zero (post-revision data?)')
except Exception as e:
    print(f'  NVDA check failed: {e}')

# --- 4. Full signal test on known date ------------------------------------

print('\n[4] SIGNAL TEST (2024-09-01, right after NVDA Q2 2024 earnings ~Aug 28)')
print()

try:
    from backtesting.data.loader import DataLoader
    from backtesting.signals.library.earnings_revision import EarningsRevisionSignal

    loader = DataLoader(cache=True)
    signal = EarningsRevisionSignal(decay_days=30, lookback_quarters=8)

    price_data = loader.get_ohlcv(SYMBOLS, '2022-01-01', '2024-10-01')
    earnings   = loader.get_earnings_history(SYMBOLS)

    print(f'  Earnings data loaded for: {list(earnings.keys())}')
    for sym, earn in earnings.items():
        print(f'    {sym}: {len(earn)} rows, range {earn.index[0].date()} to {earn.index[-1].date()}')

    test_date = pd.Timestamp('2024-09-01')
    data_before = {sym: df[df.index < test_date] for sym, df in price_data.items()}

    scores = signal.compute(test_date, data_before, earnings)
    if scores:
        print(f'\n  Scores on {test_date.date()}:')
        for sym, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f'    {sym:<8} {score:+.3f}')
    else:
        print(f'  No scores on {test_date.date()} (no earnings within decay window)')
        print('  Try increasing decay_days or check date alignment above')
except Exception as e:
    import traceback
    print(f'  Signal test failed: {e}')
    traceback.print_exc()

# --- 5. Coverage summary --------------------------------------------------

print('\n[5] COVERAGE SUMMARY')
print()
try:
    from backtesting.data.loader import DataLoader
    loader = DataLoader(cache=True)
    earnings = loader.get_earnings_history(SYMBOLS)
    for sym, earn_df in earnings.items():
        n = len(earn_df)
        has_est = 'estimated_eps' in earn_df.columns and int(earn_df['estimated_eps'].notna().sum())
        d0 = str(earn_df.index.min().date()) if n else 'N/A'
        d1 = str(earn_df.index.max().date()) if n else 'N/A'
        print(f'  {sym:<8} rows={n:<4} with_estimate={has_est:<4} {d0} to {d1}')
except Exception as e:
    print(f'  Coverage check failed: {e}')

print('\n' + '=' * 65)
print('Diagnostic complete.')
print('=' * 65)
