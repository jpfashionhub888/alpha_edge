#!/usr/bin/env python3
"""
backtesting/diagnose_ml_ic_v2.py
ML ensemble IC diagnostic -- 21-day target horizon.

Key change from v1:
  v1 trained on 5-day binary target -> IC at 5d = -0.003, at 21d = +0.026
  v2 trains on 21-day binary target -> aligns model to where IC actually exists

Mechanism: FeatureEngine respects a pre-computed 'target' column.
We inject a 21-day target before calling add_all_features(). No live files touched.

Run from /root/alpha_edge:
    python3 backtesting/diagnose_ml_ic_v2.py

Expected runtime: 15-30 minutes.
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
for name in ['lightgbm', 'xgboost', 'catboost', 'sklearn', 'urllib3']:
    logging.getLogger(name).setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CRM',
    'JPM',  'BAC',  'V',     'MA',   'GS',
    'JNJ',  'UNH',  'ABBV',  'LLY',  'MRK',
    'PG',   'KO',   'PEP',   'WMT',  'MCD',  'COST', 'HD',
    'CVX',  'CAT',  'HON',
    'TXN',  'QCOM', 'AMD',
    'ACN',  'IBM',  'AMGN',
]

TARGET_DAYS = 21
TRAIN_DAYS  = 365
MIN_TRAIN   = 150
K_FEATURES  = 20
FWD_DAYS    = [5, 10, 21]

print('\n' + '=' * 65)
print('ML Ensemble IC v2 (21-day training target)')
print('=' * 65)

from backtesting.data.loader import DataLoader
loader = DataLoader(cache=True)

print('\nLoading price data...')
price_data = loader.get_ohlcv(SYMBOLS, '2019-01-01', '2026-07-17')
print('  %d symbols' % len(price_data))

print('Importing FeatureEngine + TechnicalPredictor...')
try:
    from data.feature_engine    import FeatureEngine
    from models.technical_model import TechnicalPredictor
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    print('  OK')
except ImportError as e:
    print('  FAILED: %s' % e)
    sys.exit(1)

all_dates = pd.DatetimeIndex(sorted(set(
    d for df in price_data.values() for d in df.index
)))
eval_dates = all_dates[
    (all_dates >= pd.Timestamp('2022-01-01')) &
    (all_dates.day <= 7)
]
print('  %d evaluation dates' % len(eval_dates))

engine  = FeatureEngine()
records = []
skipped = {'too_short': 0, 'no_class': 0, 'train_fail': 0, 'no_fwd': 0}
total   = len(SYMBOLS) * len(eval_dates)
done    = 0

print('\nRunning walk-forward... ')
print('Progress: ', end='', flush=True)

for sym in SYMBOLS:
    if sym not in price_data:
        continue
    raw_df = price_data[sym].copy()

    for eval_date in eval_dates:
        done += 1
        if done % 100 == 0:
            print('%.0f%%' % (100 * done / total), end=' ', flush=True)

        hist = raw_df[raw_df.index < eval_date].copy()
        if len(hist) < TRAIN_DAYS + MIN_TRAIN:
            skipped['too_short'] += 1
            continue

        train_window = hist.tail(TRAIN_DAYS).copy()

        # Inject 21-day target before calling add_all_features.
        # FeatureEngine checks: if 'target' in df.columns -> skip target generation.
        # The last TARGET_DAYS rows will have NaN (no forward data within window)
        # and will be dropped by the engine's dropna() call.
        fwd_ret = train_window['close'].pct_change(TARGET_DAYS).shift(-TARGET_DAYS)
        train_window['target'] = (fwd_ret > 0).astype(float)
        # Mark rows without forward data as NaN so engine drops them
        train_window.loc[fwd_ret.isna(), 'target'] = np.nan

        try:
            train_feat = engine.add_all_features(train_window)
        except Exception:
            skipped['train_fail'] += 1
            continue

        train_feat = train_feat.dropna(subset=['target'])
        if len(train_feat) < MIN_TRAIN:
            skipped['too_short'] += 1
            continue
        train_feat['target'] = train_feat['target'].astype(int)

        feature_names = engine.get_feature_names()
        if not feature_names:
            skipped['train_fail'] += 1
            continue

        X_train = train_feat[feature_names].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        y_train = train_feat.loc[X_train.index, 'target']

        if len(y_train.unique()) < 2 or y_train.value_counts().min() < 5:
            skipped['no_class'] += 1
            continue

        k = min(K_FEATURES, X_train.shape[1])
        try:
            sel      = SelectKBest(mutual_info_classif, k=k)
            sel.fit(X_train, y_train)
            sel_cols = [c for c, m in zip(X_train.columns, sel.get_support()) if m]
            X_sel    = X_train[sel_cols]
        except Exception:
            sel_cols = list(X_train.columns[:k])
            X_sel    = X_train[sel_cols]

        try:
            model = TechnicalPredictor(use_lstm=False)
            model.train(X_sel, y_train)
            if getattr(model, 'overfit_flagged', False):
                skipped['no_class'] += 1
                continue
        except Exception:
            skipped['train_fail'] += 1
            continue

        # Score: eval_window has no injected target (clean prediction)
        eval_window = hist.tail(TRAIN_DAYS + 10).copy()
        try:
            eval_feat = engine.add_all_features(eval_window)
            if eval_feat.empty:
                continue
            avail = [c for c in sel_cols if c in eval_feat.columns]
            if len(avail) < len(sel_cols) * 0.8:
                continue
            X_eval = eval_feat[avail].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
            if X_eval.isnull().any().any():
                continue
            X_eval = X_eval.reindex(columns=sel_cols, fill_value=0)
            preds  = model.predict(X_eval)
            score  = float(preds[0]) if len(preds) > 0 else 0.5
        except Exception:
            continue

        future = raw_df[raw_df.index >= eval_date]['close']
        if len(future) < max(FWD_DAYS) + 1:
            skipped['no_fwd'] += 1
            continue

        entry = float(future.iloc[0])
        fwd   = {h: float(future.iloc[h] / entry - 1) if len(future) > h else np.nan
                 for h in FWD_DAYS}
        records.append({
            'symbol': sym, 'date': eval_date, 'score': score,
            'fwd_5': fwd[5], 'fwd_10': fwd[10], 'fwd_21': fwd[21],
        })

print('\n  Done. Records=%d  Skipped=%s' % (len(records), skipped))

if len(records) < 20:
    print('Not enough records.')
    sys.exit(1)

df = pd.DataFrame(records)

def ic_row(label, sub, ret_col):
    sub2 = sub[['score', ret_col]].dropna()
    n    = len(sub2)
    if n < 10:
        print('  %-42s N too small' % label)
        return
    ic, pval = stats.spearmanr(sub2['score'], sub2[ret_col])
    tst  = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  %-42s IC=%+.4f  t=%+.2f  N=%-5d  %s' % (label, ic, tst, n, gate))

print('\n[1] IC BY HORIZON')
ic_row('5-day forward',  df, 'fwd_5')
ic_row('10-day forward', df, 'fwd_10')
ic_row('21-day forward', df, 'fwd_21')

print('\n[2] IC BY YEAR (21-day)')
df['year'] = df['date'].dt.year
print('  %-8s %8s %8s %6s %-8s' % ('Year', 'IC', 't-stat', 'N', 'Gate1?'))
for yr, grp in df.groupby('year'):
    sub = grp[['score', 'fwd_21']].dropna()
    if len(sub) < 5:
        continue
    ic, _ = stats.spearmanr(sub['score'], sub['fwd_21'])
    n     = len(sub)
    tst   = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate  = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  %-8d %8.4f %8.2f %6d %-8s' % (yr, ic, tst, n, gate))

print('\n[3] HIGH-CONVICTION FILTER (21-day)')
for thresh in [0.0, 0.55, 0.60, 0.65, 0.70]:
    sub = df[df['score'] > thresh][['score', 'fwd_21']].dropna()
    if len(sub) < 20:
        continue
    ic, pval = stats.spearmanr(sub['score'], sub['fwd_21'])
    n    = len(sub)
    tst  = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  score > %.2f: IC=%+.4f  t=%+.2f  N=%-4d  %s' % (thresh, ic, tst, n, gate))

print('\n[4] QUINTILE RETURNS (21-day)')
sub = df[['score', 'fwd_21']].dropna().copy()
if len(sub) > 20:
    sub['q'] = pd.qcut(sub['score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    qr = sub.groupby('q', observed=True)['fwd_21'].agg(['mean', 'count'])
    print('  %-8s %12s %6s' % ('Quintile', 'Mean 21d Ret', 'N'))
    for q, row in qr.iterrows():
        print('  %-8s %12.2f%% %6d' % (str(q), row['mean'] * 100, int(row['count'])))
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
    if len(sub_m) >= 4:
        ic_m, _ = stats.spearmanr(sub_m['score'], sub_m['fwd_21'])
        monthly_ics.append(ic_m)
pct_pos = float(np.mean([1 if x > 0 else 0 for x in monthly_ics])) if monthly_ics else 0.0

print('  IC (overall)           : %+.4f  [need > 0.03]' % ic_all)
print('  t-stat                 : %+.2f   [need > 2.0]' % tst)
print('  p-value                : %.4f' % pval)
print('  Pct months positive IC : %.1f%%  [need > 55%%]' % (pct_pos * 100))
gate1 = ic_all > 0.03 and tst > 2.0 and pct_pos > 0.55
print('  Gate 1 result          : %s' % ('PASS' if gate1 else 'FAIL'))
print('  v1 baseline            : IC=+0.026  t=+2.18  pct_pos=49.2%%')

df.to_csv('backtesting/results/ml_ic_v2_records.csv', index=False)
print('\n  Records -> backtesting/results/ml_ic_v2_records.csv')
print('\n' + '=' * 65)
print('ML IC v2 complete.')
print('=' * 65)
