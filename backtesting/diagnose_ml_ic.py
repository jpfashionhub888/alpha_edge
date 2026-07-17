#!/usr/bin/env python3
"""
backtesting/diagnose_ml_ic.py
Gate 1 IC validation for the live ML ensemble signal.

Replays the exact signal the live system uses:
  FeatureEngine.add_all_features() -> SelectKBest(k=20) ->
  TechnicalPredictor (XGB + LGB + RF + CatBoost ensemble) ->
  predict_proba -> score

Uses a WALK-FORWARD approach:
  - For each evaluation date (monthly from 2022-present):
    - Train on prior 365 days
    - Predict on current bar
    - Record (prediction, forward_return)
  - Compute IC across all (symbol, date) pairs

Run from /root/alpha_edge:
    python3 backtesting/diagnose_ml_ic.py

Expected runtime: 15-30 minutes (training 40 models x N dates)
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence noisy loggers
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

TRAIN_DAYS   = 365
FWD_DAYS     = [5, 10, 21]
MIN_TRAIN    = 150
EVAL_FREQ    = 'M'     # score monthly
K_FEATURES   = 20

print('\n' + '=' * 65)
print('ML Ensemble IC Analysis (Walk-Forward)')
print('=' * 65)

# ── Load price data ──────────────────────────────────────────────────────────
from backtesting.data.loader import DataLoader
loader = DataLoader(cache=True)

print('\nLoading price data (2019-present)...')
price_data = loader.get_ohlcv(SYMBOLS, '2019-01-01', '2026-07-17')
print('  %d symbols loaded' % len(price_data))

# ── Import live signal components ────────────────────────────────────────────
print('Importing FeatureEngine + TechnicalPredictor...')
try:
    from data.feature_engine    import FeatureEngine
    from models.technical_model import TechnicalPredictor
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    print('  OK')
except ImportError as e:
    print('  FAILED: %s' % e)
    sys.exit(1)

# ── Build evaluation dates (monthly, first trading day) ──────────────────────
all_dates = sorted(set(
    d for df in price_data.values() for d in df.index
))
all_dates = pd.DatetimeIndex(all_dates)
eval_dates = all_dates[
    (all_dates >= pd.Timestamp('2022-01-01')) &
    (all_dates.day <= 7)
]
print('  %d evaluation dates (%s to %s)' % (
    len(eval_dates),
    eval_dates[0].date(),
    eval_dates[-1].date(),
))

# ── Walk-forward scoring ──────────────────────────────────────────────────────
engine  = FeatureEngine()
records = []
skipped = {'too_short': 0, 'no_class': 0, 'train_fail': 0, 'no_fwd': 0}

total_evals = len(SYMBOLS) * len(eval_dates)
done        = 0

print('\nRunning walk-forward evaluation (%d symbol-date pairs)...' % total_evals)
print('Progress: ', end='', flush=True)

for sym in SYMBOLS:
    if sym not in price_data:
        continue
    raw_df = price_data[sym].copy()

    # Add OHLCV column names expected by FeatureEngine
    raw_df.index.name = 'date'

    for eval_date in eval_dates:
        done += 1
        if done % 100 == 0:
            pct = 100 * done / total_evals
            print('%.0f%%' % pct, end=' ', flush=True)

        # Data strictly before eval_date for training
        hist = raw_df[raw_df.index < eval_date].copy()
        if len(hist) < TRAIN_DAYS + MIN_TRAIN:
            skipped['too_short'] += 1
            continue

        # Train window: last TRAIN_DAYS bars
        train_window = hist.tail(TRAIN_DAYS).copy()

        try:
            train_feat = engine.add_all_features(train_window)
        except Exception:
            skipped['train_fail'] += 1
            continue

        feature_names = engine.get_feature_names()
        if not feature_names or 'target' not in train_feat.columns:
            skipped['train_fail'] += 1
            continue

        X_train = train_feat[feature_names].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        y_train = train_feat.loc[X_train.index, 'target']

        if len(y_train.unique()) < 2 or y_train.value_counts().min() < 5:
            skipped['no_class'] += 1
            continue

        # Feature selection
        k = min(K_FEATURES, X_train.shape[1])
        try:
            sel = SelectKBest(mutual_info_classif, k=k)
            sel.fit(X_train, y_train)
            mask     = sel.get_support()
            sel_cols = [c for c, m in zip(X_train.columns, mask) if m]
            X_sel    = X_train[sel_cols]
        except Exception:
            sel_cols = list(X_train.columns[:k])
            X_sel    = X_train[sel_cols]

        # Train model (no LSTM for speed)
        try:
            model = TechnicalPredictor(use_lstm=False)
            model.train(X_sel, y_train)
            if getattr(model, 'overfit_flagged', False):
                skipped['no_class'] += 1
                continue
        except Exception:
            skipped['train_fail'] += 1
            continue

        # Score on eval_date bar
        eval_window = hist.tail(TRAIN_DAYS + 10).copy()
        try:
            eval_feat = engine.add_all_features(eval_window)
            if eval_feat.empty:
                continue
            X_eval = eval_feat[sel_cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]
            if X_eval.isnull().any().any():
                continue
            preds = model.predict(X_eval)
            score = float(preds[0]) if len(preds) > 0 else 0.5
        except Exception:
            continue

        # Forward returns from eval_date
        future = raw_df[raw_df.index >= eval_date]['close']
        if len(future) < max(FWD_DAYS) + 1:
            skipped['no_fwd'] += 1
            continue

        entry = float(future.iloc[0])
        fwd   = {}
        for h in FWD_DAYS:
            fwd[h] = float(future.iloc[h] / entry - 1) if len(future) > h else np.nan

        records.append({
            'symbol'  : sym,
            'date'    : eval_date,
            'score'   : score,
            'fwd_5'   : fwd[5],
            'fwd_10'  : fwd[10],
            'fwd_21'  : fwd[21],
        })

print('\n  Done.')
print('  Records: %d  |  Skipped: %s' % (len(records), skipped))

if len(records) < 20:
    print('\nNot enough records for IC analysis. Check imports and data.')
    sys.exit(1)

df = pd.DataFrame(records)

# ── IC analysis ──────────────────────────────────────────────────────────────
def ic_row(label, sub, ret_col):
    sub2 = sub[['score', ret_col]].dropna()
    n    = len(sub2)
    if n < 10:
        print('  %-40s N too small (%d)' % (label, n))
        return None
    ic, pval = stats.spearmanr(sub2['score'], sub2[ret_col])
    tst  = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    gate = 'PASS' if ic > 0.03 and tst > 2.0 else 'fail'
    print('  %-40s IC=%+.4f  t=%+.2f  N=%-4d  %s' % (label, ic, tst, n, gate))
    return ic

print('\n[1] IC BY HORIZON (all events, 2022-present)')
ic_row('5-day forward  (1 week)', df, 'fwd_5')
ic_row('10-day forward (2 weeks)', df, 'fwd_10')
ic_row('21-day forward (1 month)', df, 'fwd_21')

print('\n[2] IC BY YEAR (21-day forward)')
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

print('\n[3] SCORE DISTRIBUTION')
print('  Score stats:')
print('    mean   : %.3f' % df['score'].mean())
print('    std    : %.3f' % df['score'].std())
print('    min    : %.3f' % df['score'].min())
print('    max    : %.3f' % df['score'].max())
print('    >0.6   : %d (%.0f%%)' % (
    (df['score'] > 0.6).sum(), 100 * (df['score'] > 0.6).mean()
))
print('    >0.7   : %d (%.0f%%)' % (
    (df['score'] > 0.7).sum(), 100 * (df['score'] > 0.7).mean()
))

print('\n[4] QUINTILE RETURNS (21-day forward)')
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

print('\n[5] GATE 1 SUMMARY (21-day horizon)')
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

# Save records for further analysis
out_path = Path('backtesting/results/ml_ic_records.csv')
df.to_csv(out_path, index=False)
print('\n  Records saved -> %s' % out_path)

print('\n' + '=' * 65)
print('ML IC diagnostic complete.')
print('=' * 65)
