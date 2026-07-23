"""
Quick Gate 2 diagnostic — runs in ~3 min instead of 40.
Trains ONE model window on recent data, scores all 35 symbols,
prints score distribution to reveal why zero trades occur.

Run from /root/alpha_edge:
    python3 /tmp/diagnose_gate2.py
"""
import sys, logging
sys.path.insert(0, '/root/alpha_edge')
logging.basicConfig(level=logging.WARNING)  # suppress noise

import numpy as np
import pandas as pd

SYMBOLS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','CRM',
    'JPM','BAC','V','MA','GS',
    'JNJ','UNH','ABBV','LLY','MRK','TMO',
    'PG','KO','PEP','WMT','MCD','COST','HD',
    'CVX','CAT','HON','TXN','QCOM','AMD','ACN','IBM',
]
TRAIN_DAYS  = 504
TARGET_DAYS = 21
K_FEATURES  = 20
MIN_TRAIN   = 150

print("Loading data (2020-01-01 to today)...")
from backtesting.data.loader import DataLoader
loader   = DataLoader()
all_data = loader.get_ohlcv(SYMBOLS + ['SPY'], '2020-01-01', str(pd.Timestamp.today().date()))
price_data = {k: v for k, v in all_data.items() if k != 'SPY'}
spy        = all_data.get('SPY')
print(f"Loaded {len(price_data)} symbols\n")

# --- Train on most recent complete window ---
from data.feature_engine    import FeatureEngine
from models.technical_model import TechnicalPredictor
from sklearn.feature_selection import SelectKBest, mutual_info_classif

engine   = FeatureEngine()
segments = []
TRAIN_DATE = pd.Timestamp('2025-01-01')   # score as of Jan 2025

print(f"Training model on data up to {TRAIN_DATE.date()} ...")
for sym, df in price_data.items():
    hist = df[df.index < TRAIN_DATE].copy()
    if len(hist) < TRAIN_DAYS:
        print(f"  SKIP {sym}: only {len(hist)} rows (need {TRAIN_DAYS})")
        continue
    window = hist.tail(TRAIN_DAYS).copy()
    fwd = window['close'].pct_change(TARGET_DAYS).shift(-TARGET_DAYS)
    window['target'] = (fwd > 0).astype(float)
    window.loc[fwd.isna(), 'target'] = np.nan
    try:
        feat = engine.add_all_features(window)
    except Exception as e:
        print(f"  SKIP {sym}: feature gen failed: {e}")
        continue
    feat = feat.dropna(subset=['target'])
    if len(feat) < MIN_TRAIN:
        print(f"  SKIP {sym}: only {len(feat)} usable rows after dropna")
        continue
    feat['_sym'] = sym
    segments.append(feat)

print(f"\n{len(segments)}/{len(SYMBOLS)} symbols contributed to training panel")
if not segments:
    print("ERROR: no training data — check DataLoader and FeatureEngine")
    sys.exit(1)

panel = pd.concat(segments, axis=0)
feat_names = engine.get_feature_names()
avail_feats = [c for c in feat_names if c in panel.columns]
X = panel[avail_feats].replace([np.inf, -np.inf], np.nan).dropna(axis=1)
y = panel.loc[X.index, 'target'].astype(int)

print(f"Panel: {len(X)} rows | {X.shape[1]} features | class balance: {y.mean():.2%} positive")

k = min(K_FEATURES, X.shape[1])
sel = SelectKBest(mutual_info_classif, k=k)
sel.fit(X, y)
sel_cols = [c for c, keep in zip(X.columns, sel.get_support()) if keep]
X_sel = X[sel_cols]

predictor = TechnicalPredictor()
predictor.train(X_sel, y)
print(f"\nPredictor: val_AUC={getattr(predictor,'val_auc',None):.3f} | "
      f"overfit_flagged={predictor.overfit_flagged}")

# --- Score each symbol at TRAIN_DATE ---
print(f"\nScoring {len(price_data)} symbols as of {TRAIN_DATE.date()} ...")
scores = {}
nan_skipped = 0
cov_skipped = 0
err_skipped = 0

for sym, df in price_data.items():
    hist = df[df.index < TRAIN_DATE].copy()
    if len(hist) < 60:
        continue
    try:
        feat = engine.add_all_features(hist)
    except Exception:
        err_skipped += 1
        continue
    if len(feat) == 0:
        continue
    avail = [c for c in sel_cols if c in feat.columns]
    if len(avail) < len(sel_cols) * 0.7:
        cov_skipped += 1
        print(f"  SKIP {sym}: only {len(avail)}/{len(sel_cols)} features available (< 70%)")
        continue
    row = feat.iloc[-1][avail].replace([np.inf, -np.inf], np.nan)
    if row.isna().any():
        nan_skipped += 1
        nan_cols = row[row.isna()].index.tolist()
        print(f"  SKIP {sym}: NaN in {nan_cols[:5]}")
        continue
    X_row = pd.DataFrame([row.values], columns=avail)
    for c in sel_cols:
        if c not in X_row.columns:
            X_row[c] = 0.0
    X_row = X_row[sel_cols]
    try:
        prob = float(predictor.predict(X_row)[0])
        scores[sym] = prob
    except Exception as e:
        err_skipped += 1

print(f"\n--- Score Distribution ---")
print(f"Scored:     {len(scores)}/{len(price_data)} symbols")
print(f"NaN skip:   {nan_skipped}")
print(f"Cov skip:   {cov_skipped}")
print(f"Err skip:   {err_skipped}")
if scores:
    s = sorted(scores.values())
    print(f"Min:  {min(s):.4f}")
    print(f"Max:  {max(s):.4f}")
    print(f"Mean: {np.mean(s):.4f}")
    print(f"Std:  {np.std(s):.4f}")
    above = [(sym, sc) for sym, sc in scores.items() if sc > 0.5]
    print(f"\nAbove 0.5 threshold: {len(above)}/{len(scores)}")
    print("Top 10 scores:")
    for sym, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {sym:6s}: {sc:.4f}  {'SELECTED' if sc > 0.5 else ''}")
