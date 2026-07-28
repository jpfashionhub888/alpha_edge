"""
monitoring/ic_tracker.py
Live IC (Information Coefficient) tracker.

Each run (called from deep_audit.py at 06:00 UTC daily):
  1. SNAPSHOT: save today's predictions + prices to logs/ic_log.jsonl
  2. COMPUTE:  find snapshot from ~5 trading days ago, fetch current prices,
               compute forward return, compute Spearman IC vs predictions
  3. LOG:      append IC result to logs/ic_metrics.jsonl
  4. ALERT:    Telegram warning if rolling 30-day IC < 0.03 (Gate 1 floor)

Backtest reference: IC=0.058, t-stat=4.75 (Gate 1 PASS)
"""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).resolve().parent.parent
SIGNALS_FILE   = BASE_DIR / 'logs' / 'latest_signals.json'
IC_LOG_FILE    = BASE_DIR / 'logs' / 'ic_log.jsonl'
IC_METRICS_FILE= BASE_DIR / 'logs' / 'ic_metrics.jsonl'

HOLD_DAYS      = 5      # forward return window (trading days)
GATE1_IC_FLOOR = 0.03   # alert threshold — below this = model degrading
BACKTEST_IC    = 0.058  # reference IC from Gate 1 backtest


# ---------------------------------------------------------------------------
# Step 1: Snapshot today's predictions
# ---------------------------------------------------------------------------
def snapshot_predictions():
    """Read latest_signals.json and append to ic_log.jsonl."""
    if not SIGNALS_FILE.exists():
        logger.warning('ic_tracker: latest_signals.json not found — skip snapshot')
        return 0

    try:
        with open(SIGNALS_FILE) as f:
            signals = json.load(f)
    except Exception as e:
        logger.warning('ic_tracker: failed to read latest_signals.json: %s', e)
        return 0

    today = datetime.utcnow().strftime('%Y-%m-%d')
    rows_written = 0
    with open(IC_LOG_FILE, 'a') as out:
        for symbol, data in signals.items():
            pred  = data.get('prediction')
            price = data.get('price')
            if pred is None or price is None:
                continue
            row = {
                'date'      : today,
                'symbol'    : symbol,
                'prediction': float(pred),
                'price'     : float(price),
            }
            out.write(json.dumps(row) + '\n')
            rows_written += 1

    logger.info('ic_tracker: snapshot saved (%d symbols) for %s', rows_written, today)
    return rows_written


# ---------------------------------------------------------------------------
# Step 2: Compute IC for the cohort from ~5 trading days ago
# ---------------------------------------------------------------------------
def _load_snapshot_for_date(target_date_str):
    """Return list of {symbol, prediction, price} for a specific date."""
    if not IC_LOG_FILE.exists():
        return []
    rows = []
    with open(IC_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get('date') == target_date_str:
                    rows.append(r)
            except Exception:
                continue
    return rows


def _fetch_current_prices(symbols):
    """Fetch latest close prices for a list of symbols via yfinance."""
    try:
        import yfinance as yf
        tickers = yf.download(
            symbols,
            period='5d',
            progress=False,
            auto_adjust=True,
        )
        if tickers.empty:
            return {}
        close = tickers['Close'] if 'Close' in tickers.columns else tickers
        latest = close.iloc[-1]
        return {sym: float(latest[sym]) for sym in symbols if sym in latest.index and not np.isnan(latest[sym])}
    except Exception as e:
        logger.warning('ic_tracker: price fetch failed: %s', e)
        return {}


def _trading_days_ago(n):
    """Return the date string for approximately n trading days ago."""
    # Simple approximation: skip weekends
    d = datetime.utcnow().date()
    skipped = 0
    while skipped < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            skipped += 1
    return d.strftime('%Y-%m-%d')


def compute_ic():
    """
    Find snapshot from HOLD_DAYS ago, fetch current prices,
    compute Spearman IC, append to ic_metrics.jsonl.
    Returns IC value or None if insufficient data.
    """
    from scipy.stats import spearmanr

    target_date = _trading_days_ago(HOLD_DAYS)
    rows = _load_snapshot_for_date(target_date)

    if len(rows) < 10:
        logger.info(
            'ic_tracker: not enough symbols for IC on %s (%d found, need 10)',
            target_date, len(rows)
        )
        return None

    symbols  = [r['symbol'] for r in rows]
    prices_now = _fetch_current_prices(symbols)

    pairs = []
    for r in rows:
        sym = r['symbol']
        p0  = r['price']
        p1  = prices_now.get(sym)
        if p1 is None or p0 == 0:
            continue
        fwd_return = (p1 - p0) / p0
        pairs.append((r['prediction'], fwd_return))

    if len(pairs) < 10:
        logger.info('ic_tracker: too few price matches (%d) — skip IC', len(pairs))
        return None

    preds   = np.array([p[0] for p in pairs])
    returns = np.array([p[1] for p in pairs])
    ic, pval = spearmanr(preds, returns)
    ic = float(ic)

    result = {
        'date'          : datetime.utcnow().strftime('%Y-%m-%d'),
        'cohort_date'   : target_date,
        'ic'            : ic,
        'pval'          : float(pval),
        'n_symbols'     : len(pairs),
        'backtest_ic'   : BACKTEST_IC,
        'vs_backtest'   : round(ic - BACKTEST_IC, 4),
    }

    with open(IC_METRICS_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')

    logger.info(
        'ic_tracker: IC=%.4f (p=%.3f, n=%d) vs backtest=%.3f | delta=%+.4f',
        ic, pval, len(pairs), BACKTEST_IC, ic - BACKTEST_IC
    )
    return ic


# ---------------------------------------------------------------------------
# Step 3: Rolling IC stats + alert
# ---------------------------------------------------------------------------
def rolling_ic_stats(window_days=30):
    """Compute rolling mean IC over last window_days calendar days."""
    if not IC_METRICS_FILE.exists():
        return None

    cutoff = (datetime.utcnow() - timedelta(days=window_days)).strftime('%Y-%m-%d')
    ics = []
    with open(IC_METRICS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get('date', '') >= cutoff:
                    ics.append(r['ic'])
            except Exception:
                continue

    if not ics:
        return None

    return {
        'window_days': window_days,
        'n_obs'      : len(ics),
        'mean_ic'    : float(np.mean(ics)),
        'std_ic'     : float(np.std(ics)),
        'min_ic'     : float(np.min(ics)),
        'max_ic'     : float(np.max(ics)),
    }


def check_and_alert(ic, stats):
    """Send Telegram alert if IC is degrading."""
    if ic is None:
        return

    alerts = []
    if ic < GATE1_IC_FLOOR:
        alerts.append(
            f'DAILY IC={ic:.4f} below gate floor ({GATE1_IC_FLOOR})'
        )
    if stats and stats['n_obs'] >= 5 and stats['mean_ic'] < GATE1_IC_FLOOR:
        alerts.append(
            f'ROLLING {stats["window_days"]}d IC={stats["mean_ic"]:.4f} below gate floor'
        )

    if not alerts:
        return

    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from monitoring.telegram_bot import TelegramBot
        bot = TelegramBot()
        msg = (
            'WARNING: AlphaEdge IC Degradation\n\n'
            + '\n'.join(alerts)
            + f'\n\nBacktest reference IC: {BACKTEST_IC}'
            + '\nAction: review model quality, consider retraining'
        )
        bot.send_message(msg)
        logger.warning('ic_tracker: degradation alert sent')
    except Exception as e:
        logger.warning('ic_tracker: alert send failed: %s', e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    """Full IC tracking pipeline. Call from deep_audit.py."""
    logger.info('ic_tracker: starting')
    snapshot_predictions()
    ic    = compute_ic()
    stats = rolling_ic_stats(window_days=30)

    if stats:
        logger.info(
            'ic_tracker: 30d rolling IC=%.4f (n=%d obs)',
            stats['mean_ic'], stats['n_obs']
        )
    check_and_alert(ic, stats)
    logger.info('ic_tracker: done')
    return ic


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
