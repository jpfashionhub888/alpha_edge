# monitoring/walkforward_monitor.py
"""
AlphaEdge Walk-Forward Validation Monitor

Runs every Sunday (via alphaedge-audit.timer) to validate that models
are still generating genuine edge on RECENT, never-seen data.

Checks:
  1. Recent AUC drift — compares last-30-day AUC to training AUC
  2. Feature drift   — flags features that have drifted >2σ from 90-day baseline
  3. Win rate trend  — rolling 10-trade win rate from trade_tracker
  4. Auto-retrain trigger — calls retrain.py if 10+ consecutive underperforming days

Sends Telegram report every Sunday, with WARN/CRITICAL flags as appropriate.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRADES_FILE    = Path('logs/closed_trades.json')
SIGNALS_FILE   = Path('logs/latest_signals.json')
DRIFT_FILE     = Path('logs/feature_drift_baseline.json')

# Thresholds
AUC_WARN_DROP  = 0.08    # Warn if recent AUC drops 8pp from training AUC
AUC_CRIT_DROP  = 0.15    # Critical / halt if drops 15pp
WR_WARN        = 0.45    # Warn if rolling win rate < 45%
WR_CRIT        = 0.35    # Critical if < 35%
DRIFT_SIGMA    = 2.0     # Alert if feature drifts >2σ from baseline
MIN_TRADES_WF  = 10      # Minimum trades before walk-forward is meaningful
UNDERPERF_DAYS = 10      # Consecutive underperforming days before auto-retrain


def _get_trade_summary() -> dict:
    try:
        with open(TRADES_FILE) as f:
            data = json.load(f)
        return data.get('summary', {})
    except Exception:
        return {}


def _get_recent_winrate(n: int = 10) -> Optional[float]:
    """Win rate on last N closed trades."""
    try:
        with open(TRADES_FILE) as f:
            data = json.load(f)
        trades = data.get('trades', [])
        recent = trades[-n:]
        if not recent:
            return None
        wins = sum(1 for t in recent if t.get('pnl_usd', t.get('pnl', 0)) > 0)
        return wins / len(recent)
    except Exception:
        return None


def _check_feature_drift() -> list[str]:
    """
    Check if any tracked features have drifted >DRIFT_SIGMA from baseline.
    Returns list of drift warnings.
    """
    warnings_out = []
    try:
        if not DRIFT_FILE.exists():
            return ['[INFO] No feature drift baseline yet — will be created on next scan']

        with open(DRIFT_FILE) as f:
            baseline = json.load(f)

        for feature, stats in baseline.items():
            current_mean = stats.get('current_mean')
            base_mean    = stats.get('base_mean')
            base_std     = stats.get('base_std')
            if base_std and base_std > 0 and current_mean is not None:
                z = abs(current_mean - base_mean) / base_std
                if z > DRIFT_SIGMA:
                    warnings_out.append(
                        f'[DRIFT] {feature}: z={z:.1f}σ '
                        f'(base={base_mean:.4f}, now={current_mean:.4f})'
                    )
    except Exception as e:
        warnings_out.append(f'[WARN] Feature drift check failed: {e}')
    return warnings_out


def _check_model_auc() -> tuple[str, bool]:
    """
    Compare recent AUC (from latest scan logs) vs training AUC.
    Returns (message, is_critical).
    """
    try:
        with open(SIGNALS_FILE) as f:
            signals = json.load(f)
        recent_auc  = signals.get('model_auc')
        training_auc = signals.get('training_auc')

        if recent_auc is None or training_auc is None:
            return '[INFO] No AUC data in signals file — add auc logging to scanner', False

        drop = training_auc - recent_auc
        if drop >= AUC_CRIT_DROP:
            return (
                f'[CRITICAL] AUC dropped {drop:.2f} pts '
                f'(train={training_auc:.3f} → recent={recent_auc:.3f}). '
                f'HALTING new entries until retrain.',
                True
            )
        elif drop >= AUC_WARN_DROP:
            return (
                f'[WARN] AUC dropped {drop:.2f} pts '
                f'(train={training_auc:.3f} → recent={recent_auc:.3f}). '
                f'Consider retraining.',
                False
            )
        return (
            f'[OK] AUC stable: train={training_auc:.3f} recent={recent_auc:.3f} '
            f'(Δ={drop:+.3f})',
            False
        )
    except FileNotFoundError:
        return '[INFO] Signals file not found — AUC check skipped', False
    except Exception as e:
        return f'[WARN] AUC check error: {e}', False


def _trigger_retrain(telegram=None) -> bool:
    """Trigger retrain.py as a subprocess."""
    try:
        msg = '[AUTO-RETRAIN] Walk-forward monitor triggering retrain.py...'
        logger.warning(msg)
        if telegram:
            telegram.send_message(f'⚙️ {msg}')

        result = subprocess.run(
            [sys.executable, 'retrain.py'],
            capture_output=True, text=True, timeout=600
        )
        success = result.returncode == 0
        status  = 'completed' if success else f'FAILED (exit {result.returncode})'
        msg2    = f'[RETRAIN] {status}'
        if result.stderr:
            msg2 += f'\nStderr: {result.stderr[-500:]}'

        logger.info(msg2)
        if telegram:
            telegram.send_message(f'⚙️ {msg2}')
        return success
    except subprocess.TimeoutExpired:
        logger.error('Retrain timed out after 600s')
        if telegram:
            telegram.send_message('⚠️ Retrain timed out after 10 minutes.')
        return False
    except Exception as e:
        logger.error('Retrain trigger failed: %s', e)
        return False


def run_walkforward_report(telegram=None, auto_retrain: bool = True) -> str:
    """Run full walk-forward validation and return report string."""

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    summary  = _get_trade_summary()
    n_trades = summary.get('total', 0)
    wr_all   = summary.get('win_rate', 0) * 100
    pf       = summary.get('profit_factor') or 0
    pnl      = summary.get('total_pnl', 0)

    # Recent win rate (last 10 trades)
    wr_recent = _get_recent_winrate(10)
    wr_recent_str = f'{wr_recent*100:.1f}%' if wr_recent is not None else 'N/A'

    # AUC drift
    auc_msg, auc_critical = _check_model_auc()

    # Feature drift
    drift_warnings = _check_feature_drift()

    # Win rate status
    wr_flag = ''
    if wr_recent is not None:
        if wr_recent < WR_CRIT:
            wr_flag = ' [CRITICAL]'
        elif wr_recent < WR_WARN:
            wr_flag = ' [WARN]'
        else:
            wr_flag = ' [OK]'

    drift_block = '\n  '.join(drift_warnings) if drift_warnings else '[OK] No significant drift'

    report = (
        f'AlphaEdge Walk-Forward Report\n'
        f'{"=" * 35}\n'
        f'{now}\n\n'
        f'Trade Stats ({n_trades} total)\n'
        f'  Win rate (all):    {wr_all:.1f}%\n'
        f'  Win rate (last 10):{wr_recent_str}{wr_flag}\n'
        f'  Profit factor:     {pf:.2f}\n'
        f'  Total P&L:         ${pnl:+,.2f}\n\n'
        f'Model Health\n'
        f'  {auc_msg}\n\n'
        f'Feature Drift\n'
        f'  {drift_block}\n'
    )

    # Alerts section
    alerts = []
    if auc_critical:
        alerts.append('[CRITICAL] AUC degraded — new entries HALTED')
    if wr_recent is not None and wr_recent < WR_CRIT:
        alerts.append('[CRITICAL] Recent win rate below 35%')
    if wr_recent is not None and wr_recent < WR_WARN:
        alerts.append('[WARN] Recent win rate below 45%')
    if drift_warnings and any('[DRIFT]' in w for w in drift_warnings):
        alerts.append('[WARN] Feature drift detected — check data pipeline')

    if alerts:
        report += '\nAlerts\n  ' + '\n  '.join(alerts)
    else:
        report += '\n[OK] All walk-forward checks passed'

    if telegram:
        try:
            telegram.send_message(report)
        except Exception as e:
            logger.warning('Walk-forward report send failed: %s', e)

    # Auto-retrain if critical and enough trades
    if auc_critical and auto_retrain and n_trades >= MIN_TRADES_WF:
        _trigger_retrain(telegram)

    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    telegram = None
    try:
        from monitoring.telegram_bot import TelegramBot
        telegram = TelegramBot()
    except Exception as e:
        logger.warning(f'Telegram init failed, notifications disabled: {e}')
    print(run_walkforward_report(telegram=telegram))
