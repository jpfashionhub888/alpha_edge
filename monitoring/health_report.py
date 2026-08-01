# monitoring/health_report.py
"""
AlphaEdge Daily Health Report
Sends an 8 AM ET morning briefing via Telegram covering:
  - Service status (heartbeat age)
  - Open positions + live P&L
  - Capital + overall P&L
  - Win rate + closed trades
  - Last scan timestamp
  - Any warnings

Run via systemd timer (alphaedge-health.timer) at 08:00 ET daily,
or manually:
    cd /root/alpha_edge
    /root/alpha_edge/venv/bin/python -m monitoring.health_report
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    import pytz
    _ET = pytz.timezone('America/New_York')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(os.getenv('ALPHAEDGE_DIR', '/root/alpha_edge'))
TRADES_FILE    = BASE_DIR / 'logs' / 'paper_trades_stocks_only.json'
SIGNALS_FILE   = BASE_DIR / 'logs' / 'latest_signals.json'
HEARTBEAT_FILE = BASE_DIR / 'logs' / 'heartbeats' / 'alpaca_bot.json'
STALE_SEC      = 600   # 10 min — bot should ping at least once per scan cycle


def _load(path: Path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _age_str(iso_ts: str) -> str:
    """Convert ISO timestamp to human-readable age string."""
    try:
        dt = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - dt
        h, rem = divmod(int(age.total_seconds()), 3600)
        m = rem // 60
        if h > 0:
            return f'{h}h {m}m ago'
        return f'{m}m ago'
    except Exception:
        return 'unknown'


def build_report() -> str:
    now_et = datetime.now(_ET).strftime('%Y-%m-%d %H:%M ET')

    # ── Service health ────────────────────────────────────────────────────────
    hb = _load(HEARTBEAT_FILE, {})
    last_ping   = hb.get('last_ping', '')
    cycle_count = hb.get('cycle_count', 0)
    hb_status   = hb.get('status', 'unknown')

    service_ok = False
    service_line = 'UNKNOWN (no heartbeat file)'
    if last_ping:
        try:
            dt = datetime.fromisoformat(last_ping.replace('Z', '+00:00'))
            age_sec = (datetime.now(timezone.utc) - dt).total_seconds()
            service_ok = age_sec < STALE_SEC
            age_str = _age_str(last_ping)
            if service_ok:
                service_line = f'RUNNING  (last ping {age_str}, {cycle_count} cycles)'
            else:
                service_line = f'STALE  (last ping {age_str}) — CHECK SERVER'
        except Exception:
            service_line = f'ERROR reading heartbeat'

    status_icon = '' if service_ok else ''

    # ── Portfolio ─────────────────────────────────────────────────────────────
    port = _load(TRADES_FILE, {})
    capital   = port.get('capital', 10000.0)
    starting  = port.get('starting_capital', 10000.0)
    positions = port.get('positions', {})
    history   = port.get('trade_history', [])
    saved_at  = port.get('saved_at', '')

    pos_val = sum(
        p.get('shares', 0) * p.get('current_price', p.get('entry_price', 0))
        for p in positions.values()
    )
    total   = capital + pos_val
    pnl     = total - starting
    pnl_pct = pnl / starting * 100 if starting > 0 else 0
    pnl_dir = '+' if pnl >= 0 else ''

    sells    = [t for t in history if t.get('action') in ('SELL', 'PARTIAL_SELL')]
    wins     = sum(1 for t in sells if t.get('pnl', 0) > 0)
    losses   = sum(1 for t in sells if t.get('pnl', 0) <= 0)
    closed   = wins + losses
    wr       = wins / closed * 100 if closed > 0 else 0
    realized = sum(t.get('pnl', 0) for t in sells)

    # ── Positions detail ──────────────────────────────────────────────────────
    pos_lines = []
    for sym, p in positions.items():
        shares  = p.get('shares', 0)
        entry   = p.get('entry_price', 0)
        curr    = p.get('current_price', entry)
        p_pnl   = (curr - entry) * shares
        p_pct   = (curr - entry) / entry * 100 if entry > 0 else 0
        arrow   = '' if p_pnl >= 0 else ''
        pos_lines.append(
            f'  {arrow} {sym}: {shares:.2f} sh @ ${entry:.2f} | '
            f'now ${curr:.2f} | {pnl_dir}${p_pnl:.2f} ({p_pct:+.1f}%)'
        )

    pos_section = '\n'.join(pos_lines) if pos_lines else '  None'

    # ── Last scan ─────────────────────────────────────────────────────────────
    sigs = _load(SIGNALS_FILE, {})
    scan_ts = next(
        (v.get('saved_at', '') for v in sigs.values()
         if isinstance(v, dict) and v.get('saved_at')), ''
    )
    scan_age = _age_str(scan_ts) if scan_ts else 'never'
    buy_ct   = sum(1 for v in sigs.values()
                   if isinstance(v, dict) and v.get('signal') == 'BUY')
    sig_count = len(sigs)

    # ── Build message ─────────────────────────────────────────────────────────
    report = f"""ALPHAEDGE MORNING BRIEFING
{now_et}
{'='*35}

{status_icon} SERVICE: {service_line}

PORTFOLIO
  Net Value  : ${total:,.2f}
  Cash       : ${capital:,.2f}
  Invested   : ${pos_val:,.2f}
  Total P&L  : {pnl_dir}${pnl:.2f} ({pnl_dir}{pnl_pct:.2f}%)
  Realized   : {pnl_dir}${realized:.2f}

PERFORMANCE
  Win Rate   : {wr:.1f}%  ({wins}W / {losses}L)
  Closed     : {closed} trades
  Open Pos   : {len(positions)}

OPEN POSITIONS
{pos_section}

LAST SCAN
  Time       : {scan_age}
  Signals    : {sig_count} stocks  |  {buy_ct} BUY
  Next scan  : 16:15 ET today

AlphaEdge V7  |  Paper Trading"""

    return report


def main():
    sys.path.insert(0, str(BASE_DIR))

    report = build_report()
    print(report)
    print()

    try:
        from monitoring.telegram_bot import TelegramBot
        bot = TelegramBot()
        if bot.enabled:
            ok = bot.send_message(report)
            print('Telegram:', 'sent' if ok else 'FAILED')
        else:
            print('Telegram not configured — report printed above only')
    except Exception as e:
        print(f'Telegram error: {e}')


if __name__ == '__main__':
    os.chdir(BASE_DIR)
    main()
