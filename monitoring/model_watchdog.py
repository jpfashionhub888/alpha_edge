# monitoring/model_watchdog.py
"""
AlphaEdge Model Staleness Watchdog

Checks whether the ML models have been retrained recently.
Fires a Telegram alert if models are stale (no retrain in STALE_DAYS).

Also checks:
  - Alpaca API connectivity (not just process alive)
  - Circuit breaker state
  - Bot heartbeat freshness

Designed to be run by a systemd timer every 6 hours:
    ExecStart=/root/alpha_edge/venv/bin/python -m monitoring.model_watchdog

Or called from alpaca_live.py startup and daily audit.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
RETRAIN_LOG      = Path('logs/retrain.log')
RETRAIN_STATE    = Path('logs/retrain_state.json')   # written by retrain.py
CACHE_DIR        = Path('cache/models')
STALE_DAYS       = 35     # alert if no retrain for this many days
WARN_DAYS        = 25     # warn (not critical) at this threshold
CIRCUIT_BREAKER  = Path('logs/circuit_breaker.json')
HEARTBEAT_DIR    = Path('logs/heartbeats')
HEARTBEAT_STALE  = 600    # seconds — 10 min stale = bot may be dead


class ModelWatchdog:
    """
    Multi-check watchdog that sends Telegram alerts for:
      1. Stale ML models (no retrain in STALE_DAYS)
      2. Alpaca API connectivity failure
      3. Circuit breaker triggered (in case Telegram alert was missed)
      4. Bot heartbeat stale (process may have died silently)
    """

    def __init__(self):
        from monitoring.telegram_bot import TelegramBot
        self.telegram = TelegramBot()
        self.now      = datetime.now(timezone.utc)
        self.issues   = []    # list of (severity, message) tuples
        self.ok_msgs  = []    # list of passing checks

    # ── Main entry point ─────────────────────────────────────────────

    def run(self) -> int:
        """
        Run all checks. Returns exit code:
          0 = all OK
          1 = warnings present
          2 = critical issues found
        """
        print('\n' + '='*55)
        print('  AlphaEdge Model & System Watchdog')
        print(f'  {self.now.strftime("%Y-%m-%d %H:%M UTC")}')
        print('='*55)

        self._check_model_staleness()
        self._check_circuit_breaker()
        self._check_heartbeats()
        self._check_alpaca_api()

        self._print_summary()

        if self.issues:
            self._send_telegram_alert()

        critical = [i for i in self.issues if i[0] == 'CRITICAL']
        return 2 if critical else (1 if self.issues else 0)

    # ── Check: Model staleness ────────────────────────────────────────

    def _check_model_staleness(self) -> None:
        """Check when models were last retrained."""
        last_retrain = self._get_last_retrain_date()

        if last_retrain is None:
            self.issues.append((
                'CRITICAL',
                '🧠 Model retrain: NEVER recorded\n'
                '   No retrain log found. Models may be untrained.\n'
                '   Run: python retrain.py'
            ))
            return

        age_days = (self.now - last_retrain).days
        age_str  = last_retrain.strftime('%Y-%m-%d %H:%M UTC')

        if age_days >= STALE_DAYS:
            self.issues.append((
                'CRITICAL',
                f'🧠 Model retrain: OVERDUE ({age_days} days ago)\n'
                f'   Last retrain: {age_str}\n'
                f'   Threshold: {STALE_DAYS} days\n'
                f'   Action: ssh root@server && cd /root/alpha_edge && python retrain.py'
            ))
        elif age_days >= WARN_DAYS:
            self.issues.append((
                'WARNING',
                f'🧠 Model retrain: Due soon ({age_days} days ago)\n'
                f'   Last retrain: {age_str}\n'
                f'   Will alert as CRITICAL in {STALE_DAYS - age_days} days'
            ))
        else:
            self.ok_msgs.append(
                f'  ✅ Models fresh — retrained {age_days}d ago ({age_str})'
            )

    def _get_last_retrain_date(self) -> datetime | None:
        """
        Find the most recent retrain date from multiple sources.
        Tries: retrain_state.json → retrain.log mtime → cache/models mtime
        """
        # Source 1: retrain_state.json (written by retrain.py)
        if RETRAIN_STATE.exists():
            try:
                with open(RETRAIN_STATE) as f:
                    state = json.load(f)
                ts = state.get('last_retrain_utc') or state.get('completed_at')
                if ts:
                    return datetime.fromisoformat(
                        ts.replace('Z', '+00:00')
                    ).astimezone(timezone.utc)
            except Exception as e:
                logger.debug('retrain_state.json parse error: %s', e)

        # Source 2: retrain.log file modification time
        if RETRAIN_LOG.exists():
            try:
                mtime = RETRAIN_LOG.stat().st_mtime
                return datetime.fromtimestamp(mtime, tz=timezone.utc)
            except Exception:
                pass

        # Source 3: newest model cache file
        if CACHE_DIR.exists():
            try:
                meta_files = list(CACHE_DIR.glob('*.meta.json'))
                if meta_files:
                    newest = max(meta_files, key=lambda p: p.stat().st_mtime)
                    with open(newest) as f:
                        meta = json.load(f)
                    ts = meta.get('trained_at')
                    if ts:
                        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            except Exception as e:
                logger.debug('cache meta parse error: %s', e)

        return None

    # ── Check: Circuit breaker ────────────────────────────────────────

    def _check_circuit_breaker(self) -> None:
        """Alert if circuit breaker is still triggered (may have been missed)."""
        if not CIRCUIT_BREAKER.exists():
            self.ok_msgs.append('  ✅ Circuit breaker: file not found (OK for fresh install)')
            return
        try:
            with open(CIRCUIT_BREAKER) as f:
                cb = json.load(f)
            if cb.get('triggered'):
                reason = cb.get('trigger_reason', 'unknown')
                date   = cb.get('trigger_date', 'unknown')
                self.issues.append((
                    'CRITICAL',
                    f'🚨 Circuit breaker: TRIGGERED\n'
                    f'   Reason: {reason}\n'
                    f'   Date: {date}\n'
                    f'   No new entries until manually reset'
                ))
            else:
                peak = cb.get('peak_value', 0)
                self.ok_msgs.append(f'  ✅ Circuit breaker: Clear (peak=${peak:,.2f})')
        except Exception as e:
            self.ok_msgs.append(f'  ⚠️  Circuit breaker: Could not read ({e})')

    # ── Check: Heartbeats ─────────────────────────────────────────────

    def _check_heartbeats(self) -> None:
        """Check all service heartbeat files for staleness."""
        if not HEARTBEAT_DIR.exists():
            self.ok_msgs.append('  ⚠️  Heartbeats: directory not found (bots not started yet?)')
            return

        hb_files = list(HEARTBEAT_DIR.glob('*.json'))
        if not hb_files:
            self.ok_msgs.append('  ⚠️  Heartbeats: no files found')
            return

        for hb_file in hb_files:
            try:
                with open(hb_file) as f:
                    state = json.load(f)
                service    = state.get('service', hb_file.stem)
                last_ping  = state.get('last_ping', '')
                status     = state.get('status', 'unknown')
                cycles     = state.get('cycle_count', 0)

                ping_dt = datetime.fromisoformat(
                    last_ping.replace('Z', '+00:00')
                ).astimezone(timezone.utc)
                age_sec = (self.now - ping_dt).total_seconds()

                if age_sec > HEARTBEAT_STALE:
                    age_min = age_sec / 60
                    self.issues.append((
                        'CRITICAL',
                        f'💔 Heartbeat stale: {service}\n'
                        f'   Silent for: {age_min:.0f} minutes\n'
                        f'   Last ping: {last_ping[:19]} UTC\n'
                        f'   Status: {status} | Cycles: {cycles}\n'
                        f'   Check: systemctl status {service}.service'
                    ))
                else:
                    age_min = age_sec / 60
                    self.ok_msgs.append(
                        f'  ✅ {service}: alive ({age_min:.0f}min ago, {cycles} cycles)'
                    )
            except Exception as e:
                self.issues.append((
                    'WARNING',
                    f'💔 Heartbeat: could not read {hb_file.name}: {e}'
                ))

    # ── Check: Alpaca API ─────────────────────────────────────────────

    def _check_alpaca_api(self) -> None:
        """
        Verify Alpaca API is reachable and returns valid account data.
        This goes beyond 'process alive' to confirm the broker connection works.
        """
        try:
            import requests
            api_key    = os.getenv('ALPACA_API_KEY', '')
            secret_key = os.getenv('ALPACA_SECRET_KEY', '')
            base_url   = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if not api_key or not secret_key:
                self.ok_msgs.append('  ⚠️  Alpaca API: keys not in env (OK if watchdog runs outside bot)')
                return

            resp = requests.get(
                f'{base_url}/v2/account',
                headers={
                    'APCA-API-KEY-ID'    : api_key,
                    'APCA-API-SECRET-KEY': secret_key,
                },
                timeout=10,
            )

            if resp.status_code == 200:
                acct = resp.json()
                port = float(acct.get('portfolio_value', 0))
                status = acct.get('status', 'unknown')
                mode = 'PAPER' if 'paper' in base_url else 'LIVE'
                self.ok_msgs.append(
                    f'  ✅ Alpaca API ({mode}): OK — portfolio=${port:,.2f} status={status}'
                )
            else:
                self.issues.append((
                    'CRITICAL',
                    f'📡 Alpaca API: HTTP {resp.status_code}\n'
                    f'   Response: {resp.text[:200]}\n'
                    f'   Bot may not be able to trade!'
                ))
        except requests.exceptions.Timeout:
            self.issues.append(('WARNING', '📡 Alpaca API: Timeout (network issue?)'))
        except Exception as e:
            self.ok_msgs.append(f'  ⚠️  Alpaca API: check skipped ({e})')

    # ── Output ────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        print('\nPassing checks:')
        for msg in self.ok_msgs:
            print(msg)

        if self.issues:
            print('\nIssues found:')
            for severity, msg in self.issues:
                print(f'\n  [{severity}]')
                for line in msg.split('\n'):
                    print(f'  {line}')
        else:
            print('\n  🟢 All systems nominal')

    def _send_telegram_alert(self) -> None:
        if not self.telegram.enabled:
            logger.warning('Telegram not configured — alert not sent')
            return

        critical = [m for s, m in self.issues if s == 'CRITICAL']
        warnings = [m for s, m in self.issues if s == 'WARNING']

        header = '🚨 CRITICAL ALERT' if critical else '⚠️ WARNING'
        ts     = self.now.strftime('%Y-%m-%d %H:%M UTC')

        parts = [f'{header} — AlphaEdge Watchdog\n{ts}\n']

        for msg in critical:
            parts.append(f'\n{msg}')
        for msg in warnings:
            parts.append(f'\n{msg}')

        parts.append('\n\nRun /status for current bot state.')

        self.telegram.send_message('\n'.join(parts))
        logger.info('Watchdog alert sent via Telegram (%d issues)', len(self.issues))


# ── CLI entry point ───────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = '%(levelname)s: %(message)s',
    )
    watchdog = ModelWatchdog()
    sys.exit(watchdog.run())


if __name__ == '__main__':
    main()
