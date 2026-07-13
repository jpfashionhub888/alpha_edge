# monitoring/heartbeat.py
"""
Phase 4 — Operational Resilience: Heartbeat Monitor

Runs as a background thread inside each live bot (alpaca_live.py, gateio_live.py).
Writes a heartbeat timestamp to disk every N seconds.
A companion watchdog checks the timestamp — if stale by more than STALE_SECONDS,
it fires a Telegram CRITICAL alert.

Design principles:
  - Zero external dependencies (stdlib only + existing TelegramBot)
  - Thread-safe (uses threading.Event for clean shutdown)
  - Fail-silent (never crashes the main trading loop)
  - File-based: works even if the process is wedged (disk is always readable)

Usage:
    from monitoring.heartbeat import HeartbeatMonitor

    # In your bot startup:
    hb = HeartbeatMonitor(service_name='alpaca_bot')
    hb.start()                    # starts background writer thread
    hb.ping()                     # call inside your trading loop

    # On shutdown:
    hb.stop()

Watchdog (called from a separate cron or systemd timer):
    python -m monitoring.heartbeat --check alpaca_bot
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────
HEARTBEAT_DIR      = Path(os.getenv('HEARTBEAT_DIR', 'logs/heartbeats'))
PING_INTERVAL_SEC  = int(os.getenv('HEARTBEAT_INTERVAL', '60'))    # write every 60s
STALE_SECONDS      = int(os.getenv('HEARTBEAT_STALE_SEC', '300'))  # alert if >5 min stale


class HeartbeatMonitor:
    """
    Background thread that writes a heartbeat file every PING_INTERVAL_SEC.

    The heartbeat file is a JSON with:
      - service     : name of the bot
      - pid         : process ID
      - last_ping   : ISO timestamp of last ping
      - cycle_count : how many trading cycles completed
      - status      : 'running' | 'stopped'
    """

    def __init__(
        self,
        service_name   : str,
        ping_interval  : int = PING_INTERVAL_SEC,
        heartbeat_dir  : Path | str = HEARTBEAT_DIR,
    ):
        self.service_name  = service_name
        self.ping_interval = ping_interval
        self.heartbeat_dir = Path(heartbeat_dir)
        self.heartbeat_file= self.heartbeat_dir / f'{service_name}.json'

        self._cycle_count  = 0
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock  = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background heartbeat writer thread."""
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        self._write_state('starting')

        self._thread = threading.Thread(
            target=self._run,
            name=f'heartbeat-{self.service_name}',
            daemon=True,   # dies with main process — no orphan threads
        )
        self._thread.start()
        logger.info(f'[Heartbeat] {self.service_name} monitor started '
                    f'(interval={self.ping_interval}s, file={self.heartbeat_file})')

    def ping(self) -> None:
        """Call this inside your main trading loop to register a cycle."""
        with self._lock:
            self._cycle_count += 1
        self._write_state('running')

    def stop(self) -> None:
        """Signal the background thread to stop cleanly."""
        self._stop_event.set()
        self._write_state('stopped')
        logger.info(f'[Heartbeat] {self.service_name} stopped after '
                    f'{self._cycle_count} cycles')

    def is_alive(self) -> bool:
        """Return True if the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # ── Background thread ──────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._write_state('running')
            except Exception as e:
                logger.debug(f'[Heartbeat] write failed (non-critical): {e}')
            self._stop_event.wait(timeout=self.ping_interval)

    # ── State file ─────────────────────────────────────────────────────────────

    def _write_state(self, status: str) -> None:
        """Atomically write the heartbeat JSON file."""
        state = {
            'service'    : self.service_name,
            'pid'        : os.getpid(),
            'last_ping'  : datetime.now(timezone.utc).isoformat(),
            'cycle_count': self._cycle_count,
            'status'     : status,
        }
        try:
            self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.heartbeat_file.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            tmp.replace(self.heartbeat_file)   # atomic rename
        except Exception as e:
            logger.debug(f'[Heartbeat] atomic write failed: {e}')


# ── Watchdog ───────────────────────────────────────────────────────────────────

class HeartbeatWatchdog:
    """
    Checks heartbeat files and fires Telegram alerts when a service goes stale.

    Designed to be called by a cron job or systemd timer every 5 minutes:
        /root/alpha_edge/venv/bin/python -m monitoring.heartbeat --check all
    """

    def __init__(
        self,
        stale_seconds  : int = STALE_SECONDS,
        heartbeat_dir  : Path | str = HEARTBEAT_DIR,
    ):
        self.stale_seconds  = stale_seconds
        self.heartbeat_dir  = Path(heartbeat_dir)

    def check_all(self) -> list[dict]:
        """Check all heartbeat files. Return list of stale services."""
        if not self.heartbeat_dir.exists():
            return []

        stale = []
        for hb_file in self.heartbeat_dir.glob('*.json'):
            result = self.check_file(hb_file)
            if result.get('stale'):
                stale.append(result)

        return stale

    def check_file(self, path: Path) -> dict:
        """Check a single heartbeat file. Return status dict."""
        try:
            with open(path) as f:
                state = json.load(f)

            last_ping_str = state.get('last_ping', '')
            last_ping     = datetime.fromisoformat(
                last_ping_str.replace('Z', '+00:00')
            )
            now           = datetime.now(last_ping.tzinfo)
            age_seconds   = (now - last_ping).total_seconds()
            is_stale      = age_seconds > self.stale_seconds

            return {
                'service'    : state.get('service', path.stem),
                'pid'        : state.get('pid'),
                'last_ping'  : last_ping_str,
                'age_seconds': round(age_seconds, 1),
                'cycle_count': state.get('cycle_count', 0),
                'status'     : state.get('status', 'unknown'),
                'stale'      : is_stale,
            }
        except Exception as e:
            return {
                'service'    : path.stem,
                'stale'      : True,
                'error'      : str(e),
                'age_seconds': float('inf'),
            }

    def alert_stale(self, stale_services: list[dict]) -> None:
        """Send Telegram CRITICAL alert for each stale service."""
        try:
            from monitoring.telegram_bot import TelegramBot
            bot = TelegramBot()
            if not bot.enabled:
                logger.warning('[Watchdog] Telegram not configured — alert not sent')
                return
        except Exception:
            logger.warning('[Watchdog] Could not initialise TelegramBot')
            return

        for svc in stale_services:
            service     = svc.get('service', 'unknown')
            age_min     = svc.get('age_seconds', 0) / 60
            last_ping   = svc.get('last_ping', 'never')
            cycle_count = svc.get('cycle_count', '?')

            msg = (
                f"CRITICAL: Bot Silent\n\n"
                f"Service: {service}\n"
                f"Silent for: {age_min:.1f} min\n"
                f"Last ping: {last_ping}\n"
                f"Cycles completed: {cycle_count}\n"
                f"Action: Check server immediately\n"
                f"SSH: ssh root@{os.getenv('VPS_HOST', '67.205.185.84')}\n"
                f"Status: systemctl status {service}.service"
            )
            try:
                bot.send_message(msg)
                logger.warning(f'[Watchdog] STALE alert sent for {service}')
            except Exception as e:
                logger.error(f'[Watchdog] Alert failed for {service}: {e}')


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli():
    """
    Called by systemd timer or cron:
        python -m monitoring.heartbeat --check all
        python -m monitoring.heartbeat --check alpaca_bot
    """
    import argparse

    parser = argparse.ArgumentParser(description='AlphaEdge heartbeat watchdog')
    parser.add_argument('--check', metavar='SERVICE',
                        help='Service name to check, or "all"')
    parser.add_argument('--dir', default=str(HEARTBEAT_DIR),
                        help='Heartbeat directory path')
    parser.add_argument('--stale-sec', type=int, default=STALE_SECONDS,
                        help='Seconds before a service is considered stale')
    args = parser.parse_args()

    watchdog = HeartbeatWatchdog(
        stale_seconds=args.stale_sec,
        heartbeat_dir=Path(args.dir),
    )

    if args.check == 'all':
        stale = watchdog.check_all()
    else:
        hb_file = Path(args.dir) / f'{args.check}.json'
        result  = watchdog.check_file(hb_file)
        stale   = [result] if result.get('stale') else []

    if stale:
        print(f'STALE services: {[s["service"] for s in stale]}')
        watchdog.alert_stale(stale)
        sys.exit(1)   # non-zero exit lets systemd/cron detect failure
    else:
        print(f'All services OK (checked {args.check})')
        sys.exit(0)


if __name__ == '__main__':
    _cli()
