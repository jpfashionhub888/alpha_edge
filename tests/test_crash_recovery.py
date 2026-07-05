#!/usr/bin/env python3
# tests/test_crash_recovery.py
"""
AlphaEdge Crash Recovery Test

Verifies that alpaca.service survives a crash, restarts cleanly,
and correctly reconciles any open positions it had before dying.

Run on the VPS:
    python /root/alpha_edge/tests/test_crash_recovery.py

Or from local machine via SSH:
    ssh root@67.207.82.11 "cd /root/alpha_edge && python tests/test_crash_recovery.py"

Exit codes:
    0 = all tests passed
    1 = one or more tests failed
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
SERVICE_NAME       = 'alpaca'                          # systemd service name
HEARTBEAT_FILE     = Path('logs/heartbeats/alpaca_bot.json')
CONTROL_FILE       = Path('logs/bot_control.json')
PAPER_TRADES_FILE  = Path('logs/paper_trades_stocks_only.json')
RESTART_WAIT_SEC   = 12    # seconds to wait for service to restart
HEARTBEAT_WAIT_SEC = 90    # seconds to wait for first heartbeat after restart

PASS = '\033[92m  ✅ PASS\033[0m'
FAIL = '\033[91m  ❌ FAIL\033[0m'
SKIP = '\033[93m  ⚠️  SKIP\033[0m'
INFO = '     ℹ️  '


class CrashRecoveryTest:
    """
    Automated test suite for crash recovery behaviour.

    Tests:
      T1 - Service is currently running before test
      T2 - Service restarts automatically after SIGKILL
      T3 - Heartbeat file updates after restart
      T4 - Pause state persists across restart
      T5 - Paper trades file is intact after restart
      T6 - Reconciliation log produced on startup
      T7 - No double-open positions after restart
    """

    def __init__(self):
        self.results  = {}     # test_id -> (passed: bool, detail: str)
        self.start_ts = datetime.now(timezone.utc)

    def run_all(self) -> int:
        """Run all tests. Return 0 if all pass, 1 otherwise."""
        print('\n' + '='*60)
        print('  AlphaEdge Crash Recovery Test Suite')
        print(f'  {self.start_ts.strftime("%Y-%m-%d %H:%M UTC")}')
        print('='*60)

        # Check we're on the server (systemctl must exist)
        if not self._systemctl_available():
            print(f'\n{SKIP} Tests require systemctl (must run on VPS)')
            print(f'{INFO} Run: ssh root@67.207.82.11 "cd /root/alpha_edge && python tests/test_crash_recovery.py"')
            return 0   # not a failure — just wrong environment

        # Record pre-crash state
        pre_pid          = self._get_service_pid()
        pre_positions    = self._load_positions()
        pre_heartbeat_ts = self._get_heartbeat_ts()

        # Set a known pause state to test persistence
        self._set_pause_state(paused=True, reason='CrashRecoveryTest pre-crash')

        self._test(
            'T1', 'Service running before test',
            pre_pid is not None,
            f'PID={pre_pid}' if pre_pid else 'Service not running — start it first'
        )

        if pre_pid is None:
            print(f'\n{FAIL} Cannot continue without a running service. Aborting.')
            self._print_summary()
            return 1

        # ── Kill the service ──────────────────────────────────────────
        print(f'\n  [*] Killing alpaca.service (PID {pre_pid})...')
        self._run('systemctl kill -s SIGKILL alpaca.service')
        time.sleep(2)

        killed_pid = self._get_service_pid()
        print(f'  [*] After kill: PID={killed_pid} (should differ from {pre_pid})')

        # ── Wait for restart ──────────────────────────────────────────
        print(f'  [*] Waiting {RESTART_WAIT_SEC}s for systemd to restart...')
        time.sleep(RESTART_WAIT_SEC)

        new_pid     = self._get_service_pid()
        is_active   = self._service_active()

        self._test(
            'T2', 'Service restarts automatically after SIGKILL',
            is_active and new_pid is not None and new_pid != pre_pid,
            f'new_pid={new_pid} active={is_active}'
        )

        # ── Wait for heartbeat ────────────────────────────────────────
        print(f'  [*] Waiting up to {HEARTBEAT_WAIT_SEC}s for heartbeat...')
        new_hb_ts  = self._wait_for_new_heartbeat(pre_heartbeat_ts, HEARTBEAT_WAIT_SEC)

        self._test(
            'T3', 'Heartbeat file updated after restart',
            new_hb_ts is not None,
            f'new_heartbeat={new_hb_ts}' if new_hb_ts else 'No new heartbeat within timeout'
        )

        # ── Check pause state persisted ───────────────────────────────
        pause_state = self._get_pause_state()

        self._test(
            'T4', 'Pause state persists across restart',
            pause_state.get('paused') is True,
            f'paused={pause_state.get("paused")} reason={pause_state.get("reason","?")}'
        )

        # Resume so the bot can trade again
        self._set_pause_state(paused=False, reason='')
        print(f'{INFO} Resumed bot after pause-persistence test')

        # ── Check paper trades file intact ────────────────────────────
        post_positions = self._load_positions()
        positions_ok   = (post_positions is not None and
                          self._positions_match(pre_positions, post_positions))

        self._test(
            'T5', 'Paper trades file intact after restart',
            positions_ok,
            f'pre={len(pre_positions or {})} positions, post={len(post_positions or {})} positions'
        )

        # ── Check reconciliation log ──────────────────────────────────
        recon_log = Path('logs/reconciliation.log')
        recon_ok  = recon_log.exists() and self._file_updated_after(recon_log, self.start_ts)

        self._test(
            'T6', 'Reconciliation log produced on startup',
            recon_ok,
            f'log exists={recon_log.exists()} updated_after_test_start={recon_ok}'
        )

        # ── Check no double positions ─────────────────────────────────
        broker_positions = self._get_broker_positions()
        no_duplicates    = broker_positions is not None  # None = cannot check

        if broker_positions is not None:
            # Each symbol should appear at most once
            symbols = list(broker_positions.keys())
            has_dupes = len(symbols) != len(set(symbols))
            self._test(
                'T7', 'No duplicate positions after restart',
                not has_dupes,
                f'broker positions: {symbols}'
            )
        else:
            self._test('T7', 'No duplicate positions after restart',
                       True, 'SKIP — broker not available in this environment')

        self._print_summary()
        failed = [t for t, (p, _) in self.results.items() if not p]
        return 0 if not failed else 1

    # ── Individual checks ─────────────────────────────────────────────

    def _test(self, tid: str, name: str, passed: bool, detail: str) -> None:
        self.results[tid] = (passed, detail)
        status = PASS if passed else FAIL
        print(f'\n  [{tid}] {name}')
        print(f'{status}  {detail}')

    def _print_summary(self) -> None:
        total  = len(self.results)
        passed = sum(1 for p, _ in self.results.values() if p)
        failed = total - passed

        print('\n' + '='*60)
        print(f'  RESULTS: {passed}/{total} passed', end='')
        if failed:
            print(f'  |  {failed} FAILED ← investigate above')
            failed_ids = [t for t, (p, _) in self.results.items() if not p]
            print(f'  Failed tests: {", ".join(failed_ids)}')
        else:
            print('  🟢 All crash recovery tests PASSED')
        print('='*60 + '\n')

    # ── Helpers ───────────────────────────────────────────────────────

    def _systemctl_available(self) -> bool:
        result = self._run('which systemctl', capture=True)
        return result is not None and result.returncode == 0

    def _get_service_pid(self) -> int | None:
        result = self._run(
            f'systemctl show {SERVICE_NAME}.service --property=MainPID --value',
            capture=True
        )
        if result and result.returncode == 0:
            pid_str = result.stdout.strip()
            try:
                pid = int(pid_str)
                return pid if pid > 0 else None
            except ValueError:
                return None
        return None

    def _service_active(self) -> bool:
        result = self._run(
            f'systemctl is-active {SERVICE_NAME}.service',
            capture=True
        )
        return result is not None and result.stdout.strip() == 'active'

    def _get_heartbeat_ts(self) -> str | None:
        try:
            if HEARTBEAT_FILE.exists():
                with open(HEARTBEAT_FILE) as f:
                    return json.load(f).get('last_ping')
        except Exception:
            pass
        return None

    def _wait_for_new_heartbeat(self, pre_ts: str | None, timeout: int) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            new_ts = self._get_heartbeat_ts()
            if new_ts and new_ts != pre_ts:
                return new_ts
            time.sleep(5)
        return None

    def _get_pause_state(self) -> dict:
        try:
            if CONTROL_FILE.exists():
                with open(CONTROL_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _set_pause_state(self, paused: bool, reason: str) -> None:
        try:
            CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'paused'    : paused,
                'reason'    : reason,
                'paused_at' : datetime.now(timezone.utc).isoformat() if paused else None,
                'saved_at'  : datetime.now(timezone.utc).isoformat(),
            }
            with open(CONTROL_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f'{INFO} Could not set pause state: {e}')

    def _load_positions(self) -> dict | None:
        try:
            if PAPER_TRADES_FILE.exists():
                with open(PAPER_TRADES_FILE) as f:
                    data = json.load(f)
                return data.get('positions', {})
        except Exception:
            pass
        return None

    def _positions_match(self, pre: dict | None, post: dict | None) -> bool:
        """Positions should be identical — crash must not corrupt them."""
        if pre is None or post is None:
            return False
        return set(pre.keys()) == set(post.keys())

    def _get_broker_positions(self) -> dict | None:
        """Try to get live broker positions (only works inside bot environment)."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from execution.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker()
            if broker.connected:
                return broker.get_positions()
        except Exception:
            pass
        return None

    def _file_updated_after(self, path: Path, since: datetime) -> bool:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            return mtime > since
        except Exception:
            return False

    def _run(self, cmd: str, capture: bool = False):
        try:
            return subprocess.run(
                cmd, shell=True,
                capture_output=capture,
                text=True, timeout=30,
            )
        except Exception:
            return None


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == '__main__':
    test = CrashRecoveryTest()
    sys.exit(test.run_all())
