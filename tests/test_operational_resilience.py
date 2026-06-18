# tests/test_operational_resilience.py
"""
Phase 4 — Operational Resilience Tests

Tests:
  1. Kill switch — activation, rejection of signals, reset, wrong secret
  2. Heartbeat — write/read cycle, staleness detection, atomic file write
  3. Reconciliation — phantom/orphan/mismatch detection, clean state
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Kill Switch Tests ──────────────────────────────────────────────────────────

@pytest.fixture
def wh_client():
    """Fresh webhook Flask test client with kill switch reset to False."""
    os.environ['WEBHOOK_SECRET'] = 'test-webhook-secret'

    import importlib
    for mod in ['execution.webhook_server']:
        if mod in sys.modules:
            del sys.modules[mod]

    from execution.webhook_server import app
    import execution.webhook_server as ws
    # Reset kill switch state between tests
    ws._kill_switch_active = False
    ws._kill_switch_reason = ''

    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def secret():
    return 'test-webhook-secret'


class TestKillSwitch:
    """Kill switch endpoint must halt and resume trading safely."""

    def test_activate_with_correct_secret(self, wh_client, secret):
        """POST /kill-switch with correct secret → 200 halted."""
        r = wh_client.post(
            '/kill-switch',
            json={'secret': secret, 'reason': 'test halt'},
            content_type='application/json',
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body.get('status') == 'halted'

    def test_activate_with_wrong_secret(self, wh_client):
        """POST /kill-switch with wrong secret → 401."""
        r = wh_client.post(
            '/kill-switch',
            json={'secret': 'WRONG', 'reason': 'hack attempt'},
            content_type='application/json',
        )
        assert r.status_code == 401

    def test_signals_rejected_when_kill_switch_active(self, wh_client, secret):
        """After kill switch activated, all /webhook signals return 503."""
        # Activate
        wh_client.post(
            '/kill-switch',
            json={'secret': secret, 'reason': 'market crash'},
            content_type='application/json',
        )
        # Now try to send a signal
        with patch('execution.webhook_server.process_signal'):
            r = wh_client.post(
                '/webhook',
                json={'secret': secret, 'action': 'BUY',
                      'symbol': 'AAPL', 'price': 150.0},
                content_type='application/json',
            )
        assert r.status_code == 503
        body = r.get_json()
        assert body.get('status') == 'halted'

    def test_reset_restores_signal_processing(self, wh_client, secret):
        """POST /kill-switch/reset → signals accepted again."""
        # Activate
        wh_client.post('/kill-switch',
                       json={'secret': secret, 'reason': 'test'},
                       content_type='application/json')
        # Reset
        r = wh_client.post('/kill-switch/reset',
                           json={'secret': secret},
                           content_type='application/json')
        assert r.status_code == 200
        assert r.get_json().get('status') == 'running'

        # Signal should now be accepted
        with patch('execution.webhook_server.process_signal'):
            r2 = wh_client.post(
                '/webhook',
                json={'secret': secret, 'action': 'BUY',
                      'symbol': 'AAPL', 'price': 150.0},
                content_type='application/json',
            )
        assert r2.status_code == 200

    def test_reset_wrong_secret_rejected(self, wh_client, secret):
        """POST /kill-switch/reset with wrong secret → 401."""
        wh_client.post('/kill-switch',
                       json={'secret': secret, 'reason': 'test'},
                       content_type='application/json')
        r = wh_client.post('/kill-switch/reset',
                           json={'secret': 'WRONG'},
                           content_type='application/json')
        assert r.status_code == 401

    def test_get_kill_switch_status(self, wh_client, secret):
        """GET /kill-switch returns current state dict."""
        r = wh_client.get('/kill-switch')
        assert r.status_code == 200
        body = r.get_json()
        assert 'active' in body
        assert 'reason' in body
        assert 'time' in body

    def test_health_shows_kill_switch_state(self, wh_client, secret):
        """GET /health shows kill_switch field."""
        r = wh_client.get('/health')
        body = r.get_json()
        assert 'kill_switch' in body

    def test_reason_truncated_to_200_chars(self, wh_client, secret):
        """Kill switch reason is capped at 200 chars to prevent DoS."""
        long_reason = 'X' * 500
        r = wh_client.post(
            '/kill-switch',
            json={'secret': secret, 'reason': long_reason},
            content_type='application/json',
        )
        assert r.status_code == 200
        import execution.webhook_server as ws
        assert len(ws._kill_switch_reason) <= 200


# ── Heartbeat Tests ────────────────────────────────────────────────────────────

class TestHeartbeat:
    """Heartbeat monitor writes and reads correctly."""

    def test_heartbeat_writes_file(self, tmp_path):
        """HeartbeatMonitor.start() writes a heartbeat file."""
        from monitoring.heartbeat import HeartbeatMonitor
        hb = HeartbeatMonitor(
            service_name='test_service',
            ping_interval=1,
            heartbeat_dir=tmp_path / 'hb',
        )
        hb.start()
        time.sleep(0.2)
        hb.stop()

        hb_file = tmp_path / 'hb' / 'test_service.json'
        assert hb_file.exists(), 'Heartbeat file not created'

    def test_heartbeat_file_valid_json(self, tmp_path):
        """Heartbeat file must be valid JSON with required fields."""
        from monitoring.heartbeat import HeartbeatMonitor
        hb = HeartbeatMonitor(
            service_name='test_service',
            ping_interval=1,
            heartbeat_dir=tmp_path / 'hb',
        )
        hb.start()
        time.sleep(0.2)
        hb.stop()

        hb_file = tmp_path / 'hb' / 'test_service.json'
        with open(hb_file) as f:
            state = json.load(f)

        for key in ('service', 'pid', 'last_ping', 'cycle_count', 'status'):
            assert key in state, f'Missing key: {key}'

    def test_ping_increments_cycle_count(self, tmp_path):
        """ping() increments cycle_count."""
        from monitoring.heartbeat import HeartbeatMonitor
        hb = HeartbeatMonitor(
            service_name='test_service',
            ping_interval=60,           # long interval — only manual pings
            heartbeat_dir=tmp_path / 'hb',
        )
        hb.start()
        hb.ping()
        hb.ping()
        hb.ping()
        hb.stop()

        hb_file = tmp_path / 'hb' / 'test_service.json'
        with open(hb_file) as f:
            state = json.load(f)

        assert state['cycle_count'] == 3

    def test_watchdog_detects_stale_service(self, tmp_path):
        """HeartbeatWatchdog marks services with old last_ping as stale."""
        from monitoring.heartbeat import HeartbeatWatchdog
        from datetime import datetime, timezone, timedelta

        hb_dir = tmp_path / 'hb'
        hb_dir.mkdir()

        # Write a heartbeat that is 10 minutes old
        old_ping = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        state = {
            'service'    : 'old_service',
            'pid'        : 12345,
            'last_ping'  : old_ping,
            'cycle_count': 5,
            'status'     : 'running',
        }
        (hb_dir / 'old_service.json').write_text(json.dumps(state))

        watchdog = HeartbeatWatchdog(stale_seconds=60, heartbeat_dir=hb_dir)
        stale    = watchdog.check_all()

        assert len(stale) == 1
        assert stale[0]['service'] == 'old_service'
        assert stale[0]['stale'] is True

    def test_watchdog_ignores_fresh_service(self, tmp_path):
        """HeartbeatWatchdog does not flag recently-updated services."""
        from monitoring.heartbeat import HeartbeatMonitor, HeartbeatWatchdog

        hb = HeartbeatMonitor(
            service_name='fresh_service',
            ping_interval=60,
            heartbeat_dir=tmp_path / 'hb',
        )
        hb.start()
        hb.ping()
        time.sleep(0.1)
        hb.stop()

        watchdog = HeartbeatWatchdog(stale_seconds=300, heartbeat_dir=tmp_path / 'hb')
        stale    = watchdog.check_all()

        stale_names = [s['service'] for s in stale]
        assert 'fresh_service' not in stale_names

    def test_heartbeat_is_daemon_thread(self, tmp_path):
        """Background thread must be daemon so it dies with the main process."""
        from monitoring.heartbeat import HeartbeatMonitor
        hb = HeartbeatMonitor(
            service_name='daemon_test',
            heartbeat_dir=tmp_path / 'hb',
        )
        hb.start()
        assert hb._thread is not None
        assert hb._thread.daemon is True
        hb.stop()


# ── Reconciliation Tests ───────────────────────────────────────────────────────

class TestReconciliation:
    """Position reconciler detects phantom, orphan, and mismatch correctly."""

    def _make_local_state(self, tmp_path, positions: dict) -> Path:
        """Write a fake paper_trades JSON to tmp_path."""
        state = {
            'capital'   : 10_000.0,
            'positions' : {
                sym: {
                    'shares'       : pos['shares'],
                    'entry_price'  : pos['price'],
                    'current_price': pos['price'],
                }
                for sym, pos in positions.items()
            },
        }
        p = tmp_path / 'paper_trades.json'
        p.write_text(json.dumps(state))
        return p

    def _make_broker(self, positions: dict):
        """Mock broker.get_positions() returning {symbol: market_value}."""
        broker = MagicMock()
        broker.get_positions.return_value = {
            sym: {'market_value': val}
            for sym, val in positions.items()
        }
        return broker

    def test_clean_state_no_discrepancies(self, tmp_path):
        """Identical broker and local positions → no discrepancies."""
        from monitoring.reconciliation import PositionReconciler

        log = self._make_local_state(
            tmp_path,
            {'AAPL': {'shares': 10, 'price': 150.0}}  # market value = 1500
        )
        broker = self._make_broker({'AAPL': 1500.0})

        r = PositionReconciler(
            log_file=log, service_name='test', dollar_tolerance=20.0
        )
        discs = r.reconcile(broker)
        assert discs == []

    def test_phantom_position_detected(self, tmp_path):
        """Broker has NVDA, local state doesn't → PHANTOM discrepancy."""
        from monitoring.reconciliation import PositionReconciler

        log    = self._make_local_state(tmp_path, {})  # empty local
        broker = self._make_broker({'NVDA': 5000.0})

        r     = PositionReconciler(log_file=log, service_name='test')
        discs = r.reconcile(broker)

        assert len(discs) == 1
        assert discs[0]['type']   == 'PHANTOM'
        assert discs[0]['symbol'] == 'NVDA'

    def test_orphan_position_detected(self, tmp_path):
        """Local state has TSLA, broker doesn't → ORPHAN discrepancy."""
        from monitoring.reconciliation import PositionReconciler

        log    = self._make_local_state(
            tmp_path, {'TSLA': {'shares': 5, 'price': 200.0}}
        )
        broker = self._make_broker({})   # broker has nothing

        r     = PositionReconciler(log_file=log, service_name='test')
        discs = r.reconcile(broker)

        assert len(discs) == 1
        assert discs[0]['type']   == 'ORPHAN'
        assert discs[0]['symbol'] == 'TSLA'

    def test_value_mismatch_detected(self, tmp_path):
        """
        Both have AAPL but values differ by > tolerance ($10) → MISMATCH.
        Broker: $1500, local: $1000 → diff $500 > $10 tolerance.
        """
        from monitoring.reconciliation import PositionReconciler

        log    = self._make_local_state(
            tmp_path, {'AAPL': {'shares': 10, 'price': 100.0}}  # value=1000
        )
        broker = self._make_broker({'AAPL': 1500.0})

        r     = PositionReconciler(
            log_file=log, service_name='test', dollar_tolerance=10.0
        )
        discs = r.reconcile(broker)

        assert len(discs) == 1
        assert discs[0]['type'] == 'MISMATCH'
        assert discs[0]['diff'] > 10.0

    def test_small_value_diff_within_tolerance(self, tmp_path):
        """Values differ by $5 (< $10 tolerance) → no discrepancy."""
        from monitoring.reconciliation import PositionReconciler

        log    = self._make_local_state(
            tmp_path, {'AAPL': {'shares': 10, 'price': 150.0}}   # value=1500
        )
        broker = self._make_broker({'AAPL': 1505.0})   # +$5

        r     = PositionReconciler(
            log_file=log, service_name='test', dollar_tolerance=10.0
        )
        discs = r.reconcile(broker)
        assert discs == []

    def test_missing_local_file_treated_as_empty(self, tmp_path):
        """No local state file → treated as empty (clean first run)."""
        from monitoring.reconciliation import PositionReconciler

        log    = tmp_path / 'nonexistent.json'  # does not exist
        broker = self._make_broker({})           # broker also empty

        r     = PositionReconciler(log_file=log, service_name='test')
        discs = r.reconcile(broker)
        assert discs == []

    def test_reconciliation_log_written(self, tmp_path):
        """Reconciler writes an entry to logs/reconciliation.log."""
        from monitoring.reconciliation import PositionReconciler
        import monitoring.reconciliation as recon_mod

        log    = self._make_local_state(tmp_path, {})
        broker = self._make_broker({})

        # Redirect recon log to tmp
        original_log = recon_mod.RECON_LOG
        recon_mod.RECON_LOG = tmp_path / 'recon.log'

        try:
            r = PositionReconciler(log_file=log, service_name='test')
            r.reconcile(broker)
            assert (tmp_path / 'recon.log').exists(), 'Reconciliation log not written'
        finally:
            recon_mod.RECON_LOG = original_log

    def test_broker_fetch_failure_returns_empty(self, tmp_path):
        """If broker.get_positions() raises, reconciler skips check gracefully."""
        from monitoring.reconciliation import PositionReconciler

        log    = self._make_local_state(tmp_path, {})
        broker = MagicMock()
        broker.get_positions.side_effect = Exception('API timeout')

        r     = PositionReconciler(log_file=log, service_name='test')
        discs = r.reconcile(broker)
        assert discs == []   # graceful skip, not crash
