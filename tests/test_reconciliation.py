"""
Tests for monitoring/reconciliation.py

Covers the broker-adapter branches in _fetch_broker_positions, which
previously silently no-op'd for the actual BybitClient / GateioClient
interfaces (neither exposes get_positions() or get_balances(), which
were the only two branches originally implemented).
"""

import json

import pytest

from monitoring.reconciliation import PositionReconciler


class FakeAlpacaBroker:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return self._positions


class FakeBybitClient:
    def __init__(self, positions):
        self._positions = positions

    def get_all_positions(self):
        return self._positions


class FakeGateioClient:
    def __init__(self, balances, prices=None):
        self._balances = balances
        self._prices = prices or {}

    def get_spot_balances(self):
        return self._balances

    def get_last_price(self, pair):
        return self._prices.get(pair)


def _write_local_state(path, positions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'positions': positions}, f)


def test_alpaca_style_broker_clean(tmp_path):
    state_file = tmp_path / 'local.json'
    _write_local_state(state_file, {
        'AAPL': {'shares': 10, 'entry_price': 100.0},
    })
    broker = FakeAlpacaBroker({'AAPL': {'market_value': 1000.0}})
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    assert discrepancies == []


def test_bybit_client_recognized_and_phantom_detected(tmp_path):
    # Local file doesn't exist -> treated as empty local state
    state_file = tmp_path / 'does_not_exist.json'
    broker = FakeBybitClient([
        {'symbol': 'BTCUSDT', 'size': '0.5', 'markPrice': '60000'},
    ])
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    assert len(discrepancies) == 1
    assert discrepancies[0]['type'] == 'PHANTOM'
    assert discrepancies[0]['symbol'] == 'BTCUSDT'
    assert discrepancies[0]['broker_value'] == pytest.approx(30000.0)


def test_bybit_client_empty_positions_is_clean(tmp_path):
    state_file = tmp_path / 'does_not_exist.json'
    broker = FakeBybitClient([])
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    assert discrepancies == []


def test_gateio_client_priced_balance_detected(tmp_path):
    state_file = tmp_path / 'does_not_exist.json'
    broker = FakeGateioClient(
        balances=[
            {'currency': 'USDT', 'available': '500', 'locked': '0'},
            {'currency': 'BTC', 'available': '0.1', 'locked': '0'},
        ],
        prices={'BTC_USDT': 60000.0},
    )
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    # USDT is cash, excluded. BTC priced at 60000 * 0.1 = 6000 -> PHANTOM.
    assert len(discrepancies) == 1
    assert discrepancies[0]['symbol'] == 'BTC'
    assert discrepancies[0]['broker_value'] == pytest.approx(6000.0)


def test_gateio_client_unpriceable_balance_is_skipped_not_crashed(tmp_path):
    state_file = tmp_path / 'does_not_exist.json'
    broker = FakeGateioClient(
        balances=[{'currency': 'SOME_NEW_COIN', 'available': '10', 'locked': '0'}],
        prices={},  # get_last_price returns None -> can't verify, must skip safely
    )
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    assert discrepancies == []


def test_unknown_broker_type_returns_empty_not_none_crash(tmp_path):
    state_file = tmp_path / 'does_not_exist.json'

    class Unknown:
        pass

    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(Unknown())
    assert discrepancies == []


def test_orphan_detected_when_local_has_position_broker_does_not(tmp_path):
    state_file = tmp_path / 'local.json'
    _write_local_state(state_file, {'AAPL': {'shares': 10, 'entry_price': 100.0}})
    broker = FakeAlpacaBroker({})
    reconciler = PositionReconciler(log_file=state_file, service_name='test')
    discrepancies = reconciler.reconcile(broker)
    assert len(discrepancies) == 1
    assert discrepancies[0]['type'] == 'ORPHAN'


def test_mismatch_detected_beyond_tolerance(tmp_path):
    state_file = tmp_path / 'local.json'
    _write_local_state(state_file, {'AAPL': {'shares': 10, 'entry_price': 100.0}})
    broker = FakeAlpacaBroker({'AAPL': {'market_value': 1500.0}})
    reconciler = PositionReconciler(
        log_file=state_file, service_name='test', dollar_tolerance=10.0
    )
    discrepancies = reconciler.reconcile(broker)
    assert len(discrepancies) == 1
    assert discrepancies[0]['type'] == 'MISMATCH'
