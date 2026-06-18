# tests/test_webhook_server.py
"""
Unit tests for execution/webhook_server.py

Actual behaviour (from source):
- Wrong secret → 401 (not 403)
- Missing secret → 200 (secret check skipped if field absent)
- Malformed JSON → 500 (caught by outer except, returns error dict)
- PARTIAL_SELL action is stored and processed (not rejected)
- process_signal() handles all actions including PARTIAL_SELL

Note: The webhook does NOT use a handle_buy() function —
it calls process_signal(signal) internally.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def webhook_client():
    """Flask test client. Module reads WEBHOOK_SECRET from env at import time."""
    os.environ['WEBHOOK_SECRET'] = 'test-webhook-secret'
    # Force reload so env var is picked up
    import importlib, sys
    for mod in ['execution.webhook_server']:
        if mod in sys.modules:
            del sys.modules[mod]
    from execution.webhook_server import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def secret():
    return 'test-webhook-secret'


def _post(client, payload):
    return client.post(
        '/webhook',
        data=json.dumps(payload),
        content_type='application/json',
    )


class TestAuthentication:
    """Secret validation behaviour."""

    def test_wrong_secret_returns_401(self, webhook_client, secret):
        """Request with wrong secret is rejected with 401."""
        r = _post(webhook_client, {
            'secret': 'WRONG', 'action': 'BUY',
            'symbol': 'AAPL', 'price': 150.0,
        })
        assert r.status_code == 401

    def test_correct_secret_not_rejected(self, webhook_client, secret):
        """Request with correct secret is accepted (not 401/403)."""
        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'BUY',
                'symbol': 'AAPL', 'price': 150.0,
            })
        assert r.status_code != 401

    def test_missing_secret_returns_401(self, webhook_client):
        """
        After Phase 2 hardening: missing secret field is now rejected with 401.
        (Old behaviour was to allow missing secret — this was the security gap.)
        """
        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0
            })
        assert r.status_code == 401



class TestInputValidation:
    """Malformed input handling."""

    def test_malformed_json_returns_400(self, webhook_client):
        """After hardening: invalid JSON body returns 400 Bad Request (not 500)."""
        r = webhook_client.post(
            '/webhook',
            data='{ NOT VALID JSON !!!',
            content_type='application/json',
        )
        assert r.status_code == 400
        body = r.get_json()
        assert body.get('status') == 'error'


    def test_empty_body_returns_error_or_processes(self, webhook_client):
        """Empty body → webhook either processes with empty signal or errors."""
        with patch('execution.webhook_server.process_signal'):
            r = webhook_client.post('/webhook', data='', content_type='application/json')
        # Should not crash the server process
        assert r.status_code in (200, 400, 422, 500)

    def test_valid_payload_returns_200(self, webhook_client, secret):
        """A well-formed payload with correct secret returns 200."""
        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'BUY',
                'symbol': 'AAPL', 'price': 150.0,
            })
        assert r.status_code == 200
        body = r.get_json()
        assert body.get('status') == 'success'


class TestActionHandling:
    """Signal actions are stored and processed correctly."""

    def test_partial_sell_stored_in_signal(self, webhook_client, secret):
        """
        Regression: old == 'SELL' filter excluded PARTIAL_SELL.
        PARTIAL_SELL must be accepted and stored in received_signals.
        """
        from execution.webhook_server import received_signals
        initial_count = len(received_signals)

        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'PARTIAL_SELL',
                'symbol': 'AAPL', 'price': 155.0,
            })

        assert r.status_code == 200
        # Signal must have been stored
        assert len(received_signals) > initial_count
        last = received_signals[-1]
        assert last['action'] == 'PARTIAL_SELL'

    def test_buy_signal_stored(self, webhook_client, secret):
        """BUY signal is stored correctly."""
        from execution.webhook_server import received_signals

        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'BUY',
                'symbol': 'TSLA', 'price': 200.0,
            })

        assert r.status_code == 200
        last = received_signals[-1]
        assert last['symbol'] == 'TSLA'
        assert last['action'] == 'BUY'

    def test_sell_signal_stored(self, webhook_client, secret):
        """SELL signal is stored correctly."""
        from execution.webhook_server import received_signals

        with patch('execution.webhook_server.process_signal'):
            _post(webhook_client, {
                'secret': secret, 'action': 'SELL',
                'symbol': 'AAPL', 'price': 160.0,
            })

        last = received_signals[-1]
        assert last['action'] == 'SELL'
