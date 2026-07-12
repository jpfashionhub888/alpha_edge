# tests/test_webhook_server.py
"""
Unit tests for execution/webhook_server.py

Fix 4.1 update: All /webhook requests now require X-AlphaEdge-Signature HMAC header.
Tests updated to include proper HMAC signatures on all webhook POST calls.

Actual behaviour:
- Missing/wrong HMAC → 403 (Fix 4.1)
- Wrong payload secret → 401 (after HMAC passes)
- Malformed JSON → 400
- PARTIAL_SELL action is stored and processed (not rejected)
"""

import hashlib
import hmac
import json
import os
from unittest.mock import MagicMock, patch

import pytest

_TEST_SECRET = 'test-webhook-secret'


def _compute_sig(payload_bytes: bytes, secret: str = _TEST_SECRET) -> str:
    """Compute correct X-AlphaEdge-Signature header value."""
    digest = hmac.new(secret.encode('utf-8'), payload_bytes, hashlib.sha256).hexdigest()
    return f'sha256={digest}'


@pytest.fixture
def webhook_client():
    """Flask test client with HMAC secret set in env.

    Also resets the kill switch state so tests from test_operational_resilience.py
    (which leave the kill switch active on disk) don't bleed into these tests.
    """
    import sys
    os.environ['ALPHAEDGE_WEBHOOK_SECRET'] = _TEST_SECRET
    os.environ['WEBHOOK_SECRET']            = _TEST_SECRET
    # Remove stale module so it re-imports cleanly
    for mod in list(sys.modules.keys()):
        if 'webhook_server' in mod:
            del sys.modules[mod]
    from execution.webhook_server import app
    # Reset kill switch module globals — the module loads disk state at import
    # time; if a prior test left it active we get 503 on every request.
    import execution.webhook_server as _ws
    _ws._kill_switch_active = False
    _ws._kill_switch_reason = ''
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def secret():
    return _TEST_SECRET


def _post(client, payload, secret=_TEST_SECRET):
    """POST to /webhook with correct HMAC signature."""
    body = json.dumps(payload).encode('utf-8')
    sig  = _compute_sig(body, secret)
    return client.post(
        '/webhook',
        data=body,
        content_type='application/json',
        headers={'X-AlphaEdge-Signature': sig},
    )


class TestAuthentication:
    """HMAC and secret validation behaviour."""

    def test_no_hmac_returns_403(self, webhook_client, secret):
        """Fix 4.1: Request without X-AlphaEdge-Signature must return 403."""
        r = webhook_client.post(
            '/webhook',
            data=json.dumps({'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0}),
            content_type='application/json',
        )
        assert r.status_code == 403

    def test_wrong_hmac_returns_403(self, webhook_client, secret):
        """Fix 4.1: Incorrect HMAC signature must return 403."""
        r = webhook_client.post(
            '/webhook',
            data=json.dumps({'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0}),
            content_type='application/json',
            headers={'X-AlphaEdge-Signature': 'sha256=badhash'},
        )
        assert r.status_code == 403

    def test_wrong_secret_returns_401(self, webhook_client, secret):
        """After HMAC passes, wrong payload secret → 401."""
        r = _post(webhook_client, {
            'secret': 'WRONG', 'action': 'BUY',
            'symbol': 'AAPL', 'price': 150.0,
        })
        assert r.status_code == 401

    def test_correct_secret_not_rejected(self, webhook_client, secret):
        """Request with correct HMAC + correct secret is accepted."""
        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'BUY',
                'symbol': 'AAPL', 'price': 150.0,
            })
        assert r.status_code not in (401, 403)

    def test_missing_secret_returns_401(self, webhook_client, secret):
        """After HMAC passes, missing payload secret → 401."""
        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0
            })
        assert r.status_code == 401


class TestInputValidation:
    """Malformed input handling."""

    def test_malformed_json_returns_400(self, webhook_client, secret):
        """Invalid JSON body with valid HMAC → 400 Bad Request."""
        body = b'{ NOT VALID JSON !!!'
        sig  = _compute_sig(body)
        r    = webhook_client.post(
            '/webhook',
            data=body,
            content_type='application/json',
            headers={'X-AlphaEdge-Signature': sig},
        )
        assert r.status_code == 400

    def test_empty_body_returns_error_or_processes(self, webhook_client, secret):
        """Empty body with valid HMAC → error or processes gracefully."""
        body = b''
        sig  = _compute_sig(body)
        with patch('execution.webhook_server.process_signal'):
            r = webhook_client.post(
                '/webhook',
                data=body,
                content_type='application/json',
                headers={'X-AlphaEdge-Signature': sig},
            )
        assert r.status_code in (200, 400, 401, 422, 500)

    def test_valid_payload_returns_200(self, webhook_client, secret):
        """Well-formed payload with correct HMAC + secret returns 200."""
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
        assert len(received_signals) > initial_count
        last = received_signals[-1]
        assert last['action'] == 'PARTIAL_SELL'

    def test_buy_signal_stored(self, webhook_client, secret):
        """BUY signal with correct HMAC is stored correctly."""
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
        """SELL signal with correct HMAC is stored correctly."""
        from execution.webhook_server import received_signals

        with patch('execution.webhook_server.process_signal'):
            r = _post(webhook_client, {
                'secret': secret, 'action': 'SELL',
                'symbol': 'NVDA', 'price': 500.0,
            })

        assert r.status_code == 200
        last = received_signals[-1]
        assert last['symbol'] == 'NVDA'
        assert last['action'] == 'SELL'
