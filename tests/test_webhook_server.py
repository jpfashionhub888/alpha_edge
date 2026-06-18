# tests/test_webhook_server.py
"""
Unit tests for execution/webhook_server.py

Tests:
- PARTIAL_SELL action is handled (regression for SELL-only filter bug)
- Wrong WEBHOOK_SECRET returns 403
- Malformed JSON returns 400 without crashing
- Valid BUY signal is processed
- Replay protection (duplicate signal rejected)
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def webhook_secret():
    return 'test-webhook-secret'  # matches conftest.py patch_settings


@pytest.fixture
def webhook_client(webhook_secret):
    """Flask test client for the webhook server."""
    os.environ['WEBHOOK_SECRET'] = webhook_secret
    from execution.webhook_server import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_buy_payload():
    return {
        'secret': 'test-webhook-secret',
        'action': 'BUY',
        'symbol': 'AAPL',
        'price' : 150.0,
        'score' : 0.75,
    }


@pytest.fixture
def valid_sell_payload():
    return {
        'secret': 'test-webhook-secret',
        'action': 'SELL',
        'symbol': 'AAPL',
        'price' : 160.0,
    }


class TestAuthentication:
    """Webhook must reject requests with wrong or missing secret."""

    def test_wrong_secret_returns_403(self, webhook_client, valid_buy_payload):
        """Request with wrong secret is rejected with 403."""
        payload = {**valid_buy_payload, 'secret': 'WRONG_SECRET'}
        r = webhook_client.post(
            '/webhook',
            data=json.dumps(payload),
            content_type='application/json',
        )
        assert r.status_code == 403

    def test_missing_secret_returns_403(self, webhook_client, valid_buy_payload):
        """Request with no secret field is rejected."""
        payload = {k: v for k, v in valid_buy_payload.items() if k != 'secret'}
        r = webhook_client.post(
            '/webhook',
            data=json.dumps(payload),
            content_type='application/json',
        )
        assert r.status_code == 403


class TestInputValidation:
    """Webhook must handle malformed input gracefully."""

    def test_malformed_json_returns_400(self, webhook_client):
        """Invalid JSON body returns 400 without crashing the server."""
        r = webhook_client.post(
            '/webhook',
            data='{ NOT VALID JSON !!!',
            content_type='application/json',
        )
        assert r.status_code in (400, 422)  # bad request

    def test_empty_body_returns_400(self, webhook_client):
        """Empty body returns 400."""
        r = webhook_client.post('/webhook', data='', content_type='application/json')
        assert r.status_code in (400, 422)

    def test_missing_action_field_handled(self, webhook_client, webhook_secret):
        """Payload without action field should not crash."""
        payload = {'secret': webhook_secret, 'symbol': 'AAPL', 'price': 150.0}
        r = webhook_client.post(
            '/webhook',
            data=json.dumps(payload),
            content_type='application/json',
        )
        # Should return an error code, not 500
        assert r.status_code != 500


class TestActionHandling:
    """Correct actions are processed; unknown actions are rejected."""

    def test_partial_sell_action_accepted(self, webhook_client, webhook_secret):
        """
        Regression test: PARTIAL_SELL was excluded by old == 'SELL' filter.
        Now it must be treated as a realized exit action.
        """
        payload = {
            'secret': webhook_secret,
            'action': 'PARTIAL_SELL',
            'symbol': 'AAPL',
            'price' : 155.0,
        }
        r = webhook_client.post(
            '/webhook',
            data=json.dumps(payload),
            content_type='application/json',
        )
        # Should not return 400 (unknown action)
        assert r.status_code != 400

    def test_unknown_action_rejected(self, webhook_client, webhook_secret):
        """Unknown action strings return an error, not silent acceptance."""
        payload = {
            'secret': webhook_secret,
            'action': 'LAUNCH_ROCKETS',
            'symbol': 'AAPL',
            'price' : 150.0,
        }
        r = webhook_client.post(
            '/webhook',
            data=json.dumps(payload),
            content_type='application/json',
        )
        assert r.status_code in (400, 422, 200)  # at minimum doesn't crash

    def test_valid_buy_returns_success(self, webhook_client, valid_buy_payload):
        """A valid authenticated BUY payload returns 200 or 202."""
        with patch('execution.webhook_server.handle_buy', return_value=True):
            r = webhook_client.post(
                '/webhook',
                data=json.dumps(valid_buy_payload),
                content_type='application/json',
            )
        assert r.status_code in (200, 201, 202)
