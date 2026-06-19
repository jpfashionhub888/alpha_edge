"""
tests/test_critical.py
Fix 6.2 — Critical path unit tests required by alphaedge_roadmap.md audit.

Run with: pytest tests/test_critical.py -v
All tests must pass before deployment.
"""

import sys
import os
import hmac
import hashlib
import json
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_TEST_SECRET = 'test-critical-secret'


def _make_hmac_sig(payload: dict, secret: str = _TEST_SECRET) -> str:
    """Compute X-AlphaEdge-Signature header value for test payloads."""
    body = json.dumps(payload).encode('utf-8')
    digest = hmac.new(secret.encode('utf-8'), body, hashlib.sha256).hexdigest()
    return f'sha256={digest}'


# ─── Fix 3.2 audit test — circuit breaker must NOT auto-reset on time alone ──

class TestCircuitBreakerNoAutoReset:

    def test_circuit_breaker_does_not_auto_reset_on_time(self):
        """
        Fix 3.2: After 25 hours, circuit breaker must still be triggered.
        The old code auto-reset after 24 hours — this was removed.
        A -5% day followed by another -5% day must NOT re-enable trading
        just because a calendar day boundary was crossed.
        """
        from risk_circuit_breaker import RiskCircuitBreaker
        cb = RiskCircuitBreaker()
        cb.reset(manual=True)   # clean slate

        # Trigger it with a -6% daily loss
        cb._trigger("Test trigger — daily loss -6%", current_value=9400.0, telegram=None)
        assert cb.is_triggered(), "Circuit breaker should be triggered"

        # Simulate 25 hours passing by backdating trigger_date
        past = datetime.datetime.now() - datetime.timedelta(hours=25)
        cb.state['trigger_date'] = past.isoformat()
        cb._save_state()

        # Still at depressed value (no recovery) — must remain triggered
        cb2 = RiskCircuitBreaker()   # reload from disk
        result = cb2.check(current_value=9400.0, starting_capital=10000.0)
        assert result is True, (
            "Circuit breaker must stay active after 25h with no portfolio recovery. "
            "The 24-hour auto-reset was a bug — it re-enabled trading into a continuing crash."
        )

        # Clean up
        cb2.reset(manual=True)

    def test_circuit_breaker_resets_on_recovery_not_time(self):
        """Portfolio recovering above threshold should allow reset."""
        from risk_circuit_breaker import RiskCircuitBreaker
        cb = RiskCircuitBreaker()
        cb.reset(manual=True)

        cb._trigger("Test", current_value=9400.0, telegram=None)
        assert cb.is_triggered()

        # Recovery: 2% above trigger value → should reset
        result = cb.check(current_value=9600.0, starting_capital=10000.0)
        # After recovery auto-reset, should return False (trading allowed)
        assert result is False, "Should reset after portfolio recovers above threshold"

    def test_daily_loss_triggers_breaker(self):
        """A -5% daily loss must trigger the circuit breaker."""
        from risk_circuit_breaker import RiskCircuitBreaker
        cb = RiskCircuitBreaker()
        cb.reset(manual=True)
        cb.state['daily_start_val'] = 10000.0
        cb.state['daily_start'] = datetime.date.today().isoformat()
        cb._save_state()

        triggered = cb.check(current_value=9400.0, starting_capital=10000.0)
        assert triggered is True, "Daily loss of -6% must trigger circuit breaker"
        cb.reset(manual=True)


# ─── Fix 4.3 audit test — veto agent must fail CLOSED ─────────────────────────

class TestVetoAgentFailsClosed:

    def test_groq_not_installed_returns_veto(self):
        """
        Fix 4.3: When groq package is not importable, veto agent must return VETO.
        This tests the ImportError path inside review_signal().
        """
        import unittest.mock as mock
        from veto_agent import VetoAgent

        agent = VetoAgent()
        agent.enabled = True
        agent.api_key = 'fake_key'

        # Remove groq from sys.modules to simulate not-installed
        with mock.patch.dict('sys.modules', {'groq': None}):
            result = agent.review_signal(
                symbol='TEST', price=100.0, prediction=0.8,
                regime='bull', sentiment=0.2, sector='Tech',
                market_regime='BULL', mtf_score=0.9,
                current_positions={}, vix=14.0,
            )

        # ImportError → returns VETO (fail-closed), never APPROVE
        assert result['decision'] == 'VETO', (
            f"Got {result['decision']} — veto agent must return VETO when groq unavailable"
        )
        assert result['confidence'] == 0.0

    def test_veto_agent_disabled_returns_approve(self):
        """When disabled (no API key), APPROVE is acceptable — veto explicitly off."""
        from veto_agent import VetoAgent
        agent = VetoAgent()
        agent.enabled = False
        result = agent.review_signal(
            symbol='MSFT', price=400.0, prediction=0.65,
            regime='uptrend', sentiment=0.1, sector='Tech',
            market_regime='BULL', mtf_score=0.7,
            current_positions={}, vix=15.0,
        )
        assert result['decision'] == 'APPROVE'


# ─── Fix 4.1 — HMAC signature verification ───────────────────────────────────

class TestWebhookHMAC:
    """Fix 4.1: HMAC-SHA256 webhook authentication."""

    def _client(self, secret=_TEST_SECRET):
        import importlib, sys as _sys
        os.environ['ALPHAEDGE_WEBHOOK_SECRET'] = secret
        for mod in ['execution.webhook_server']:
            if mod in _sys.modules:
                del _sys.modules[mod]
        from execution.webhook_server import app
        app.config['TESTING'] = True
        return app.test_client(), secret

    def test_missing_signature_returns_403(self):
        """Requests without X-AlphaEdge-Signature must be rejected 403."""
        client, secret = self._client()
        with client:
            payload = {'secret': secret, 'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0}
            r = client.post('/webhook', json=payload)
        assert r.status_code == 403

    def test_wrong_signature_returns_403(self):
        """Wrong HMAC signature must be rejected 403."""
        client, secret = self._client()
        with client:
            payload = {'secret': secret, 'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0}
            r = client.post(
                '/webhook',
                data=json.dumps(payload),
                content_type='application/json',
                headers={'X-AlphaEdge-Signature': 'sha256=badhash'},
            )
        assert r.status_code == 403

    def test_valid_signature_passes_auth(self):
        """Correct HMAC signature must pass authentication."""
        import unittest.mock as mock
        client, secret = self._client()
        with client:
            payload = {'secret': secret, 'action': 'BUY', 'symbol': 'AAPL', 'price': 150.0}
            body_bytes = json.dumps(payload).encode('utf-8')
            sig = 'sha256=' + hmac.new(
                secret.encode('utf-8'), body_bytes, hashlib.sha256
            ).hexdigest()
            with mock.patch('execution.webhook_server.process_signal'):
                r = client.post(
                    '/webhook',
                    data=body_bytes,
                    content_type='application/json',
                    headers={'X-AlphaEdge-Signature': sig},
                )
        # Past auth — may return 200 or 401 (wrong secret in payload, separate check)
        assert r.status_code != 403, "Valid HMAC must not return 403"


# ─── Fix 1.6 — Signal logic gates ────────────────────────────────────────────

class TestSignalLogic:

    def test_compute_signal_blocks_earnings(self):
        """Fix 1.6: Earnings blackout gate must block BUY during earnings week."""
        from main import compute_signal
        # compute_signal(pred, regime, sent_score, sect_mult, symbol, earnings_symbols, mtf_composite)
        signal, _ = compute_signal(
            pred=0.80, regime='uptrend', sent_score=0.2,
            sect_mult=1.1, symbol='AAPL', earnings_symbols=['AAPL'],
            mtf_composite=0.5,
        )
        assert signal in ('EARNINGS_HOLD', 'HOLD'), (
            f"Expected EARNINGS_HOLD for stock in earnings, got {signal}"
        )

    def test_compute_signal_blocks_weak_prediction(self):
        """Fix 1.6: Prediction below threshold must not generate BUY."""
        from main import compute_signal
        signal, _ = compute_signal(
            pred=0.50, regime='uptrend', sent_score=0.3,
            sect_mult=1.1, symbol='MSFT', earnings_symbols=[],
            mtf_composite=0.5,
        )
        assert signal != 'BUY', f"Prediction 0.50 should not produce BUY, got {signal}"

    def test_compute_signal_blocks_negative_sentiment(self):
        """Fix 1.6: Strong negative sentiment must block BUY."""
        from main import compute_signal
        signal, _ = compute_signal(
            pred=0.75, regime='uptrend', sent_score=-0.5,
            sect_mult=1.1, symbol='NVDA', earnings_symbols=[],
            mtf_composite=0.5,
        )
        assert signal != 'BUY', f"Sentiment -0.5 should not produce BUY, got {signal}"

    def test_compute_signal_blocks_downtrend_regime(self):
        """Fix 1.6: Downtrend regime must block BUY even with strong prediction."""
        from main import compute_signal
        signal, _ = compute_signal(
            pred=0.90, regime='downtrend', sent_score=0.3,
            sect_mult=1.2, symbol='MSFT', earnings_symbols=[],
            mtf_composite=0.5,
        )
        assert signal in ('AVOID', 'HOLD', 'CAUTION'), (
            f"Downtrend regime must not produce BUY, got {signal}"
        )


# ─── Exposure cap test ────────────────────────────────────────────────────────

class TestPortfolioExposureCap:

    def test_exposure_cap_blocks_beyond_max_positions(self):
        """Fix 3.4: Max positions cap must prevent over-allocation."""
        from execution.paper_trader import PaperTrader
        trader = PaperTrader(starting_capital=10000)
        # Open max positions
        for sym in ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']:
            trader.open_position(sym, 10.0, 0.7, reason='test')
        # 6th position must be blocked
        result = trader.open_position('GOOG', 10.0, 0.7, reason='test')
        assert result is False, "6th position must be blocked by max_positions cap"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
