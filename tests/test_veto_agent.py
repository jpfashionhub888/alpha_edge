# tests/test_veto_agent.py
"""
Unit tests for veto_agent.py

VetoAgent stores its Groq client in self._client (created once in __init__).
The correct way to mock it is to replace self._client directly —
NOT to patch sys.modules['groq'], which has no effect on an already-
instantiated client object.

The VetoAgent MUST fail closed — any error → VETO, never APPROVE.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def veto_agent():
    """VetoAgent with enabled=True and a mock _client (no real Groq calls)."""
    with patch.dict('os.environ', {'GROQ_API_KEY': 'fake-test-key'}):
        if 'veto_agent' in sys.modules:
            del sys.modules['veto_agent']
        from veto_agent import VetoAgent
        agent = VetoAgent()
    # Replace the real Groq client with a MagicMock so no network call escapes
    agent._client = MagicMock()
    return agent


@pytest.fixture
def signal_kwargs():
    return {
        'symbol'           : 'AAPL',
        'price'            : 150.0,
        'prediction'       : 0.75,
        'regime'           : 'uptrend',
        'sentiment'        : 0.3,
        'sector'           : 'tech',
        'market_regime'    : 'bull',
        'mtf_score'        : 0.5,
        'current_positions': {},
        'vix'              : 18.0,
    }


def _configure_client_response(agent, response_text: str) -> None:
    """Configure agent._client to return response_text from chat completions."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_text
    agent._client.chat.completions.create.return_value = mock_response


def _configure_client_error(agent, exc: Exception) -> None:
    """Configure agent._client to raise exc from chat completions."""
    agent._client.chat.completions.create.side_effect = exc


class TestFailClosed:
    """VetoAgent must return VETO on any failure."""

    def test_network_timeout_returns_veto(self, veto_agent, signal_kwargs):
        """A Timeout during AI call must return VETO."""
        import requests
        _configure_client_error(veto_agent, requests.Timeout('timed out'))
        result = veto_agent.review_signal(**signal_kwargs)
        assert result.get('decision') == 'VETO'

    def test_runtime_error_returns_veto(self, veto_agent, signal_kwargs):
        """Any unexpected RuntimeError → VETO."""
        _configure_client_error(veto_agent, RuntimeError('unexpected'))
        result = veto_agent.review_signal(**signal_kwargs)
        assert result.get('decision') == 'VETO'

    def test_groq_unavailable_returns_veto(self, veto_agent, signal_kwargs):
        """If _client is None → VETO (not crash)."""
        veto_agent._client = None
        result = veto_agent.review_signal(**signal_kwargs)
        assert 'decision' in result
        assert result['decision'] == 'VETO'

    def test_no_api_key_is_safe(self):
        """With no GROQ_API_KEY, VetoAgent initialises without crashing."""
        with patch.dict('os.environ', {'GROQ_API_KEY': ''}):
            if 'veto_agent' in sys.modules:
                del sys.modules['veto_agent']
            from veto_agent import VetoAgent
            agent = VetoAgent()
        assert agent.enabled is False   # graceful degradation


class TestAlwaysStructured:
    """review_signal() always returns a dict with required keys."""

    def test_result_has_decision_reason_confidence(self, veto_agent, signal_kwargs):
        """Result dict must always have decision, reason, confidence."""
        _configure_client_response(veto_agent, 'APPROVE: Strong setup, low VIX')
        result = veto_agent.review_signal(**signal_kwargs)
        assert isinstance(result, dict)
        assert 'decision' in result
        assert 'reason' in result
        assert 'confidence' in result

    def test_decision_is_veto_or_approve(self, veto_agent, signal_kwargs):
        """Decision is always VETO or APPROVE — no other values."""
        _configure_client_response(veto_agent, 'APPROVE: all clear')
        result = veto_agent.review_signal(**signal_kwargs)
        assert result.get('decision') in ('VETO', 'APPROVE'), (
            f"Unexpected decision value: {result.get('decision')}"
        )

    def test_confidence_is_numeric(self, veto_agent, signal_kwargs):
        """Confidence is a float in [0, 1]."""
        _configure_client_response(veto_agent, 'APPROVE: clean signal')
        result = veto_agent.review_signal(**signal_kwargs)
        conf = result.get('confidence', -1)
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0


class TestApprove:
    """Clean signals with mocked AI APPROVE response."""

    def test_clean_signal_approve(self, veto_agent, signal_kwargs):
        """Strong clean signal with AI returning valid JSON APPROVE → APPROVE."""
        import json as _json
        approve_json = _json.dumps({
            'decision'  : 'APPROVE',
            'reason'    : 'Strong uptrend, low VIX, good R:R',
            'confidence': 0.85,
        })
        _configure_client_response(veto_agent, approve_json)
        result = veto_agent.review_signal(**signal_kwargs)
        assert result.get('decision') == 'APPROVE'
