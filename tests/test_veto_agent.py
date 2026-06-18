# tests/test_veto_agent.py
"""
Unit tests for veto_agent.py

Groq is a LAZY import inside review_signal() — it's not a module attribute.
Correct way to mock it: patch 'groq.Groq' directly, or patch the import
inside the function scope using sys.modules.

The VetoAgent MUST fail closed — any error → VETO, never APPROVE.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def veto_agent():
    """VetoAgent with a fake API key so self.enabled=True."""
    with patch.dict('os.environ', {'GROQ_API_KEY': 'fake-test-key'}):
        # Clear cached module so reload picks up env var
        if 'veto_agent' in sys.modules:
            del sys.modules['veto_agent']
        from veto_agent import VetoAgent
        return VetoAgent()


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


def _mock_groq_module(response_text: str) -> ModuleType:
    """Build a fake groq module whose Groq class returns response_text."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_text

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_response

    MockGroq = MagicMock(return_value=mock_client_instance)

    fake_groq = ModuleType('groq')
    fake_groq.Groq = MockGroq
    return fake_groq


def _mock_groq_error(exc: Exception) -> ModuleType:
    """Build a fake groq module whose create() raises exc."""
    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.side_effect = exc

    MockGroq = MagicMock(return_value=mock_client_instance)

    fake_groq = ModuleType('groq')
    fake_groq.Groq = MockGroq
    return fake_groq


class TestFailClosed:
    """VetoAgent must return VETO on any failure."""

    def test_network_timeout_returns_veto(self, veto_agent, signal_kwargs):
        """A Timeout during AI call must return VETO."""
        import requests
        fake_groq = _mock_groq_error(requests.Timeout('timed out'))

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') == 'VETO'

    def test_runtime_error_returns_veto(self, veto_agent, signal_kwargs):
        """Any unexpected RuntimeError → VETO."""
        fake_groq = _mock_groq_error(RuntimeError('unexpected'))

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') == 'VETO'

    def test_groq_unavailable_returns_veto(self, veto_agent, signal_kwargs):
        """If groq is not importable at all → VETO (not crash)."""
        # Remove groq from sys.modules to simulate ImportError
        original = sys.modules.pop('groq', None)
        try:
            result = veto_agent.review_signal(**signal_kwargs)
            # Must return VETO or a valid dict — not raise
            assert 'decision' in result
            assert result['decision'] == 'VETO'
        except SystemExit:
            pytest.fail('VetoAgent called sys.exit instead of returning VETO')
        finally:
            if original is not None:
                sys.modules['groq'] = original

    def test_no_api_key_is_safe(self):
        """With no GROQ_API_KEY, VetoAgent initialises without crashing."""
        with patch.dict('os.environ', {'GROQ_API_KEY': ''}):
            if 'veto_agent' in sys.modules:
                del sys.modules['veto_agent']
            from veto_agent import VetoAgent
            agent = VetoAgent()
        assert agent.enabled is False  # graceful degradation


class TestAlwaysStructured:
    """review_signal() always returns a dict with required keys."""

    def test_result_has_decision_reason_confidence(self, veto_agent, signal_kwargs):
        """Result dict must always have decision, reason, confidence."""
        fake_groq = _mock_groq_module('APPROVE: Strong setup, low VIX')

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        assert isinstance(result, dict)
        assert 'decision' in result
        assert 'reason' in result
        assert 'confidence' in result

    def test_decision_is_veto_or_approve(self, veto_agent, signal_kwargs):
        """Decision is always VETO or APPROVE — no other values."""
        fake_groq = _mock_groq_module('APPROVE: all clear')

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') in ('VETO', 'APPROVE'), (
            f"Unexpected decision value: {result.get('decision')}"
        )

    def test_confidence_is_numeric(self, veto_agent, signal_kwargs):
        """Confidence is a float in [0, 1]."""
        fake_groq = _mock_groq_module('APPROVE: clean signal')

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        conf = result.get('confidence', -1)
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0


class TestApprove:
    """Clean signals with mocked AI APPROVE response."""

    def test_clean_signal_approve(self, veto_agent, signal_kwargs):
        """Strong clean signal with AI returning valid JSON APPROVE → APPROVE."""
        import json as _json
        # veto_agent does json.loads() on the AI response — must be valid JSON
        approve_json = _json.dumps({
            'decision'  : 'APPROVE',
            'reason'    : 'Strong uptrend, low VIX, good R:R',
            'confidence': 0.85,
        })
        fake_groq = _mock_groq_module(approve_json)

        with patch.dict(sys.modules, {'groq': fake_groq}):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') == 'APPROVE'

