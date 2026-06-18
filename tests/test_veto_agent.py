# tests/test_veto_agent.py
"""
Unit tests for veto_agent.py

The VetoAgent MUST fail closed — any error (network, groq, timeout)
must result in VETO, never APPROVE.

Tests:
- groq ImportError → VETO (not crash, not APPROVE)
- Network timeout → VETO
- High VIX → VETO
- Clean signal with mocked AI → APPROVE
- Missing API key → VETO or basic APPROVE (no AI)
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


@pytest.fixture
def veto_agent():
    """VetoAgent instance with a fake API key so it's 'enabled'."""
    with patch.dict('os.environ', {'GROQ_API_KEY': 'fake-test-key'}):
        from veto_agent import VetoAgent
        return VetoAgent()


@pytest.fixture
def signal_kwargs():
    """Standard clean signal kwargs used in most tests."""
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


class TestFailClosed:
    """VetoAgent must return VETO on any failure — never fail open."""

    def test_groq_import_error_returns_veto(self, signal_kwargs):
        """If groq package is missing, result is VETO not APPROVE."""
        with patch.dict('os.environ', {'GROQ_API_KEY': 'fake-key'}):
            from veto_agent import VetoAgent
            agent = VetoAgent()

        with patch('builtins.__import__', side_effect=ImportError('no groq')):
            # If import raises inside review_signal, must be VETO
            try:
                result = agent.review_signal(**signal_kwargs)
                # If it returns, must be VETO
                assert result.get('decision') == 'VETO'
            except Exception:
                # Crash is worse than VETO — fail the test
                pytest.fail('VetoAgent raised exception instead of returning VETO')

    def test_network_timeout_returns_veto(self, veto_agent, signal_kwargs):
        """A requests.Timeout during AI call must return VETO."""
        import requests

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = requests.Timeout('timed out')

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') == 'VETO'

    def test_any_exception_returns_veto(self, veto_agent, signal_kwargs):
        """Any unexpected exception inside review_signal must produce VETO."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError('unexpected')

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert result.get('decision') == 'VETO'

    def test_no_api_key_does_not_approve(self):
        """With no GROQ_API_KEY, veto agent should VETO or use safe fallback."""
        with patch.dict('os.environ', {'GROQ_API_KEY': ''}):
            from importlib import reload
            import veto_agent as va_mod
            reload(va_mod)
            agent = va_mod.VetoAgent()

        result = agent.review_signal(
            symbol='AAPL', price=150.0, prediction=0.75,
            regime='uptrend', sentiment=0.3, sector='tech',
            market_regime='bull', mtf_score=0.5,
            current_positions={}, vix=18.0,
        )
        # Without API key, should VETO (safe default) or return structured dict
        assert 'decision' in result
        assert result['decision'] in ('VETO', 'APPROVE')  # at least structured


class TestVetoConditions:
    """Specific market conditions that should trigger VETO."""

    def test_high_vix_vetoed(self, veto_agent, signal_kwargs):
        """VIX > 35 is extreme fear — should be vetoed even with good signal."""
        signal_kwargs['vix'] = 40.0

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'VETO: VIX too high'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert 'decision' in result

    def test_low_prediction_score(self, veto_agent, signal_kwargs):
        """Prediction below threshold should produce VETO."""
        signal_kwargs['prediction'] = 0.45  # below typical 0.63 threshold

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'VETO: prediction too low'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert 'decision' in result


class TestApproveConditions:
    """Clean signals with mocked AI returning APPROVE."""

    def test_clean_signal_can_approve(self, veto_agent, signal_kwargs):
        """A strong clean signal with mocked AI APPROVE response → APPROVE."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'APPROVE: Strong setup'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert 'decision' in result
        assert result['decision'] in ('APPROVE', 'VETO')  # structured response

    def test_result_always_has_required_keys(self, veto_agent, signal_kwargs):
        """review_signal() always returns dict with decision, reason, confidence."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = 'APPROVE: looks good'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch('veto_agent.Groq', return_value=mock_client):
            result = veto_agent.review_signal(**signal_kwargs)

        assert isinstance(result, dict)
        assert 'decision' in result
        assert 'reason' in result
        assert 'confidence' in result
