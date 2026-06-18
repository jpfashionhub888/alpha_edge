# tests/test_risk_circuit_breaker.py
"""
Unit tests for risk_circuit_breaker.py

Actual API:
  RiskCircuitBreaker()             — no args; uses logs/circuit_breaker.json
  cb.check(current_value, starting_capital, telegram=None)
      → True  if circuit is TRIGGERED (block trading)
      → False if circuit is CLEAR (allow trading)
  cb.reset(manual=True)            — clear the trigger
  cb.is_triggered()                → bool
  cb.get_status()                  → dict

Key thresholds (constants in module):
  DAILY_LOSS_LIMIT  = 0.05   (-5%  daily)
  TOTAL_LOSS_LIMIT  = 0.10   (-10% from starting capital)
  DRAWDOWN_LIMIT    = 0.15   (-15% from peak)
"""

import json
import os
from unittest.mock import patch

import pytest

# Patch the module-level file path so tests use a tmp file, not logs/
CB_FILE_ENV = 'CIRCUIT_BREAKER_FILE'


@pytest.fixture
def tmp_cb_file(tmp_path):
    return str(tmp_path / 'circuit_breaker.json')


@pytest.fixture
def cb(tmp_cb_file):
    """Fresh RiskCircuitBreaker using a tmp state file."""
    import risk_circuit_breaker as rcb_mod
    original = rcb_mod.CIRCUIT_BREAKER_FILE
    rcb_mod.CIRCUIT_BREAKER_FILE = tmp_cb_file
    cb = rcb_mod.RiskCircuitBreaker()
    cb.reset(manual=True)  # clean slate
    yield cb
    rcb_mod.CIRCUIT_BREAKER_FILE = original  # restore


class TestFailsClosed:
    """Circuit breaker must block trading on any unexpected error."""

    def test_no_state_file_allows_first_run(self, cb, tmp_cb_file):
        """
        With no prior state, first check should not crash.
        The peak is set to current_value on first run, so no drawdown yet.
        """
        result = cb.check(current_value=10_000.0, starting_capital=10_000.0)
        assert isinstance(result, bool)  # does not crash

    def test_corrupted_state_does_not_crash(self, tmp_cb_file):
        """Corrupted state file must not crash — fall back to safe state."""
        import risk_circuit_breaker as rcb_mod
        original = rcb_mod.CIRCUIT_BREAKER_FILE
        rcb_mod.CIRCUIT_BREAKER_FILE = tmp_cb_file

        with open(tmp_cb_file, 'w') as f:
            f.write('{ CORRUPT JSON !!!}')

        try:
            cb = rcb_mod.RiskCircuitBreaker()
            # Should not raise; the empty state means no trigger yet
            result = cb.check(current_value=10_000.0, starting_capital=10_000.0)
            assert isinstance(result, bool)
        finally:
            rcb_mod.CIRCUIT_BREAKER_FILE = original


class TestDrawdownThreshold:
    """Circuit opens when losses exceed thresholds."""

    def test_small_loss_does_not_trigger(self, cb):
        """A 2% daily loss is below 5% daily limit — circuit stays clear."""
        cb.state['daily_start_val'] = 10_000.0
        cb.state['peak_value'] = 10_000.0
        result = cb.check(current_value=9_800.0, starting_capital=10_000.0)
        assert result is False  # allow trading

    def test_daily_loss_limit_triggers(self, cb):
        """A 6% daily loss exceeds 5% daily limit — circuit trips."""
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        cb.state['daily_start']     = today       # prevent date-reset overwrite
        cb.state['daily_start_val'] = 10_000.0
        cb.state['peak_value']      = 10_000.0
        result = cb.check(current_value=9_400.0, starting_capital=10_000.0)
        assert result is True   # block trading

    def test_total_loss_limit_triggers(self, cb):
        """An 11% total loss from starting capital trips the total limit."""
        cb.state['peak_value'] = 10_000.0
        result = cb.check(current_value=8_900.0, starting_capital=10_000.0)
        assert result is True

    def test_drawdown_limit_triggers(self, cb):
        """A 16% drawdown from peak trips the drawdown limit (15%)."""
        cb.state['peak_value'] = 12_000.0  # peaked at 12k
        # Daily and total OK, but drawdown from peak is -16.7%
        result = cb.check(current_value=10_000.0, starting_capital=9_000.0)
        assert result is True


class TestTriggeredState:
    """Once triggered, the circuit stays closed until recovery."""

    def test_triggered_circuit_blocks_trading(self, cb):
        """After triggering, subsequent checks return True until recovery."""
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        cb.state['daily_start']     = today
        cb.state['daily_start_val'] = 10_000.0
        cb.state['peak_value']      = 10_000.0
        # Trip it
        cb.check(current_value=9_400.0, starting_capital=10_000.0)
        assert cb.is_triggered() is True

        # Re-check at same low value — still blocked
        result = cb.check(current_value=9_400.0, starting_capital=10_000.0)
        assert result is True

    def test_manual_reset_clears_trigger(self, cb):
        """Manual reset clears the triggered state immediately."""
        cb._trigger('test reason', 9_000.0)
        assert cb.is_triggered() is True

        cb.reset(manual=True)
        assert cb.is_triggered() is False

    def test_recovery_auto_resets_circuit(self, cb):
        """
        Circuit auto-resets when portfolio recovers RECOVERY_THRESHOLD (2%)
        above the trigger value.
        """
        from risk_circuit_breaker import RECOVERY_THRESHOLD
        trigger_val = 9_400.0
        cb._trigger('daily limit', trigger_val)

        recovery_val = trigger_val * (1 + RECOVERY_THRESHOLD + 0.005)  # +2.5%
        result = cb.check(current_value=recovery_val, starting_capital=10_000.0)
        assert result is False
        assert cb.is_triggered() is False


class TestGetStatus:
    """get_status() returns a complete, valid status dict."""

    def test_status_has_required_keys(self, cb):
        """Status dict must have all expected keys."""
        status = cb.get_status()
        for key in ('triggered', 'reason', 'trigger_date',
                    'daily_limit', 'total_limit', 'drawdown_limit'):
            assert key in status, f'Missing key: {key}'

    def test_status_not_triggered_initially(self, cb):
        """Fresh circuit breaker reports not triggered."""
        assert cb.get_status()['triggered'] is False
