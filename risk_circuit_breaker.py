# risk_circuit_breaker.py
# ALPHAEDGE - Risk Circuit Breaker
# Stops trading if portfolio drops too much
# Protects capital during bad days/crashes

import json
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_FILE = 'logs/circuit_breaker.json'

# Risk thresholds
DAILY_LOSS_LIMIT     = 0.05   # Stop if down 5% in one day
TOTAL_LOSS_LIMIT     = 0.10   # Cash mode if down 10% total
WEEKLY_LOSS_LIMIT    = 0.07   # Warning if down 7% this week


class RiskCircuitBreaker:
    """
    Portfolio protection system.
    Automatically stops trading during bad periods.
    Like a fuse box for your trading account!
    """

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self):
        """Load circuit breaker state."""
        if not os.path.exists(CIRCUIT_BREAKER_FILE):
            return {
                'triggered'      : False,
                'trigger_reason' : None,
                'trigger_date'   : None,
                'daily_start'    : None,
                'daily_start_val': None,
                'weekly_start'   : None,
                'weekly_start_val': None,
            }
        try:
            with open(CIRCUIT_BREAKER_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self):
        """Save circuit breaker state."""
        os.makedirs('logs', exist_ok=True)
        with open(CIRCUIT_BREAKER_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def check(self, current_value, starting_capital, telegram=None):
        """
        Check if circuit breaker should trigger.
        Returns True if trading should STOP.
        """
        now = datetime.now()

        # Initialize daily tracking
        today = now.strftime('%Y-%m-%d')
        if self.state.get('daily_start') != today:
            self.state['daily_start']     = today
            self.state['daily_start_val'] = current_value
            self._save_state()

        # Initialize weekly tracking
        week_start = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        if self.state.get('weekly_start') != week_start:
            self.state['weekly_start']      = week_start
            self.state['weekly_start_val']  = current_value
            self._save_state()

        # Check if already triggered
        if self.state.get('triggered'):
            trigger_date = self.state.get('trigger_date', '')
            print(f"\n   ⚠️ CIRCUIT BREAKER ACTIVE!")
            print(f"   Triggered: {trigger_date}")
            print(f"   Reason: {self.state.get('trigger_reason')}")
            print(f"   Status: No new trades allowed")

            # Auto-reset after 24 hours
            try:
                trigger_dt = datetime.fromisoformat(trigger_date)
                if (now - trigger_dt).total_seconds() > 86400:
                    print("   Auto-resetting after 24 hours...")
                    self.reset()
                    return False
            except Exception:
                pass

            return True

        # Calculate losses
        daily_start  = self.state.get('daily_start_val', current_value)
        weekly_start = self.state.get('weekly_start_val', current_value)

        daily_loss   = (current_value - daily_start) / daily_start if daily_start > 0 else 0
        weekly_loss  = (current_value - weekly_start) / weekly_start if weekly_start > 0 else 0
        total_loss   = (current_value - starting_capital) / starting_capital

        print(f"\n   Risk Check:")
        print(f"   Daily P&L:   {daily_loss:+.2%}")
        print(f"   Weekly P&L:  {weekly_loss:+.2%}")
        print(f"   Total P&L:   {total_loss:+.2%}")

        # Check daily loss limit
        if daily_loss <= -DAILY_LOSS_LIMIT:
            reason = f"Daily loss limit hit: {daily_loss:.2%} (limit: -{DAILY_LOSS_LIMIT:.0%})"
            self._trigger(reason, telegram)
            return True

        # Check total loss limit
        if total_loss <= -TOTAL_LOSS_LIMIT:
            reason = f"Total loss limit hit: {total_loss:.2%} (limit: -{TOTAL_LOSS_LIMIT:.0%})"
            self._trigger(reason, telegram)
            return True

        # Warning for weekly loss
        if weekly_loss <= -WEEKLY_LOSS_LIMIT:
            print(f"   ⚠️ WARNING: Weekly loss {weekly_loss:.2%} approaching limit!")
            if telegram:
                telegram.send_message(
                    f"⚠️ ALPHAEDGE WARNING\n"
                    f"Weekly loss: {weekly_loss:.2%}\n"
                    f"Approaching circuit breaker limit!\n"
                    f"Monitoring closely..."
                )

        print(f"   ✅ Risk check passed - Trading allowed")
        return False

    def _trigger(self, reason, telegram=None):
        """Trigger the circuit breaker."""
        self.state['triggered']       = True
        self.state['trigger_reason']  = reason
        self.state['trigger_date']    = datetime.now().isoformat()
        self._save_state()

        print(f"\n   🚨 CIRCUIT BREAKER TRIGGERED!")
        print(f"   Reason: {reason}")
        print(f"   All new trades STOPPED!")
        print(f"   Will auto-reset in 24 hours")

        if telegram:
            telegram.send_message(
                f"🚨 ALPHAEDGE CIRCUIT BREAKER!\n"
                f"========================\n"
                f"Reason: {reason}\n"
                f"Action: All new trades STOPPED\n"
                f"Reset: Automatic in 24 hours\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Your capital is being protected."
            )

    def reset(self, manual=False):
        """Reset the circuit breaker."""
        self.state['triggered']      = False
        self.state['trigger_reason'] = None
        self.state['trigger_date']   = None
        self._save_state()

        if manual:
            print("   Circuit breaker manually reset ✅")
        else:
            print("   Circuit breaker auto-reset after 24 hours ✅")

    def is_triggered(self):
        """Check if circuit breaker is active."""
        return self.state.get('triggered', False)

    def get_status(self):
        """Get current circuit breaker status."""
        return {
            'triggered'    : self.state.get('triggered', False),
            'reason'       : self.state.get('trigger_reason'),
            'trigger_date' : self.state.get('trigger_date'),
            'daily_limit'  : f"{DAILY_LOSS_LIMIT:.0%}",
            'total_limit'  : f"{TOTAL_LOSS_LIMIT:.0%}",
            'weekly_limit' : f"{WEEKLY_LOSS_LIMIT:.0%}",
        }


if __name__ == '__main__':
    print("\nTesting Risk Circuit Breaker...")
    cb = RiskCircuitBreaker()

    # Test normal conditions
    print("\n--- Normal Market ---")
    triggered = cb.check(
        current_value    = 10050.0,
        starting_capital = 10000.0,
    )
    print(f"Trading allowed: {not triggered}")

    # Test daily loss scenario
    print("\n--- Bad Day Scenario (-6%) ---")
    cb.state['daily_start_val'] = 10000.0
    triggered = cb.check(
        current_value    = 9400.0,
        starting_capital = 10000.0,
    )
    print(f"Trading allowed: {not triggered}")

    # Reset for next test
    cb.reset()

    # Test total loss scenario
    print("\n--- Total Loss Scenario (-12%) ---")
    triggered = cb.check(
        current_value    = 8800.0,
        starting_capital = 10000.0,
    )
    print(f"Trading allowed: {not triggered}")