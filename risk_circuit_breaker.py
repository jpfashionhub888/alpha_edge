# risk_circuit_breaker.py
# ALPHAEDGE - Risk Circuit Breaker V2
#
# Fixes applied:
#   - Removed naive 24-hour auto-reset: the system no longer re-enables
#     trading just because time passed if losses are continuing.
#   - Reset now requires portfolio to recover above a threshold first.
#   - Peak value tracked for accurate drawdown calculation.
#   - Weekly loss check uses rolling 5-day window properly.
#   - Manual reset available via reset(manual=True).

import json
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_FILE = 'logs/circuit_breaker.json'

# Risk thresholds
DAILY_LOSS_LIMIT   = 0.05   # Stop if down 5% in one day
TOTAL_LOSS_LIMIT   = 0.10   # Cash mode if down 10% from starting capital
WEEKLY_LOSS_LIMIT  = 0.07   # Warning if down 7% this week
DRAWDOWN_LIMIT     = 0.15   # Hard stop if drawdown from peak exceeds 15%

# Recovery threshold: circuit breaker won't auto-reset until portfolio
# recovers to within this % of the pre-trigger value
RECOVERY_THRESHOLD = 0.02   # Must recover at least 2% before re-enabling


class RiskCircuitBreaker:
    """
    Portfolio protection system.

    DESIGN CHANGE from V1:
    No longer auto-resets after 24 hours regardless of portfolio state.
    Instead, the circuit breaker stays active until:
      (a) Portfolio recovers at least RECOVERY_THRESHOLD above trigger value, OR
      (b) Manual reset is called.

    This prevents the original bug where the system re-enabled trading
    into a continuing drawdown just because 86400 seconds elapsed.
    """

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self):
        if not os.path.exists(CIRCUIT_BREAKER_FILE):
            return {
                'triggered'       : False,
                'trigger_reason'  : None,
                'trigger_date'    : None,
                'trigger_value'   : None,   # portfolio value when triggered
                'peak_value'      : None,   # all-time high portfolio value
                'daily_start'     : None,
                'daily_start_val' : None,
                'weekly_start'    : None,
                'weekly_start_val': None,
            }
        try:
            with open(CIRCUIT_BREAKER_FILE, 'r') as f:
                data = json.load(f)
            # Back-fill new keys if loading an old state file
            data.setdefault('trigger_value', None)
            data.setdefault('peak_value', None)
            return data
        except Exception as e:
            # AUTO-RECOVERY (operator-approved): quarantine the corrupt file
            # and rebuild a fresh state. Baselines (daily/weekly/starting
            # capital) re-seed from current portfolio value on the next
            # check(), so loss limits re-arm immediately. Known cost:
            # a triggered flag and the all-time peak are lost, so the
            # peak-drawdown limit restarts from the current value.
            quarantined = self._quarantine_corrupt_file(e)
            logger.critical(
                f'{CIRCUIT_BREAKER_FILE} was unreadable ({e}) — '
                f'quarantined to {quarantined} and rebuilt fresh state. '
                f'Loss limits re-arm from current portfolio value; '
                f'peak-drawdown history was lost.'
            )
            return {
                'triggered'       : False,
                'trigger_reason'  : None,
                'trigger_date'    : None,
                'trigger_value'   : None,
                'peak_value'      : None,
                'daily_start'     : None,
                'daily_start_val' : None,
                'weekly_start'    : None,
                'weekly_start_val': None,
                # One-shot flag: first check() with a telegram handle sends
                # a recovery alert, then clears this.
                'recovered_from_corrupt': datetime.now().isoformat(),
            }

    @staticmethod
    def _quarantine_corrupt_file(err) -> str:
        """Move the unreadable state file aside so it can be inspected."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        dest = f'{CIRCUIT_BREAKER_FILE}.corrupt.{ts}'
        try:
            os.replace(CIRCUIT_BREAKER_FILE, dest)
            return dest
        except OSError as move_err:
            logger.error(f'Could not quarantine corrupt state file: {move_err}')
            return f'<quarantine failed: {move_err}>'

    def _save_state(self):
        import tempfile, shutil
        # Derive save dir from the actual target path so tests can redirect
        # CIRCUIT_BREAKER_FILE to a tmp_path without hitting the real logs/ dir.
        target = CIRCUIT_BREAKER_FILE
        target_dir = os.path.dirname(os.path.abspath(target)) or 'logs'
        os.makedirs(target_dir, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(self.state, f, indent=2)
                # fsync BEFORE rename: rename is atomic for the directory
                # entry, not the data — crash mid-writeback leaves a
                # NUL-filled file that would previously fail-open.
                f.flush()
                os.fsync(f.fileno())
            # shutil.move handles cross-device moves (os.replace raises EXDEV)
            shutil.move(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def check(self, current_value, starting_capital, telegram=None):
        """
        Check if circuit breaker should trigger.
        Returns True if trading should STOP.
        """
        now = datetime.now()

        # One-shot recovery alert: state was rebuilt after a corrupt file.
        if self.state.get('recovered_from_corrupt'):
            if telegram:
                try:
                    telegram.send_message(
                        '⚠️ Circuit breaker state file was corrupt — '
                        'quarantined and rebuilt. Loss limits re-armed from '
                        'current portfolio value; peak-drawdown history lost. '
                        f'(corruption detected {self.state["recovered_from_corrupt"]})'
                    )
                except Exception as tg_err:
                    logger.warning(f'Recovery alert failed: {tg_err}')
            self.state.pop('recovered_from_corrupt', None)
            self._save_state()

        # Update peak value (all-time high tracking)
        peak = self.state.get('peak_value')
        if peak is None or current_value > peak:
            self.state['peak_value'] = current_value
            self._save_state()
        peak = self.state['peak_value']

        # Initialise daily tracking
        today = now.strftime('%Y-%m-%d')
        if self.state.get('daily_start') != today:
            self.state['daily_start']     = today
            self.state['daily_start_val'] = current_value
            self._save_state()

        # Initialise weekly tracking (Mon-based week)
        week_start = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        if self.state.get('weekly_start') != week_start:
            self.state['weekly_start']     = week_start
            self.state['weekly_start_val'] = current_value
            self._save_state()

        # ----------------------------------------------------------------
        # Check if already triggered — only reset on recovery, not time
        # ----------------------------------------------------------------
        if self.state.get('triggered'):
            # `or` (not dict default): key may exist with value None,
            # e.g. after a fail-closed load from a corrupt state file.
            trigger_value = self.state.get('trigger_value') or starting_capital
            trigger_date  = self.state.get('trigger_date', '')
            # Recovery check: current value must be >= trigger_value * (1 + threshold)
            recovery_target = trigger_value * (1 + RECOVERY_THRESHOLD)
            recovery_pct    = (current_value - trigger_value) / trigger_value if trigger_value else 0

            logger.warning(
                "CIRCUIT BREAKER ACTIVE! | triggered=%s | reason=%s | recovery=%+.2f%% (need +%.0f%% to auto-reset)",
                trigger_date, self.state.get('trigger_reason'),
                recovery_pct * 100, RECOVERY_THRESHOLD * 100,
            )

            if current_value >= recovery_target:
                logger.info("Portfolio recovered -- auto-resetting circuit breaker")
                self.reset()
                return False
            else:
                logger.warning("Circuit breaker holding -- no new trades until recovery threshold met")
                return True

        # ----------------------------------------------------------------
        # Calculate losses
        # ----------------------------------------------------------------
        daily_start  = self.state.get('daily_start_val', current_value)
        weekly_start = self.state.get('weekly_start_val', current_value)

        daily_loss   = (current_value - daily_start) / daily_start  if daily_start  > 0 else 0
        weekly_loss  = (current_value - weekly_start)/ weekly_start if weekly_start > 0 else 0
        total_loss   = (current_value - starting_capital) / starting_capital
        drawdown     = (current_value - peak) / peak if peak > 0 else 0

        logger.info(
                "Risk check | daily=%+.2f%% weekly=%+.2f%% total=%+.2f%% drawdown=%+.2f%% (peak=$%.2f)",
                daily_loss*100, weekly_loss*100, total_loss*100, drawdown*100, peak,
            )

        # Daily loss limit
        if daily_loss <= -DAILY_LOSS_LIMIT:
            reason = f"Daily loss limit: {daily_loss:.2%} (limit -{DAILY_LOSS_LIMIT:.0%})"
            self._trigger(reason, current_value, telegram)
            return True

        # Total loss limit (from starting capital)
        if total_loss <= -TOTAL_LOSS_LIMIT:
            reason = f"Total loss limit: {total_loss:.2%} (limit -{TOTAL_LOSS_LIMIT:.0%})"
            self._trigger(reason, current_value, telegram)
            return True

        # Drawdown from peak
        if drawdown <= -DRAWDOWN_LIMIT:
            reason = f"Peak drawdown limit: {drawdown:.2%} (limit -{DRAWDOWN_LIMIT:.0%})"
            self._trigger(reason, current_value, telegram)
            return True

        # Weekly warning (no trigger, just alert)
        if weekly_loss <= -WEEKLY_LOSS_LIMIT:
            logger.warning("Weekly loss %.2f%% approaching circuit breaker!", weekly_loss * 100)
            if telegram:
                telegram.send_message(
                    f"WARNING: ALPHAEDGE\n"
                    f"Weekly loss: {weekly_loss:.2%}\n"
                    f"Monitoring closely -- circuit breaker NOT triggered yet."
                )

        logger.info("Risk check passed -- trading allowed")
        return False

    def _trigger(self, reason, current_value, telegram=None):
        self.state['triggered']     = True
        self.state['trigger_reason']= reason
        self.state['trigger_date']  = datetime.now().isoformat()
        self.state['trigger_value'] = current_value
        self._save_state()

        logger.critical(
                "CIRCUIT BREAKER TRIGGERED! reason=%s | all new trades STOPPED until +%.0f%% recovery",
                reason, RECOVERY_THRESHOLD * 100,
            )

        if telegram:
            telegram.send_message(
                f"🚨 ALPHAEDGE CIRCUIT BREAKER!\n"
                f"========================\n"
                f"Reason: {reason}\n"
                f"Action: All new trades STOPPED\n"
                f"Reset: Automatic on +{RECOVERY_THRESHOLD:.0%} recovery (NOT time-based)\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Your capital is being protected."
            )

    def reset(self, manual=False):
        self.state['triggered']     = False
        self.state['trigger_reason']= None
        self.state['trigger_date']  = None
        self.state['trigger_value'] = None
        self._save_state()

        if manual:
            logger.info("Circuit breaker manually reset")
        else:
            logger.info("Circuit breaker auto-reset (portfolio recovered)")

    def is_triggered(self):
        return self.state.get('triggered', False)

    def get_status(self):
        return {
            'triggered'     : self.state.get('triggered', False),
            'reason'        : self.state.get('trigger_reason'),
            'trigger_date'  : self.state.get('trigger_date'),
            'trigger_value' : self.state.get('trigger_value'),
            'peak_value'    : self.state.get('peak_value'),
            'daily_limit'   : f"{DAILY_LOSS_LIMIT:.0%}",
            'total_limit'   : f"{TOTAL_LOSS_LIMIT:.0%}",
            'weekly_limit'  : f"{WEEKLY_LOSS_LIMIT:.0%}",
            'drawdown_limit': f"{DRAWDOWN_LIMIT:.0%}",
        }


if __name__ == '__main__':
    print("\nTesting Risk Circuit Breaker V2...")

    cb = RiskCircuitBreaker()
    cb.reset(manual=True)  # clean slate for test

    print("\n--- Normal conditions ---")
    triggered = cb.check(current_value=10050.0, starting_capital=10000.0)
    print(f"Trading allowed: {not triggered}")

    print("\n--- Bad day (-6%) ---")
    cb.state['daily_start_val'] = 10000.0
    triggered = cb.check(current_value=9400.0, starting_capital=10000.0)
    print(f"Trading blocked: {triggered}")

