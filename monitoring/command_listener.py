# monitoring/command_listener.py
"""
AlphaEdge Telegram Command Listener — Emergency Kill Switch

Polls the Telegram Bot API for commands and controls the live
trading bot remotely from your phone.

Commands:
  /pause   — Stop all new trade entries immediately (positions still managed)
  /resume  — Re-enable new trade entries
  /status  — Report current positions, P&L, bot health, mode
  /help    — Show available commands

Architecture:
  - Runs as a daemon thread inside alpaca_live.py
  - Communicates via a shared BotControlState object
  - State also written to logs/bot_control.json for persistence across restarts
  - No external dependencies beyond requests (already used by TelegramBot)
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
CONTROL_FILE   = 'logs/bot_control.json'
POLL_INTERVAL  = 5      # seconds between Telegram getUpdates calls
MAX_BACKOFF    = 120     # max seconds to wait on repeated errors
TIMEOUT        = 15      # HTTP timeout for Telegram API calls


# ── Shared control state ──────────────────────────────────────────────

class BotControlState:
    """
    Thread-safe shared state between CommandListener and trading loop.

    The trading loop checks `state.is_paused` before entering any new
    position. Existing positions are always managed regardless of pause.
    """

    def __init__(self):
        self._lock     = threading.Lock()
        self._paused   = False
        self._reason   = ''
        self._paused_at = None
        self._load()

    # ── Public interface ──────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause(self, reason: str = 'Manual /pause command') -> None:
        with self._lock:
            self._paused    = True
            self._reason    = reason
            self._paused_at = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.warning('Bot PAUSED — %s', reason)

    def resume(self) -> None:
        with self._lock:
            self._paused    = False
            self._reason    = ''
            self._paused_at = None
        self._save()
        logger.info('Bot RESUMED')

    def status_dict(self) -> dict:
        with self._lock:
            return {
                'paused'    : self._paused,
                'reason'    : self._reason,
                'paused_at' : self._paused_at,
            }

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            os.makedirs('logs', exist_ok=True)
            with self._lock:
                data = {
                    'paused'    : self._paused,
                    'reason'    : self._reason,
                    'paused_at' : self._paused_at,
                    'saved_at'  : datetime.now(timezone.utc).isoformat(),
                }
            with open(CONTROL_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning('Could not save control state: %s', e)

    def _load(self) -> None:
        """Restore pause state across bot restarts."""
        try:
            with open(CONTROL_FILE) as f:
                data = json.load(f)
            with self._lock:
                self._paused    = data.get('paused', False)
                self._reason    = data.get('reason', '')
                self._paused_at = data.get('paused_at')
            if self._paused:
                logger.warning(
                    'Bot started in PAUSED state (from previous session). '
                    'Send /resume to re-enable trading.'
                )
        except FileNotFoundError:
            pass   # first run — no state file yet
        except Exception as e:
            logger.warning('Could not load control state: %s', e)


# ── Command listener ──────────────────────────────────────────────────

class CommandListener:
    """
    Polls Telegram for commands and controls the bot state.

    Usage:
        state    = BotControlState()
        listener = CommandListener(state)
        listener.start()   # starts daemon thread

        # In trading loop:
        if state.is_paused:
            print('Bot paused — skipping new entries')
            return
    """

    COMMANDS = {
        '/pause'  : 'pause',
        '/resume' : 'resume',
        '/status' : 'status',
        '/help'   : 'help',
        '/p'      : 'pause',    # shortcuts
        '/r'      : 'resume',
        '/s'      : 'status',
    }

    def __init__(self, state: BotControlState,
                 token: str = None, chat_id: str = None,
                 get_portfolio_fn=None):
        """
        Parameters
        ----------
        state           : BotControlState — shared with trading loop
        token           : Telegram bot token (falls back to env var)
        chat_id         : Telegram chat ID  (falls back to env var)
        get_portfolio_fn: optional callable() → dict with portfolio info
                          for the /status command
        """
        self.state    = state
        self.token    = token    or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id  = chat_id  or os.getenv('TELEGRAM_CHAT_ID', '')
        self.base_url = f'https://api.telegram.org/bot{self.token}'
        self.get_portfolio = get_portfolio_fn

        self._offset   = 0       # Telegram update_id offset for getUpdates
        self._running  = False
        self._thread   = None
        self._errors   = 0       # consecutive error count for backoff

        self.enabled = bool(self.token and self.chat_id
                            and 'YOUR_BOT_TOKEN' not in self.token)

        if not self.enabled:
            logger.warning(
                'CommandListener disabled — TELEGRAM_BOT_TOKEN or '
                'TELEGRAM_CHAT_ID not set'
            )

    # ── Thread control ────────────────────────────────────────────────

    def start(self) -> None:
        """Start polling in a background daemon thread."""
        if not self.enabled:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            name='TelegramCommandListener',
            daemon=True,
        )
        self._thread.start()
        logger.info(
            'Telegram CommandListener started — polling every %ds', POLL_INTERVAL
        )
        self._send(
            '🤖 AlphaEdge bot started.\n\n'
            'Commands:\n'
            '/pause  — halt new entries\n'
            '/resume — re-enable entries\n'
            '/status — portfolio snapshot\n'
            '/help   — show this message'
        )

    def stop(self) -> None:
        self._running = False

    # ── Poll loop ─────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
                self._errors = 0
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                self._errors += 1
                backoff = min(POLL_INTERVAL * (2 ** self._errors), MAX_BACKOFF)
                logger.warning(
                    'CommandListener error #%d: %s — retry in %ds',
                    self._errors, e, backoff
                )
                time.sleep(backoff)

    def _get_updates(self) -> list:
        """Call Telegram getUpdates with long-poll timeout."""
        resp = requests.get(
            f'{self.base_url}/getUpdates',
            params={'offset': self._offset, 'timeout': 4},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f'getUpdates HTTP {resp.status_code}: {resp.text[:200]}')
        data = resp.json()
        if not data.get('ok'):
            raise RuntimeError(f'getUpdates not ok: {data}')
        return data.get('result', [])

    # ── Command dispatcher ────────────────────────────────────────────

    def _handle_update(self, update: dict) -> None:
        update_id = update.get('update_id', 0)
        self._offset = update_id + 1   # advance offset so we don't replay

        msg  = update.get('message') or update.get('edited_message', {})
        text = msg.get('text', '').strip() if msg else ''

        if not text:
            return

        # Security: only accept commands from the configured chat
        from_chat = str(msg.get('chat', {}).get('id', ''))
        if from_chat != str(self.chat_id):
            logger.warning(
                'Rejected command from unauthorized chat %s: %s', from_chat, text
            )
            return

        # Match command (case-insensitive, strip @BotName suffix)
        cmd = text.split()[0].split('@')[0].lower()
        action = self.COMMANDS.get(cmd)

        if action:
            logger.info('Telegram command received: %s → %s', text, action)
            getattr(self, f'_cmd_{action}')()
        else:
            if text.startswith('/'):
                self._send(
                    f'Unknown command: {cmd}\n'
                    'Send /help for available commands.'
                )

    # ── Command handlers ──────────────────────────────────────────────

    def _cmd_pause(self) -> None:
        if self.state.is_paused:
            self._send('⏸️ Bot is already paused.\nSend /resume to re-enable.')
            return
        self.state.pause('Telegram /pause command')
        self._send(
            '⏸️ *BOT PAUSED*\n\n'
            'New trade entries are DISABLED.\n'
            'Existing positions are still monitored and exits will execute.\n\n'
            'Send /resume to re-enable trading.\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}'
        )

    def _cmd_resume(self) -> None:
        if not self.state.is_paused:
            self._send('▶️ Bot is already running.\nNothing to resume.')
            return
        self.state.resume()
        self._send(
            '▶️ *BOT RESUMED*\n\n'
            'New trade entries are ENABLED.\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}'
        )

    def _cmd_status(self) -> None:
        ctrl    = self.state.status_dict()
        mode    = os.getenv('ALPACA_BASE_URL', '')
        mode    = '📄 PAPER' if 'paper' in mode.lower() else '💰 LIVE'
        paused  = '⏸️ PAUSED' if ctrl['paused'] else '▶️ RUNNING'
        paused_since = ''
        if ctrl['paused'] and ctrl['paused_at']:
            paused_since = f"\nPaused since: {ctrl['paused_at'][:16].replace('T',' ')} UTC"

        # Portfolio info (if callback provided)
        portfolio_text = ''
        if self.get_portfolio:
            try:
                pf = self.get_portfolio()
                if pf:
                    portfolio_text = (
                        f"\n\n📊 Portfolio:\n"
                        f"  Value: ${pf.get('value', 0):,.2f}\n"
                        f"  Cash: ${pf.get('cash', 0):,.2f}\n"
                        f"  Positions: {pf.get('n_positions', 0)}\n"
                        f"  Total P&L: {'+' if pf.get('pnl', 0) >= 0 else ''}${pf.get('pnl', 0):,.2f}"
                    )
                    positions = pf.get('positions', {})
                    if positions:
                        portfolio_text += '\n\n📈 Open Positions:'
                        for sym, pos in positions.items():
                            pnl_sym = pos.get('unrealized_pnl', 0)
                            sign = '+' if pnl_sym >= 0 else ''
                            portfolio_text += (
                                f"\n  {sym}: {pos.get('qty', 0)} shares "
                                f"@ ${pos.get('avg_entry', 0):.2f} "
                                f"({sign}${pnl_sym:.2f})"
                            )
            except Exception as e:
                portfolio_text = f'\n\n⚠️ Portfolio data error: {e}'

        # Circuit breaker status
        cb_text = ''
        try:
            with open('logs/circuit_breaker.json') as f:
                cb = json.load(f)
            if cb.get('triggered'):
                cb_text = f"\n\n🚨 CIRCUIT BREAKER: {cb.get('trigger_reason', 'Unknown')}"
        except Exception:
            pass

        msg = (
            f'🤖 AlphaEdge Status\n'
            f'{"=" * 25}\n'
            f'Mode: {mode}\n'
            f'State: {paused}{paused_since}\n'
            f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M ET")}'
            f'{portfolio_text}'
            f'{cb_text}'
        )
        self._send(msg)

    def _cmd_help(self) -> None:
        self._send(
            '🤖 AlphaEdge Commands\n\n'
            '/pause  (or /p)  — Stop all new entries immediately\n'
            '/resume (or /r)  — Re-enable new entries\n'
            '/status (or /s)  — Portfolio snapshot + bot health\n'
            '/help            — Show this message\n\n'
            'Note: /pause does NOT close existing positions.\n'
            'Stops, take-profits, and trailing stops continue to work.'
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _send(self, text: str) -> None:
        """Send a message back to the Telegram chat."""
        try:
            requests.post(
                f'{self.base_url}/sendMessage',
                json={
                    'chat_id'    : self.chat_id,
                    'text'       : text,
                    'parse_mode' : 'Markdown',
                },
                timeout=TIMEOUT,
            )
        except Exception as e:
            logger.warning('CommandListener send failed: %s', e)


# ── Convenience factory ───────────────────────────────────────────────

def start_command_listener(get_portfolio_fn=None) -> tuple:
    """
    Create and start a CommandListener + BotControlState.

    Returns
    -------
    (state, listener) — state is checked in the trading loop,
                        listener runs in background thread.

    Example
    -------
    state, listener = start_command_listener(get_portfolio_fn=my_fn)
    listener.start()

    # In trading loop:
    if state.is_paused:
        logger.info('Bot paused — skipping new entries')
        return
    """
    state    = BotControlState()
    listener = CommandListener(state, get_portfolio_fn=get_portfolio_fn)
    return state, listener
