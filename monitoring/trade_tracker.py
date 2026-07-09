# monitoring/trade_tracker.py
"""
AlphaEdge Trade Tracker

Logs every closed paper trade to a structured JSON file and
fires Telegram milestone alerts at 25 and 50 closed trades.

Used by:
  - alpaca_live.py  (writes a closed trade on every stop/take-profit)
  - command_listener.py  (reads stats for /status)
  - model_watchdog.py    (reads stats for 6h watchdog report)

File: logs/closed_trades.json
Schema:
  {
    "trades": [
      {
        "id": 1,
        "symbol": "AAPL",
        "entry_price": 185.00,
        "exit_price": 192.30,
        "shares": 5.3,
        "pnl_usd": 38.69,
        "pnl_pct": 3.95,
        "reason": "TAKE PROFIT",
        "entry_time": "2026-07-07T16:15:00Z",
        "exit_time": "2026-07-09T14:22:00Z",
        "hold_days": 2
      },
      ...
    ],
    "summary": {
      "total": 3,
      "wins": 2,
      "losses": 1,
      "win_rate": 0.667,
      "total_pnl": 112.50,
      "avg_win": 75.25,
      "avg_loss": -38.00,
      "profit_factor": 3.97,
      "last_updated": "2026-07-09T14:22:00Z"
    }
  }
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TRADES_FILE = Path('logs/closed_trades.json')
MILESTONES  = [10, 25, 50, 75, 100]   # alert at these trade counts


class TradeTracker:
    """
    Thread-safe closed trade logger with milestone Telegram alerts.

    Usage in alpaca_live.py:
        self.trade_tracker = TradeTracker(telegram=self.telegram)

        # When a position closes:
        self.trade_tracker.record(
            symbol='AAPL', entry_price=185.0, exit_price=192.3,
            shares=5.3, reason='TAKE PROFIT',
            entry_time=open_time_iso, exit_time=now_iso,
        )
    """

    def __init__(self, telegram=None):
        self.telegram = telegram
        self._lock    = threading.Lock()
        self._data    = self._load()

    # ── Public API ────────────────────────────────────────────────────

    def record(self,
               symbol: str,
               entry_price: float,
               exit_price: float,
               shares: float,
               reason: str,
               entry_time: str = None,
               exit_time: str  = None) -> None:
        """
        Record one closed trade. Fires milestone alert if needed.

        Parameters
        ----------
        symbol      : ticker e.g. 'AAPL'
        entry_price : average entry price
        exit_price  : fill price on close
        shares      : number of shares (can be fractional)
        reason      : 'TAKE PROFIT' | 'STOP LOSS' | 'TRAILING STOP' | 'MANUAL'
        entry_time  : ISO string of trade open time (optional)
        exit_time   : ISO string of trade close time (optional, defaults to now)
        """
        now     = datetime.now(timezone.utc).isoformat()
        pnl_usd = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Calculate hold days
        hold_days = 0
        if entry_time:
            try:
                et  = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                ext = datetime.fromisoformat((exit_time or now).replace('Z', '+00:00'))
                hold_days = max(0, (ext - et).days)
            except Exception as e:
                logger.debug(f'Hold-days calc skipped: {e}')

        trade = {
            'id'          : len(self._data['trades']) + 1,
            'symbol'      : symbol,
            'entry_price' : round(entry_price, 4),
            'exit_price'  : round(exit_price, 4),
            'shares'      : round(shares, 6),
            'pnl_usd'     : round(pnl_usd, 2),
            'pnl_pct'     : round(pnl_pct, 3),
            'reason'      : reason,
            'entry_time'  : entry_time or '',
            'exit_time'   : exit_time or now,
            'hold_days'   : hold_days,
        }

        with self._lock:
            self._data['trades'].append(trade)
            self._recalc_summary()
            self._save()
            total = self._data['summary']['total']

        logger.info(
            'Trade #%d logged: %s %s $%.2f (%.1f%%)',
            trade['id'], symbol, reason, pnl_usd, pnl_pct
        )

        # Milestone check
        if total in MILESTONES:
            self._fire_milestone(total)

    def get_summary(self) -> dict:
        """Return the current trade summary dict."""
        with self._lock:
            return dict(self._data['summary'])

    def get_trades(self) -> list:
        """Return list of all closed trades."""
        with self._lock:
            return list(self._data['trades'])

    def count(self) -> int:
        """Return total closed trade count."""
        with self._lock:
            return self._data['summary'].get('total', 0)

    # ── Milestone alert ───────────────────────────────────────────────

    def _fire_milestone(self, count: int) -> None:
        summary = self._data['summary']
        wr      = summary.get('win_rate', 0) * 100
        pf      = summary.get('profit_factor', 0)
        pnl     = summary.get('total_pnl', 0)

        if count == 25:
            header = '🎯 MILESTONE: 25 Closed Trades!'
            note   = 'Halfway to live trading eligibility (50 trades).\nReview win rate vs 55% target.'
        elif count == 50:
            header = '🚀 MILESTONE: 50 Closed Trades!'
            note   = 'LIVE TRADING ELIGIBILITY REACHED!\nReview all go/no-go criteria now.'
        elif count == 100:
            header = '💎 MILESTONE: 100 Closed Trades!'
            note   = 'Excellent statistical sample. Strong confidence in system.'
        else:
            header = f'📊 MILESTONE: {count} Closed Trades'
            note   = 'Keep running. System is accumulating evidence.'

        msg = (
            f'{header}\n\n'
            f'Win rate:      {wr:.1f}%  (target ≥55%)\n'
            f'Profit factor: {pf:.2f}  (target ≥1.5)\n'
            f'Total P&L:     ${pnl:+,.2f}\n\n'
            f'{note}\n\n'
            f'Send /status for full portfolio snapshot.'
        )

        if self.telegram and getattr(self.telegram, 'enabled', False):
            try:
                self.telegram.send_message(msg)
                logger.info('Milestone alert sent for %d trades', count)
            except Exception as e:
                logger.warning('Milestone Telegram failed: %s', e)

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
            if TRADES_FILE.exists():
                with open(TRADES_FILE) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning('Could not load trades file: %s', e)
        return {'trades': [], 'summary': self._empty_summary()}

    def _save(self) -> None:
        try:
            tmp = TRADES_FILE.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(self._data, f, indent=2)
            tmp.replace(TRADES_FILE)
        except Exception as e:
            logger.warning('Could not save trades file: %s', e)

    def _recalc_summary(self) -> None:
        trades = self._data['trades']
        wins   = [t for t in trades if t['pnl_usd'] > 0]
        losses = [t for t in trades if t['pnl_usd'] <= 0]
        total  = len(trades)

        total_pnl  = sum(t['pnl_usd'] for t in trades)
        gross_win  = sum(t['pnl_usd'] for t in wins)
        gross_loss = abs(sum(t['pnl_usd'] for t in losses))
        pf         = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

        self._data['summary'] = {
            'total'         : total,
            'wins'          : len(wins),
            'losses'        : len(losses),
            'win_rate'      : round(len(wins) / total, 4) if total > 0 else 0,
            'total_pnl'     : round(total_pnl, 2),
            'avg_win'       : round(gross_win / len(wins), 2) if wins else 0,
            'avg_loss'      : round(-gross_loss / len(losses), 2) if losses else 0,
            'profit_factor' : round(pf, 3) if pf != float('inf') else None,
            'last_updated'  : datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _empty_summary() -> dict:
        return {
            'total': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'total_pnl': 0,
            'avg_win': 0, 'avg_loss': 0,
            'profit_factor': None, 'last_updated': None,
        }


# ── Convenience reader (no telegram needed) ───────────────────────────

def get_trade_stats() -> dict:
    """
    Read trade stats without instantiating the full tracker.
    Safe to call from watchdog, status command, dashboard, etc.
    """
    try:
        if TRADES_FILE.exists():
            with open(TRADES_FILE) as f:
                data = json.load(f)
            return data.get('summary', {})
    except Exception as e:
        logger.debug(f'Trade summary read failed: {e}')
    return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'profit_factor': None}
