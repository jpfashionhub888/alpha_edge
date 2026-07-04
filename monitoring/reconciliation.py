# monitoring/reconciliation.py
"""
Phase 4 — Operational Resilience: Startup Position Reconciliation

Compares broker-reported positions (Alpaca/Gate.io) against the local
paper_trades state file on every bot startup. Discrepancies are:
  - Logged at ERROR level
  - Sent as a Telegram alert
  - Written to logs/reconciliation.log

Why this matters:
  A crash mid-trade can leave the local state out of sync with the broker.
  If the bot restarts thinking it has no position, it may double-buy.
  This module detects and reports that before any new trades are placed.

Usage:
    from monitoring.reconciliation import reconcile_on_startup

    # Call this ONCE at the start of your bot's __init__ or main():
    discrepancies = reconcile_on_startup(
        broker   = self.broker,        # AlpacaBroker or GateIOBroker instance
        log_file = 'logs/paper_trades_stocks_only.json',
        service  = 'alpaca_bot',
    )
    if discrepancies:
        # Bot pauses for human review — never auto-corrects
        logger.error('Reconciliation failed — manual review required')
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RECON_LOG = Path(os.getenv('RECON_LOG', 'logs/reconciliation.log'))

# Tolerance: positions within this dollar value are considered matching
DOLLAR_TOLERANCE = float(os.getenv('RECON_DOLLAR_TOLERANCE', '10.0'))


class PositionReconciler:
    """
    Compares broker positions against local state file.

    Rules:
      - Positions in broker but NOT in local state → PHANTOM (broker has
        a position we don't know about — possible double-buy risk)
      - Positions in local state but NOT in broker  → ORPHAN (we think
        we hold it but broker doesn't — possible data corruption)
      - Positions in both but value differs by > DOLLAR_TOLERANCE → MISMATCH
    """

    def __init__(
        self,
        log_file        : str | Path = 'logs/paper_trades_stocks_only.json',
        service_name    : str        = 'bot',
        dollar_tolerance: float      = DOLLAR_TOLERANCE,
    ):
        self.log_file         = Path(log_file)
        self.service_name     = service_name
        self.dollar_tolerance = dollar_tolerance

    # ── Public ─────────────────────────────────────────────────────────────────

    def reconcile(self, broker) -> list[dict]:
        """
        Run reconciliation. Returns list of discrepancy dicts.
        Empty list = clean state.
        """
        discrepancies = []

        # 1. Load broker positions
        broker_positions = self._fetch_broker_positions(broker)
        if broker_positions is None:
            logger.warning(
                '[Reconcile] Could not fetch broker positions — skipping check'
            )
            return []

        # 2. Load local state
        local_positions = self._load_local_positions()
        if local_positions is None:
            logger.warning(
                '[Reconcile] Could not load local state — skipping check'
            )
            return []

        broker_symbols = set(broker_positions.keys())
        local_symbols  = set(local_positions.keys())

        # 3. Phantom positions (broker has, local doesn't)
        for sym in broker_symbols - local_symbols:
            d = {
                'type'          : 'PHANTOM',
                'symbol'        : sym,
                'broker_value'  : broker_positions[sym],
                'local_value'   : 0.0,
                'description'   : (
                    f'Broker holds {sym} (${broker_positions[sym]:.2f}) '
                    f'but local state has no position'
                ),
            }
            discrepancies.append(d)
            logger.error(
                f'[Reconcile] PHANTOM position: {sym} '
                f'broker=${broker_positions[sym]:.2f} local=none'
            )

        # 4. Orphan positions (local has, broker doesn't)
        for sym in local_symbols - broker_symbols:
            d = {
                'type'          : 'ORPHAN',
                'symbol'        : sym,
                'broker_value'  : 0.0,
                'local_value'   : local_positions[sym],
                'description'   : (
                    f'Local state has {sym} (${local_positions[sym]:.2f}) '
                    f'but broker reports no position'
                ),
            }
            discrepancies.append(d)
            logger.error(
                f'[Reconcile] ORPHAN position: {sym} '
                f'broker=none local=${local_positions[sym]:.2f}'
            )

        # 5. Value mismatches
        for sym in broker_symbols & local_symbols:
            bv   = broker_positions[sym]
            lv   = local_positions[sym]
            diff = abs(bv - lv)
            if diff > self.dollar_tolerance:
                d = {
                    'type'        : 'MISMATCH',
                    'symbol'      : sym,
                    'broker_value': bv,
                    'local_value' : lv,
                    'diff'        : diff,
                    'description' : (
                        f'{sym} value differs: broker=${bv:.2f} '
                        f'local=${lv:.2f} diff=${diff:.2f}'
                    ),
                }
                discrepancies.append(d)
                logger.error(
                    f'[Reconcile] MISMATCH: {sym} '
                    f'broker=${bv:.2f} local=${lv:.2f} diff=${diff:.2f}'
                )

        # 6. Write reconciliation log
        self._write_recon_log(
            broker_positions = broker_positions,
            local_positions  = local_positions,
            discrepancies    = discrepancies,
        )

        if not discrepancies:
            logger.info(
                f'[Reconcile] Clean startup: '
                f'{len(broker_symbols)} broker positions match local state'
            )

        return discrepancies

    # ── Private: data loading ──────────────────────────────────────────────────

    @staticmethod
    def _fetch_broker_positions(broker) -> Optional[dict]:
        """
        Fetch positions from broker.
        Returns {symbol: market_value_usd} or None on failure.
        """
        try:
            # Alpaca broker
            if hasattr(broker, 'get_positions'):
                raw = broker.get_positions()
                if raw is None:
                    return {}
                # Normalise: {symbol: market_value}
                result = {}
                for sym, pos in raw.items():
                    if isinstance(pos, dict):
                        mv = float(pos.get('market_value',
                                   pos.get('value',
                                   pos.get('qty', 0) * pos.get('price', 0))))
                    else:
                        mv = float(pos) if pos else 0.0
                    result[sym.upper()] = mv
                return result

            # Gate.io / crypto broker (Alpaca-style crypto wrapper)
            if hasattr(broker, 'get_balances'):
                raw = broker.get_balances()
                if raw is None:
                    return {}
                result = {}
                for sym, bal in raw.items():
                    if isinstance(bal, dict):
                        mv = float(bal.get('value', bal.get('available', 0)))
                    else:
                        mv = float(bal) if bal else 0.0
                    if mv > 1.0:   # ignore dust < $1
                        result[sym.upper()] = mv
                return result

            # Bybit client: get_all_positions() -> list of position dicts
            # with 'symbol', 'size', 'markPrice'/'avgPrice'
            if hasattr(broker, 'get_all_positions'):
                raw = broker.get_all_positions()
                result = {}
                for pos in (raw or []):
                    sym = pos.get('symbol')
                    if not sym:
                        continue
                    size  = float(pos.get('size', 0))
                    price = float(pos.get('markPrice', pos.get('avgPrice', 0)) or 0)
                    mv    = abs(size * price)
                    if mv > 1.0:
                        result[str(sym).upper()] = mv
                return result

            # Gate.io client: get_spot_balances() -> list of
            # {'currency': ..., 'available': ..., 'locked': ...}
            # Balances are coin amounts, not USD values, so price each
            # non-cash currency against USDT via get_last_price if possible.
            if hasattr(broker, 'get_spot_balances'):
                raw = broker.get_spot_balances()
                result = {}
                for bal in (raw or []):
                    currency = str(bal.get('currency', '')).upper()
                    if not currency or currency in ('USDT', 'USD', 'USDC'):
                        continue  # cash, not a position
                    amount = float(bal.get('available', 0) or 0) + float(bal.get('locked', 0) or 0)
                    if amount <= 0:
                        continue
                    price = None
                    if hasattr(broker, 'get_last_price'):
                        try:
                            price = broker.get_last_price(f'{currency}_USDT')
                        except Exception:
                            price = None
                    if not price:
                        logger.warning(
                            f'[Reconcile] Could not price {currency} balance '
                            f'({amount}) — cannot verify against local state'
                        )
                        continue
                    mv = amount * price
                    if mv > 1.0:
                        result[currency] = mv
                return result

            logger.warning('[Reconcile] Unknown broker type — cannot fetch positions')
            return None

        except Exception as e:
            logger.error(f'[Reconcile] Broker position fetch failed: {e}')
            return None

    def _load_local_positions(self) -> Optional[dict]:
        """
        Load local paper trade positions.
        Returns {symbol: estimated_value} or None on failure.
        """
        if not self.log_file.exists():
            logger.info(f'[Reconcile] No local state file: {self.log_file} — treating as empty')
            return {}

        try:
            with open(self.log_file) as f:
                state = json.load(f)

            positions = state.get('positions', {})
            result    = {}

            for sym, pos in positions.items():
                if isinstance(pos, dict):
                    shares   = float(pos.get('shares', 0))
                    price    = float(
                        pos.get('current_price',
                        pos.get('entry_price', 0))
                    )
                    mv       = shares * price
                else:
                    mv = 0.0
                if mv > 0:
                    result[sym.upper()] = mv

            return result

        except json.JSONDecodeError as e:
            logger.error(f'[Reconcile] Local state file corrupted: {e}')
            return None
        except Exception as e:
            logger.error(f'[Reconcile] Local state load failed: {e}')
            return None

    # ── Private: logging ───────────────────────────────────────────────────────

    def _write_recon_log(
        self,
        broker_positions: dict,
        local_positions : dict,
        discrepancies   : list,
    ) -> None:
        """Append reconciliation result to logs/reconciliation.log."""
        try:
            RECON_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                'timestamp'      : datetime.now(timezone.utc).isoformat(),
                'service'        : self.service_name,
                'broker_symbols' : sorted(broker_positions.keys()),
                'local_symbols'  : sorted(local_positions.keys()),
                'discrepancies'  : len(discrepancies),
                'clean'          : len(discrepancies) == 0,
                'issues'         : discrepancies,
            }
            with open(RECON_LOG, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.debug(f'[Reconcile] Failed to write recon log: {e}')


# ── Convenience function ───────────────────────────────────────────────────────

def reconcile_on_startup(
    broker  ,
    log_file: str | Path = 'logs/paper_trades_stocks_only.json',
    service : str        = 'bot',
) -> list[dict]:
    """
    One-line startup reconciliation.
    Returns list of discrepancies. Sends Telegram alert if any found.

    Example:
        discrepancies = reconcile_on_startup(broker=self.broker, service='alpaca_bot')
    """
    reconciler     = PositionReconciler(log_file=log_file, service_name=service)
    discrepancies  = reconciler.reconcile(broker)

    if discrepancies:
        _send_recon_alert(discrepancies, service)

    return discrepancies


def _send_recon_alert(discrepancies: list, service: str) -> None:
    """Send Telegram alert listing all reconciliation discrepancies."""
    try:
        from monitoring.telegram_bot import TelegramBot
        bot = TelegramBot()
        if not bot.enabled:
            return

        lines = [
            f"WARNING: Position Mismatch on Startup",
            f"",
            f"Service: {service}",
            f"Issues found: {len(discrepancies)}",
            f"",
        ]
        for d in discrepancies[:5]:    # cap at 5 to avoid long message
            lines.append(f"  [{d['type']}] {d['description']}")

        lines += [
            f"",
            f"Action: Do NOT place new trades until resolved.",
            f"Check: logs/reconciliation.log for details",
        ]

        bot.send_message('\n'.join(lines))

    except Exception as e:
        logger.debug(f'[Reconcile] Telegram alert failed: {e}')
