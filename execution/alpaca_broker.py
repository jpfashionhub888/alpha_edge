# execution/alpaca_broker.py

"""
Alpaca Live/Paper Broker Connection.

SECURITY FIX (CRITICAL):
    Hardcoded API key fallbacks removed. If env vars are not set,
    construction fails fast instead of silently using committed keys.

    REGENERATE YOUR KEYS at alpaca.markets — the old hardcoded keys
    have been in git history and must be considered compromised even
    if this file is patched.

Required environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_BASE_URL  (optional, defaults to paper)
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Read an env var; raise with a clear message if missing."""
    val = os.environ.get(name)
    if not val or 'YOUR_API_KEY_HERE' in val:
        raise RuntimeError(
            f"{name} is not set. AlphaEdge will not start without "
            f"valid Alpaca credentials. Set it via config/secrets.env "
            f"or as an environment variable. Never commit it to git."
        )
    return val


class AlpacaBroker:
    """
    Connects to Alpaca for paper or live trading.

    Drop-in replacement for PaperTrader.
    """

    DEFAULT_PAPER_URL = 'https://paper-api.alpaca.markets'

    def __init__(self):
        self.api = None
        self.connected = False
        self.mode = 'paper'
        self._api_key = None
        self._secret_key = None
        self._base_url = None

        try:
            self._api_key    = _require_env('ALPACA_API_KEY')
            self._secret_key = _require_env('ALPACA_SECRET_KEY')
            self._base_url   = os.environ.get(
                'ALPACA_BASE_URL', self.DEFAULT_PAPER_URL
            )

            # Safety: explicit confirmation required for live trading.
            # Setting only the URL is not enough — must also set
            # ALPACA_LIVE_CONFIRM=I_UNDERSTAND.
            if 'paper' not in self._base_url:
                confirm = os.environ.get('ALPACA_LIVE_CONFIRM', '')
                if confirm != 'I_UNDERSTAND':
                    raise RuntimeError(
                        "ALPACA_BASE_URL is set to a live trading URL but "
                        "ALPACA_LIVE_CONFIRM is not set to I_UNDERSTAND. "
                        "Refusing to connect. This protects you from "
                        "accidentally trading real money."
                    )
                self.mode = 'live'
            else:
                self.mode = 'paper'

        except RuntimeError as e:
            # Credentials missing or live-mode not confirmed.
            # Stay disconnected; broker becomes a no-op so the rest of
            # the system can keep running in pure-simulation mode.
            logger.warning("Alpaca not initialised: %s", e)
            self.connected = False
            return

        self._connect()

    def _connect(self):
        """Connect to Alpaca API."""
        # Lazy import so the rest of the system runs without alpaca-trade-api
        try:
            from alpaca_trade_api.rest import REST
        except ImportError:
            logger.warning(
                "alpaca-trade-api not installed; broker disabled. "
                "Run: pip install alpaca-trade-api"
            )
            self.connected = False
            return

        try:
            self.api = REST(
                self._api_key,
                self._secret_key,
                self._base_url,
            )

            account = self.api.get_account()

            logger.info(
                "Alpaca connected (%s mode). Buying power: $%.2f, "
                "Portfolio value: $%.2f, Cash: $%.2f",
                self.mode,
                float(account.buying_power),
                float(account.portfolio_value),
                float(account.cash),
            )
            self.connected = True

        except Exception as e:
            logger.error("Alpaca connection failed: %s", e)
            self.connected = False

    # ------------------------------------------------------------------
    #  Account / positions
    # ------------------------------------------------------------------

    def get_account(self):
        if not self.connected:
            return None
        try:
            account = self.api.get_account()
            return {
                'buying_power'    : float(account.buying_power),
                'portfolio_value' : float(account.portfolio_value),
                'cash'            : float(account.cash),
                'equity'          : float(account.equity),
                'status'          : account.status,
                'mode'            : self.mode,
            }
        except Exception as e:
            logger.error("Get account failed: %s", e)
            return None

    def get_positions(self):
        if not self.connected:
            return {}
        try:
            positions = self.api.list_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    'shares'        : int(pos.qty),
                    'entry_price'   : float(pos.avg_entry_price),
                    'current_price' : float(pos.current_price),
                    'market_value'  : float(pos.market_value),
                    'pnl'           : float(pos.unrealized_pl),
                    'pnl_pct'       : float(pos.unrealized_plpc),
                    'side'          : pos.side,
                }
            return result
        except Exception as e:
            logger.error("Get positions failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    #  Buy / sell
    # ------------------------------------------------------------------

    def buy(self, symbol, amount_dollars, order_type='market'):
        """
        Buy a stock with a dollar amount. Uses fractional shares if
        the broker supports them.
        """
        if not self.connected:
            logger.warning("Cannot buy %s: not connected", symbol)
            return False

        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.info(
                    "Market closed. %s order queued for next open.",
                    symbol,
                )

            order = self.api.submit_order(
                symbol      = symbol,
                notional    = amount_dollars,   # fractional shares
                side        = 'buy',
                type        = order_type,
                time_in_force='day',
            )
            logger.info(
                "Submitted BUY %s for $%.2f (order id %s)",
                symbol, amount_dollars, order.id,
            )
            return True

        except Exception as e:
            logger.error("Buy %s failed: %s", symbol, e)
            return False

    def sell(self, symbol, shares=None, order_type='market'):
        """Sell a position. If shares is None, sells full position."""
        if not self.connected:
            logger.warning("Cannot sell %s: not connected", symbol)
            return False

        try:
            if shares is None:
                # Liquidate full position
                self.api.close_position(symbol)
                logger.info("Closed full position %s", symbol)
                return True

            self.api.submit_order(
                symbol       = symbol,
                qty          = shares,
                side         = 'sell',
                type         = order_type,
                time_in_force='day',
            )
            logger.info("Submitted SELL %s shares of %s", shares, symbol)
            return True

        except Exception as e:
            logger.error("Sell %s failed: %s", symbol, e)
            return False

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------

    def get_summary(self):
        if not self.connected:
            print("   Alpaca: not connected")
            return

        acct = self.get_account()
        if acct is None:
            return

        print("\n" + "=" * 60)
        print(f"  ALPACA ACCOUNT ({self.mode.upper()})")
        print("=" * 60)
        print(f"  Cash:           ${acct['cash']:>12,.2f}")
        print(f"  Portfolio:      ${acct['portfolio_value']:>12,.2f}")
        print(f"  Buying Power:   ${acct['buying_power']:>12,.2f}")
        print(f"  Status:         {acct['status']}")

        positions = self.get_positions()
        if positions:
            print("\n  Open Positions:")
            for sym, p in positions.items():
                emoji = "🟢" if p['pnl'] >= 0 else "🔴"
                print(
                    f"    {emoji} {sym:<6} {p['shares']:>5} shares "
                    f"@ ${p['entry_price']:>7.2f} | "
                    f"now ${p['current_price']:>7.2f} | "
                    f"PnL ${p['pnl']:>+8.2f} ({p['pnl_pct']:>+6.2%})"
                )
        print("=" * 60)
