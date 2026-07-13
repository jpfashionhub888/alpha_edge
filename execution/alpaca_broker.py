# execution/alpaca_broker.py
"""
Alpaca Live/Paper Broker Connection — migrated to alpaca-py SDK.

alpaca-trade-api==0.26 is retired by Alpaca (v2 endpoints silently misbehave).
This file uses the official replacement: alpaca-py (pip install alpaca-py).

SDK docs: https://alpaca.markets/sdkdocs/
"""

import os
import logging
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest

logger = logging.getLogger(__name__)

# ── Credentials (set via environment variables) ────────────────────────────
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')

# paper=True  → paper-api.alpaca.markets (default / safe)
# paper=False → live api.alpaca.markets  (real money)
_PAPER_MODE = os.getenv('ALPACA_PAPER', 'true').lower() != 'false'


class AlpacaBroker:
    """
    Connects to Alpaca for real/paper trading via alpaca-py SDK.
    Drop-in replacement for PaperTrader.

    Migration note (from alpaca-trade-api==0.26):
        - TradingClient         replaces REST for account/orders/positions
        - StockHistoricalDataClient replaces REST for latest quote/bar lookups
        - All calls now use typed request objects (MarketOrderRequest, etc.)
    """

    def __init__(self):
        self.client      = None   # TradingClient
        self.data_client = None   # StockHistoricalDataClient
        self.connected   = False
        self.mode        = 'paper' if _PAPER_MODE else 'live'
        self._connect()

    def _connect(self):
        """Connect to Alpaca API using alpaca-py."""
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.warning(
                "Alpaca not configured — set ALPACA_API_KEY / ALPACA_SECRET_KEY env vars"
            )
            return

        try:
            self.client = TradingClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_SECRET_KEY,
                paper      = _PAPER_MODE,
            )
            self.data_client = StockHistoricalDataClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_SECRET_KEY,
            )

            account = self.client.get_account()
            self.connected = True

            logger.info(
                "Alpaca connected (%s) | buying_power=$%.2f | portfolio=$%.2f | cash=$%.2f",
                self.mode,
                float(account.buying_power),
                float(account.portfolio_value),
                float(account.cash),
            )

        except Exception as e:
            logger.error("Alpaca connection failed: %s", e)
            self.connected = False

    # ── Account ─────────────────────────────────────────────────────────────

    def get_account(self):
        """Get account details. Returns dict or None."""
        if not self.connected:
            return None
        try:
            account = self.client.get_account()
            return {
                'buying_power'   : float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash'           : float(account.cash),
                'equity'         : float(getattr(account, 'equity', account.portfolio_value)),
                'status'         : str(account.status),
                'mode'           : self.mode,
            }
        except Exception as e:
            logger.error("get_account failed: %s", e)
            return None

    # ── Positions ────────────────────────────────────────────────────────────

    def get_positions(self):
        """Get all open positions. Returns {symbol: {...}} dict."""
        if not self.connected:
            return {}
        try:
            positions = self.client.get_all_positions()
            return {
                pos.symbol: {
                    'shares'       : float(pos.qty),
                    'entry_price'  : float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value' : float(pos.market_value),
                    'pnl'          : float(pos.unrealized_pl),
                    # unrealized_plpc is a decimal ratio in alpaca-py (0.052 = 5.2%)
                    'pnl_pct'      : float(pos.unrealized_plpc),
                    'side'         : pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                }
                for pos in positions
            }
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            return {}

    # ── Orders — Buy ─────────────────────────────────────────────────────────

    def buy(self, symbol, amount_dollars, order_type='market'):
        """
        Buy a stock with a dollar amount (notional order).
        Uses fractional shares automatically.
        """
        if not self.connected:
            logger.error("Cannot buy %s: broker not connected", symbol)
            return False
        try:
            clock = self.client.get_clock()
            if not clock.is_open:
                logger.warning("Market closed — BUY %s will execute at next open", symbol)

            req   = MarketOrderRequest(
                symbol        = symbol,
                notional      = round(amount_dollars, 2),
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            logger.info(
                "BUY ORDER SUBMITTED | %s | $%.2f | order_id=%s",
                symbol, amount_dollars, order.id,
            )
            return True
        except Exception as e:
            logger.error("buy(%s) failed: %s", symbol, e)
            return False

    def buy_shares(self, symbol, shares, order_type='market'):
        """Buy a specific number of shares."""
        if not self.connected:
            return False
        try:
            req   = MarketOrderRequest(
                symbol        = symbol,
                qty           = shares,
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            logger.info("BUY %s shares %s | order_id=%s", shares, symbol, order.id)
            return True
        except Exception as e:
            logger.error("buy_shares(%s) failed: %s", symbol, e)
            return False

    # ── Orders — Sell ────────────────────────────────────────────────────────

    def sell(self, symbol, shares=None, order_type='market'):
        """
        Sell a stock. If shares is None, closes the entire position.
        """
        if not self.connected:
            return False
        try:
            if shares is None:
                self.client.close_position(symbol_or_asset_id=symbol)
                logger.info("SOLD all %s (close_position)", symbol)
            else:
                req   = MarketOrderRequest(
                    symbol        = symbol,
                    qty           = shares,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
                order = self.client.submit_order(req)
                logger.info("SELL %s shares %s | order_id=%s", shares, symbol, order.id)
            return True
        except Exception as e:
            logger.error("sell(%s) failed: %s", symbol, e)
            return False

    # ── Orders — Stop / Limit ────────────────────────────────────────────────

    def set_stop_loss(self, symbol, stop_price):
        """Set a GTC stop-loss order for an existing position."""
        if not self.connected:
            return False
        try:
            positions = self.get_positions()
            if symbol not in positions:
                logger.warning("set_stop_loss: no position in %s", symbol)
                return False

            shares = positions[symbol]['shares']
            req    = StopOrderRequest(
                symbol        = symbol,
                qty           = shares,
                side          = OrderSide.SELL,
                stop_price    = round(stop_price, 2),
                time_in_force = TimeInForce.GTC,
            )
            order = self.client.submit_order(req)
            logger.info(
                "Stop loss set | %s @ $%.2f | order_id=%s",
                symbol, stop_price, order.id,
            )
            return True
        except Exception as e:
            logger.error("set_stop_loss(%s) failed: %s", symbol, e)
            return False

    def set_take_profit(self, symbol, limit_price):
        """Set a GTC limit take-profit order for an existing position."""
        if not self.connected:
            return False
        try:
            positions = self.get_positions()
            if symbol not in positions:
                return False

            shares = positions[symbol]['shares']
            req    = LimitOrderRequest(
                symbol        = symbol,
                qty           = shares,
                side          = OrderSide.SELL,
                limit_price   = round(limit_price, 2),
                time_in_force = TimeInForce.GTC,
            )
            order = self.client.submit_order(req)
            logger.info(
                "Take profit set | %s @ $%.2f | order_id=%s",
                symbol, limit_price, order.id,
            )
            return True
        except Exception as e:
            logger.error("set_take_profit(%s) failed: %s", symbol, e)
            return False

    # ── Orders — Bracket ─────────────────────────────────────────────────────

    def set_bracket_order(self, symbol, amount_dollars,
                          stop_loss_pct=0.03, take_profit_pct=0.08):
        """
        Buy with automatic stop loss and take profit (bracket order).
        This is the safest way to enter a trade.
        """
        if not self.connected:
            return False
        try:
            price = self._get_latest_price(symbol)
            if price is None or price == 0:
                logger.error("bracket_order(%s): cannot resolve current price", symbol)
                return False

            shares = int(amount_dollars / price)
            if shares == 0:
                logger.warning("bracket_order(%s): cannot afford at $%.2f", symbol, price)
                return False

            stop_price  = round(price * (1 - stop_loss_pct), 2)
            limit_price = round(price * (1 + take_profit_pct), 2)

            req   = MarketOrderRequest(
                symbol        = symbol,
                qty           = shares,
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
                order_class   = OrderClass.BRACKET,
                stop_loss     = StopLossRequest(stop_price=stop_price),
                take_profit   = TakeProfitRequest(limit_price=limit_price),
            )
            self.client.submit_order(req)
            logger.info(
                "BRACKET ORDER | %d %s @ ~$%.2f | SL=$%.2f (-%.0f%%) TP=$%.2f (+%.0f%%)",
                shares, symbol, price,
                stop_price,  stop_loss_pct   * 100,
                limit_price, take_profit_pct * 100,
            )
            return True
        except Exception as e:
            logger.error("bracket_order(%s) failed: %s", symbol, e)
            return False

    # ── Orders — Query / Cancel ───────────────────────────────────────────────

    def get_orders(self, status='open'):
        """Get orders by status ('open', 'closed', 'all')."""
        if not self.connected:
            return []
        try:
            req    = GetOrdersRequest(status=QueryOrderStatus(status))
            orders = self.client.get_orders(req)
            return [
                {
                    'id'       : str(order.id),
                    'symbol'   : order.symbol,
                    'side'     : order.side.value if hasattr(order.side, 'value') else str(order.side),
                    'qty'      : order.qty,
                                        'status'   : order.status.value if hasattr(order.status, 'value') else str(order.status),
                    'filled_qty': order.filled_qty,
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f'get_orders({status}) failed: {e}')
            return []

    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.connected:
            return False
        try:
            self.client.cancel_orders()
            logger.info("All open orders cancelled")
            return True
        except Exception as e:
            logger.error("cancel_all_orders failed: %s", e)
            return False

    # ── Summary ──────────────────────────────────────────────────────────────

    def get_summary(self):
        """Log account summary."""
        account   = self.get_account()
        positions = self.get_positions()

        if account is None:
            logger.error("Cannot get account summary")
            return

        logger.info(
            "=== ALPACA ACCOUNT (%s) === cash=$%.2f | bp=$%.2f | portfolio=$%.2f",
            self.mode.upper(),
            account['cash'], account['buying_power'], account['portfolio_value'],
        )
        if positions:
            total_pnl = sum(p['pnl'] for p in positions.values())
            for symbol, pos in positions.items():
                pnl_pct = pos['pnl_pct'] * 100
                emoji   = "🟢" if pos['pnl'] >= 0 else "🔴"
                logger.info(
                    "  %s %-6s | %.4f sh | entry=$%.2f | now=$%.2f | PnL=$%+.2f (%+.1f%%)",
                    emoji, symbol, pos['shares'],
                    pos['entry_price'], pos['current_price'],
                    pos['pnl'], pnl_pct,
                )
            logger.info("  Total Unrealized PnL: $%+.2f", total_pnl)
        else:
            logger.info("  No open positions")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_latest_price(self, symbol: str) -> 'float | None':
        """
        Fetch latest ask price via StockHistoricalDataClient.
        Falls back to latest bar close if quote unavailable.
        """
        if self.data_client is None:
            return None
        try:
            quotes = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol)
            )
            price = float(quotes[symbol].ask_price)
            if price and price > 0:
                return price
        except Exception as e:
            logger.debug("_get_latest_price(%s): quote failed, trying bar: %s", symbol, e)
        try:
            bars = self.data_client.get_stock_latest_bar(
                StockLatestBarRequest(symbol_or_symbols=symbol)
            )
            return float(bars[symbol].close)
        except Exception as e:
            logger.warning("_get_latest_price(%s): both quote and bar failed: %s", symbol, e)
            return None


# ── Standalone connection test ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Testing Alpaca connection (mode=%s)...", 'paper' if _PAPER_MODE else 'LIVE')
    broker = AlpacaBroker()
    if broker.connected:
        broker.get_summary()
    else:
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables first")
