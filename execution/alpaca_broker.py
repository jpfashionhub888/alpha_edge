# execution/alpaca_broker.py

"""
Alpaca Live/Paper Broker Connection.
Executes real trades on your Alpaca account.
"""

import os
import logging
from datetime import datetime
from alpaca_trade_api.rest import REST, TimeFrame

logger = logging.getLogger(__name__)

# ==========================================
# PUT YOUR ALPACA CREDENTIALS HERE
# ==========================================
ALPACA_API_KEY = os.getenv(
    'ALPACA_API_KEY',
    'PK5TI3TNXIJLPTQ46UUPZIUHXM'
)
ALPACA_SECRET_KEY = os.getenv(
    'ALPACA_SECRET_KEY',
    'DUvwnAtnL49fZQ6RwiRDeXzb1EoNiVYs1TFvKG2w2M1A'
)

# IMPORTANT: Use paper URL for testing first
# Paper: https://paper-api.alpaca.markets
# Live:  https://api.alpaca.markets
ALPACA_BASE_URL = os.getenv(
    'ALPACA_BASE_URL',
    'https://paper-api.alpaca.markets'
)


class AlpacaBroker:
    """
    Connects to Alpaca for real/paper trading.
    Drop-in replacement for PaperTrader.
    """

    def __init__(self):
        self.api = None
        self.connected = False
        self.mode = 'paper'

        if 'paper' in ALPACA_BASE_URL:
            self.mode = 'paper'
        else:
            self.mode = 'live'

        self._connect()

    def _connect(self):
        """Connect to Alpaca API."""

        if 'YOUR_API_KEY_HERE' in ALPACA_API_KEY:
            print(
                "   ⚠️ Alpaca not configured."
                " Set keys in alpaca_broker.py"
            )
            return

        try:
            self.api = REST(
                ALPACA_API_KEY,
                ALPACA_SECRET_KEY,
                ALPACA_BASE_URL
            )

            account = self.api.get_account()

            print(f"\n   ✅ Alpaca connected ({self.mode} mode)")
            print(f"   💰 Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   📈 Cash: ${float(account.cash):,.2f}")

            self.connected = True

        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            print(f"   ❌ Alpaca connection failed: {e}")
            self.connected = False

    def get_account(self):
        """Get account details."""

        if not self.connected:
            return None

        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'status': account.status,
                'mode': self.mode,
            }
        except Exception as e:
            logger.error(f"Get account failed: {e}")
            return None

    def get_positions(self):
        """Get all open positions."""

        if not self.connected:
            return {}

        try:
            positions = self.api.list_positions()
            result = {}

            for pos in positions:
                result[pos.symbol] = {
                    'shares': int(pos.qty),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'pnl': float(pos.unrealized_pl),
                    'pnl_pct': float(pos.unrealized_plpc),
                    'side': pos.side,
                }

            return result

        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            return {}

    def buy(self, symbol, amount_dollars,
            order_type='market'):
        """
        Buy a stock with a dollar amount.
        Uses fractional shares if needed.
        """

        if not self.connected:
            print(f"   ❌ Cannot buy {symbol}: not connected")
            return False

        try:
            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                print(
                    f"   ⚠️ Market closed."
                    f" Order will execute at next open."
                )

            # Submit order
            order = self.api.submit_order(
                symbol=symbol,
                notional=round(amount_dollars, 2),
                side='buy',
                type=order_type,
                time_in_force='day'
            )

            print(
                f"   🟢 BUY ORDER SUBMITTED"
                f" | {symbol}"
                f" | ${amount_dollars:.2f}"
                f" | Order ID: {order.id}"
            )

            return True

        except Exception as e:
            logger.error(f"Buy {symbol} failed: {e}")
            print(f"   ❌ Buy {symbol} failed: {e}")
            return False

    def buy_shares(self, symbol, shares,
                   order_type='market'):
        """Buy specific number of shares."""

        if not self.connected:
            return False

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type=order_type,
                time_in_force='day'
            )

            print(
                f"   🟢 BUY {shares} {symbol}"
                f" | Order ID: {order.id}"
            )

            return True

        except Exception as e:
            logger.error(f"Buy {symbol} failed: {e}")
            print(f"   ❌ Buy {symbol} failed: {e}")
            return False

    def sell(self, symbol, shares=None,
             order_type='market'):
        """
        Sell a stock. If shares is None, sells entire position.
        """

        if not self.connected:
            return False

        try:
            if shares is None:
                # Sell entire position
                self.api.close_position(symbol)
                print(
                    f"   🔴 SOLD all {symbol}"
                )
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type=order_type,
                    time_in_force='day'
                )
                print(
                    f"   🔴 SELL {shares} {symbol}"
                    f" | Order ID: {order.id}"
                )

            return True

        except Exception as e:
            logger.error(f"Sell {symbol} failed: {e}")
            print(f"   ❌ Sell {symbol} failed: {e}")
            return False

    def set_stop_loss(self, symbol, stop_price):
        """Set a stop loss order."""

        if not self.connected:
            return False

        try:
            positions = self.get_positions()

            if symbol not in positions:
                print(f"   ⚠️ No position in {symbol}")
                return False

            shares = positions[symbol]['shares']

            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                type='stop',
                stop_price=round(stop_price, 2),
                time_in_force='gtc'
            )

            print(
                f"   🛑 Stop loss set for {symbol}"
                f" at ${stop_price:.2f}"
                f" | Order ID: {order.id}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Stop loss for {symbol} failed: {e}"
            )
            return False

    def set_take_profit(self, symbol, limit_price):
        """Set a take profit order."""

        if not self.connected:
            return False

        try:
            positions = self.get_positions()

            if symbol not in positions:
                return False

            shares = positions[symbol]['shares']

            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                type='limit',
                limit_price=round(limit_price, 2),
                time_in_force='gtc'
            )

            print(
                f"   🎯 Take profit set for {symbol}"
                f" at ${limit_price:.2f}"
                f" | Order ID: {order.id}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Take profit for {symbol} failed: {e}"
            )
            return False

    def set_bracket_order(self, symbol, amount_dollars,
                          stop_loss_pct=0.03,
                          take_profit_pct=0.08):
        """
        Buy with automatic stop loss and take profit.
        This is the safest way to enter a trade.
        """

        if not self.connected:
            return False

        try:
            # Get current price
            quote = self.api.get_latest_quote(symbol)
            price = float(quote.ap)

            if price == 0:
                bars = self.api.get_latest_bar(symbol)
                price = float(bars.c)

            shares = int(amount_dollars / price)

            if shares == 0:
                print(f"   ⚠️ Cannot afford {symbol}")
                return False

            stop_price = round(
                price * (1 - stop_loss_pct), 2
            )
            limit_price = round(
                price * (1 + take_profit_pct), 2
            )

            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='bracket',
                stop_loss={
                    'stop_price': stop_price
                },
                take_profit={
                    'limit_price': limit_price
                }
            )

            print(
                f"   🟢 BRACKET ORDER: {shares} {symbol}"
                f" @ ~${price:.2f}"
            )
            print(
                f"      Stop Loss: ${stop_price:.2f}"
                f" ({stop_loss_pct:.0%})"
            )
            print(
                f"      Take Profit: ${limit_price:.2f}"
                f" ({take_profit_pct:.0%})"
            )

            return True

        except Exception as e:
            logger.error(
                f"Bracket order for {symbol} failed: {e}"
            )
            print(f"   ❌ Bracket order failed: {e}")
            return False

    def get_orders(self, status='open'):
        """Get all orders."""

        if not self.connected:
            return []

        try:
            orders = self.api.list_orders(status=status)
            result = []

            for order in orders:
                result.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': order.qty,
                    'type': order.type,
                    'status': order.status,
                    'submitted': str(order.submitted_at),
                })

            return result

        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []

    def cancel_all_orders(self):
        """Cancel all open orders."""

        if not self.connected:
            return False

        try:
            self.api.cancel_all_orders()
            print("   ✅ All open orders cancelled")
            return True

        except Exception as e:
            logger.error(f"Cancel orders failed: {e}")
            return False

    def get_summary(self):
        """Print account summary."""

        account = self.get_account()
        positions = self.get_positions()

        if account is None:
            print("   ❌ Cannot get account summary")
            return

        print("\n" + "="*60)
        print(f"ALPACA ACCOUNT ({self.mode.upper()} MODE)")
        print("="*60)

        print(f"   Cash: ${account['cash']:,.2f}")
        print(f"   Buying Power: ${account['buying_power']:,.2f}")
        print(f"   Portfolio: ${account['portfolio_value']:,.2f}")

        if positions:
            print(f"\n   Open Positions ({len(positions)}):")

            total_pnl = 0
            for symbol, pos in positions.items():
                pnl = pos['pnl']
                pnl_pct = pos['pnl_pct'] * 100
                total_pnl += pnl

                emoji = "🟢" if pnl >= 0 else "🔴"

                print(
                    f"      {emoji} {symbol:6s}"
                    f" | {pos['shares']} shares"
                    f" | Entry: ${pos['entry_price']:.2f}"
                    f" | Now: ${pos['current_price']:.2f}"
                    f" | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
                )

            print(f"\n   Total Unrealized PnL: ${total_pnl:+,.2f}")
        else:
            print("\n   No open positions")

        print("="*60)


if __name__ == "__main__":
    print("\n🔌 Testing Alpaca Connection...")
    broker = AlpacaBroker()

    if broker.connected:
        broker.get_summary()
    else:
        print("\n   Set your API keys in the file first")